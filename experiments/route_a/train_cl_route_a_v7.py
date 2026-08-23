"""
ManiSkill3 CL - Route A v7
  Phase 0: Load OFFICIAL PokeCube-v1 checkpoint, evaluate
  Phase 1: Transfer feature extractor, REINITIALIZE action head, train on PickCube

  Why v7:
  v5/v6 fine-tuned the ENTIRE network from PokeCube -> PickCube.
  PokeCube trains a POKING behavior; PickCube requires GRASPING.
  The pretrained action head actively fights the new skill.
  v5: high LR -> policy collapse (std=0.195)
  v6: low LR  -> frozen, can't even reach cube (avg_R=1.15)

  v7 fix:
  - Keep pretrained feature extractor (robot state representation transfers)
  - REINITIALIZE action_net + log_std (fresh head for new motor skill)
  - Reinitialize value_net too (different reward structure)
  - Higher LR (3e-4 -> 3e-5 linear) so new head can actually learn
  - ent_coef=0.01 for exploration, target_kl=0.02 for stability
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_conda = r"E:\My_programs\anaconda\envs\diffcl10\Library\bin"
if _conda not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda + os.pathsep + os.environ.get("PATH", "")

import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import get_linear_fn

# ═══════════════════════════════════════════════════════════════════
#  Fix terminal_observation shape
# ═══════════════════════════════════════════════════════════════════
class FixedDummyVecEnv(DummyVecEnv):
    def step_wait(self):
        results = super().step_wait()
        obs, rewards, dones, infos = results
        for info in infos:
            if "terminal_observation" in info:
                tobs = info["terminal_observation"]
                if isinstance(tobs, torch.Tensor):
                    tobs = tobs.cpu().numpy()
                info["terminal_observation"] = np.asarray(tobs, dtype=np.float32).flatten()
        return obs, rewards, dones, infos

def _patched_obs_to_tensor(self, observation):
    if not isinstance(observation, np.ndarray):
        observation = np.array(observation)
    if observation.ndim == 1:
        observation = observation.reshape(1, -1)
    return torch.as_tensor(observation, dtype=torch.float32).to(self.device), True

BasePolicy.obs_to_tensor = _patched_obs_to_tensor

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
TASK_SEQUENCE = ["PokeCube-v1", "PickCube-v1"]
TASK_CONFIGS = {
    "PokeCube-v1":  {"raw_obs_dim": 54, "act_dim": 8},
    "PickCube-v1":  {"raw_obs_dim": 42, "act_dim": 8},
}
UNIFIED_OBS_DIM = 54
CONTROL_MODE = "pd_joint_delta_pos"

TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 4
NET_ARCH_PI = [256, 256, 256]
NET_ARCH_VF = [256, 256, 256]
ACTIVATION_FN = nn.Tanh

# Phase 1 — fresh head needs higher LR to learn
LR_START = 3e-4          # official rate, so new head can learn
LR_END = 3e-5            # decay to 1/10 for fine-tuning
ENT_COEF = 0.01          # exploration for grasp
N_STEPS = 2048
BATCH_SIZE = 128
N_EPOCHS = 8
GAMMA = 0.85
CLIP_RANGE = 0.2
TARGET_KL = 0.02         # strict KL to prevent collapse
MAX_GRAD_NORM = 0.5
GAE_LAMBDA = 0.95

EVAL_FREQ = 50_000
EVAL_EPS = 20

OFFICIAL_CKPT = os.path.expanduser(
    r"~\.maniskill\demos\PokeCube-v1\rl\ppo_pd_joint_delta_pos_ckpt.pt"
)
CKPT_SAVE_DIR = r"E:\munichi\cl_checkpoints"
RESULT_DIR = r"E:\munichi\cl_results"
os.makedirs(CKPT_SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  CHECKPOINT CONVERTER
# ═══════════════════════════════════════════════════════════════════
def convert_official_to_sb3(official_ckpt_path):
    raw = torch.load(official_ckpt_path, map_location="cpu", weights_only=False)
    print(f"  Loaded: {official_ckpt_path}")
    print(f"  actor_logstd: {raw['actor_logstd'].flatten().tolist()}")

    sb3_sd = {}
    for src, dst in [(0, 0), (2, 2), (4, 4)]:
        sb3_sd[f"mlp_extractor.policy_net.{dst}.weight"] = raw[f"actor_mean.{src}.weight"].clone()
        sb3_sd[f"mlp_extractor.policy_net.{dst}.bias"] = raw[f"actor_mean.{src}.bias"].clone()
    sb3_sd["action_net.weight"] = raw["actor_mean.6.weight"].clone()
    sb3_sd["action_net.bias"] = raw["actor_mean.6.bias"].clone()
    sb3_sd["log_std"] = raw["actor_logstd"].clone().flatten()
    for src, dst in [(0, 0), (2, 2), (4, 4)]:
        sb3_sd[f"mlp_extractor.value_net.{dst}.weight"] = raw[f"critic.{src}.weight"].clone()
        sb3_sd[f"mlp_extractor.value_net.{dst}.bias"] = raw[f"critic.{src}.bias"].clone()
    sb3_sd["value_net.weight"] = raw["critic.6.weight"].clone()
    sb3_sd["value_net.bias"] = raw["critic.6.bias"].clone()

    print(f"  Converted {len(sb3_sd)} keys")
    return sb3_sd


def reinit_head_in_state_dict(sd, act_dim=8):
    """Reinitialize action head and value head weights, keep feature extractor.

    Keeps: mlp_extractor.policy_net.* (shared representation)
    Keeps: mlp_extractor.value_net.*  (shared representation)
    Reinit: action_net.weight/bias, log_std, value_net.weight/bias
    """
    import math
    # Reinit action_net (Linear 256 -> act_dim)
    nn.init.orthogonal_(sd["action_net.weight"], gain=0.01)
    nn.init.zeros_(sd["action_net.bias"])
    # Reinit log_std to 0 (std=1.0, neutral exploration)
    sd["log_std"] = torch.zeros(act_dim)
    # Reinit value_net (Linear 256 -> 1)
    nn.init.orthogonal_(sd["value_net.weight"], gain=1.0)
    nn.init.zeros_(sd["value_net.bias"])
    print("  Reinitialized action_net, log_std, value_net (kept feature extractor)")
    return sd


# ═══════════════════════════════════════════════════════════════════
#  WRAPPERS
# ═══════════════════════════════════════════════════════════════════
class UnifiedObsWrapper(gym.Wrapper):
    """Flatten obs and zero-pad to UNIFIED_OBS_DIM."""
    def __init__(self, env, raw_obs_dim):
        super().__init__(env)
        self.raw_obs_dim = raw_obs_dim
        self.pad_dim = UNIFIED_OBS_DIM - raw_obs_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(UNIFIED_OBS_DIM,), dtype=np.float32
        )

    def _pad(self, obs):
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        obs = np.array(obs, dtype=np.float32).flatten()
        if self.pad_dim > 0:
            obs = np.concatenate([obs, np.zeros(self.pad_dim, dtype=np.float32)])
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._pad(obs)
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.item()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.item()
        return obs, float(reward), bool(terminated), bool(truncated), info


class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "success" in info:
            s = info["success"]
            if hasattr(s, "item"):
                s = s.item()
            if hasattr(s, "__len__") and not isinstance(s, (str, bytes)):
                s = s[0]
                if hasattr(s, "item"):
                    s = s.item()
            info["is_success"] = bool(s)
        return obs, reward, terminated, truncated, info

# ═══════════════════════════════════════════════════════════════════
#  ENV FACTORY
# ═══════════════════════════════════════════════════════════════════
def make_env(task_id, idx=0):
    def _init():
        env = gym.make(task_id, obs_mode="state", control_mode=CONTROL_MODE)
        env = UnifiedObsWrapper(env, TASK_CONFIGS[task_id]["raw_obs_dim"])
        env = SuccessInfoWrapper(env)
        env = Monitor(env)
        return env
    return _init

def make_eval_env(task_id):
    env = gym.make(task_id, obs_mode="state", control_mode=CONTROL_MODE)
    env = UnifiedObsWrapper(env, TASK_CONFIGS[task_id]["raw_obs_dim"])
    env = SuccessInfoWrapper(env)
    return env

# ═══════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_policy(model, task_id, n_episodes=EVAL_EPS):
    env = make_eval_env(task_id)
    obs, _ = env.reset()
    successes = []
    total_rewards = []
    ep_reward = 0.0

    while len(successes) < n_episodes:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = model.predict(obs_tensor, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        if terminated or truncated:
            total_rewards.append(ep_reward)
            s = info.get("is_success", False)
            if not s and "success" in info:
                sv = info["success"]
                if hasattr(sv, "item"):
                    sv = sv.item()
                if hasattr(sv, "__len__"):
                    sv = sv[0]
                    if hasattr(sv, "item"):
                        sv = sv.item()
                s = bool(sv)
            successes.append(s)
            ep_reward = 0.0
            obs, _ = env.reset()

    env.close()
    return float(np.mean(successes)), float(np.mean(total_rewards))


# ═══════════════════════════════════════════════════════════════════
#  QUICK OBS ALIGNMENT CHECK
# ═══════════════════════════════════════════════════════════════════
def check_obs_alignment():
    """Print obs statistics for both tasks to verify zero-padding makes sense."""
    print("\n  -- Obs alignment check --")
    for task_id in TASK_SEQUENCE:
        env = gym.make(task_id, obs_mode="state", control_mode=CONTROL_MODE)
        obs, _ = env.reset(seed=42)
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy().flatten()
        else:
            obs_np = np.array(obs).flatten()
        print(f"  {task_id}: dim={len(obs_np)}")
        print(f"    first 5:  {obs_np[:5]}")
        print(f"    last 5:   {obs_np[-5:]}")
        print(f"    mean={obs_np.mean():.4f}, std={obs_np.std():.4f}, "
              f"min={obs_np.min():.4f}, max={obs_np.max():.4f}")
        env.close()

    # Check if first 42 dims are on similar scale (needed for zero-padding to work)
    env0 = gym.make("PokeCube-v1", obs_mode="state", control_mode=CONTROL_MODE)
    env1 = gym.make("PickCube-v1", obs_mode="state", control_mode=CONTROL_MODE)
    obs0, _ = env0.reset(seed=42)
    obs1, _ = env1.reset(seed=42)
    obs0 = obs0.cpu().numpy().flatten() if isinstance(obs0, torch.Tensor) else np.array(obs0).flatten()
    obs1 = obs1.cpu().numpy().flatten() if isinstance(obs1, torch.Tensor) else np.array(obs1).flatten()

    print(f"\n  PokeCube[0:42] mean={obs0[:42].mean():.4f}, std={obs0[:42].std():.4f}")
    print(f"  PickCube[0:42] mean={obs1[:42].mean():.4f}, std={obs1[:42].std():.4f}")
    print(f"  PokeCube[42:54] (will be zero for PickCube): mean={obs0[42:].mean():.4f}, std={obs0[42:].std():.4f}")
    env0.close()
    env1.close()

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("  ManiSkill3 CL - Route A v7 (Transfer Feature Extractor + Fresh Head)")
    print("=" * 80)
    print(f"  Unified obs_dim: {UNIFIED_OBS_DIM}")
    for task, cfg in TASK_CONFIGS.items():
        print(f"    {task}: raw_obs={cfg['raw_obs_dim']} -> padded={UNIFIED_OBS_DIM}, "
              f"act={cfg['act_dim']}, ctrl={CONTROL_MODE}")
    print(f"  Steps: {TOTAL_TIMESTEPS}  envs: {N_ENVS}")
    print(f"  v7 strategy: KEEP feature extractor, REINIT action/value heads")
    print(f"  Phase 1: lr={LR_START}->{LR_END} linear, ent={ENT_COEF}, gamma={GAMMA}")
    print(f"    n_steps={N_STEPS}, batch={BATCH_SIZE}, epochs={N_EPOCHS}")
    print(f"    clip={CLIP_RANGE}, target_kl={TARGET_KL}, grad_norm={MAX_GRAD_NORM}")
    print(f"  Eval: every {EVAL_FREQ} steps, {EVAL_EPS} episodes")
    print("=" * 80)

    check_obs_alignment()

    results = {}

    # ─────────────────────────────────────────────────────────────
    #  PHASE 0
    # ─────────────────────────────────────────────────────────────
    task0 = "PokeCube-v1"
    print(f"\n{'='*80}")
    print(f"  Phase 0: Load official {task0} checkpoint")
    print(f"{'='*80}")

    if not os.path.isfile(OFFICIAL_CKPT):
        print(f"  ERROR: Official checkpoint not found: {OFFICIAL_CKPT}")
        sys.exit(1)

    sb3_sd = convert_official_to_sb3(OFFICIAL_CKPT)

    print("\n  Creating SB3 PPO model...")
    train_envs = FixedDummyVecEnv([make_env(task0, i) for i in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        train_envs,
        policy_kwargs=dict(
            net_arch=dict(pi=NET_ARCH_PI, vf=NET_ARCH_VF),
            activation_fn=ACTIVATION_FN,
        ),
        learning_rate=LR_START,
        ent_coef=ENT_COEF,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        target_kl=TARGET_KL,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
    )

    print("\n  Loading official weights...")
    model.policy.load_state_dict(sb3_sd, strict=False)
    print("  Weights loaded!")

    print(f"\n  -- Eval {task0} (official ckpt) --")
    sr0, r0 = eval_policy(model, task0)
    print(f"    {task0}: SR={sr0*100:.1f}%  avg_R={r0:.2f}")
    results[task0] = {"phase0_official": {"SR": sr0, "avg_R": r0}}

    if sr0 < 0.1:
        print(f"  *** WARNING: SR={sr0*100:.1f}% is very low! ***")

    ckpt0_path = os.path.join(CKPT_SAVE_DIR, "route_a_v7_phase0_pokecube.pt")
    torch.save(model.policy.state_dict(), ckpt0_path)
    print(f"  Saved Phase 0: {ckpt0_path}")

    # ─────────────────────────────────────────────────────────────
    #  PHASE 1: Transfer feature extractor, fresh head
    # ─────────────────────────────────────────────────────────────
    task1 = "PickCube-v1"
    print(f"\n{'='*80}")
    print(f"  Phase 1: Transfer to {task1} (feature extractor + fresh head)")
    print(f"{'='*80}")

    # Get Phase 0 state dict and reinitialize heads
    phase1_sd = {k: v.clone() for k, v in model.policy.state_dict().items()}
    phase1_sd = reinit_head_in_state_dict(phase1_sd, act_dim=TASK_CONFIGS[task1]["act_dim"])

    train_envs.close()
    train_envs = FixedDummyVecEnv([make_env(task1, i) for i in range(N_ENVS)])

    lr_schedule = get_linear_fn(LR_START, LR_END, 1.0)

    model = PPO(
        "MlpPolicy",
        train_envs,
        policy_kwargs=dict(
            net_arch=dict(pi=NET_ARCH_PI, vf=NET_ARCH_VF),
            activation_fn=ACTIVATION_FN,
        ),
        learning_rate=lr_schedule,
        ent_coef=ENT_COEF,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        target_kl=TARGET_KL,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
    )
    model.policy.load_state_dict(phase1_sd, strict=True)
    print(f"  Phase 1 model ready. LR: {LR_START} -> {LR_END} (linear)")

    # Quick sanity check: fresh head on PickCube should be random (SR~0, R~0)
    print(f"\n  -- Sanity: fresh head on {task1} (before training) --")
    sr_pre, r_pre = eval_policy(model, task1, n_episodes=5)
    print(f"    {task1}: SR={sr_pre*100:.1f}%  avg_R={r_pre:.2f} (expected ~0)")

    # Also check: does the model still do well on PokeCube after head reinit?
    print(f"  -- Sanity: reinit head on {task0} (feature extractor only) --")
    sr_pre0, r_pre0 = eval_policy(model, task0, n_episodes=5)
    print(f"    {task0}: SR={sr_pre0*100:.1f}%  avg_R={r_pre0:.2f} (head reinit -> likely 0)")

    phase1_start = time.time()
    best_avg_sr = 0.0
    best_pickcube_sr = 0.0
    total_steps_done = 0
    eval_history = []

    while total_steps_done < TOTAL_TIMESTEPS:
        steps_this_round = min(EVAL_FREQ, TOTAL_TIMESTEPS - total_steps_done)
        model.learn(total_timesteps=steps_this_round, reset_num_timesteps=False)
        total_steps_done += steps_this_round

        print(f"\n  -- [eval @ step {total_steps_done}/{TOTAL_TIMESTEPS}] --")
        sr_t0, r_t0 = eval_policy(model, task0)
        sr_t1, r_t1 = eval_policy(model, task1)
        avg_sr = (sr_t0 + sr_t1) / 2

        pct = total_steps_done / TOTAL_TIMESTEPS * 100
        elapsed = (time.time() - phase1_start) / 60
        print(f"    {task0}: SR={sr_t0*100:6.1f}%  avg_R={r_t0:.2f}")
        print(f"    {task1}: SR={sr_t1*100:6.1f}%  avg_R={r_t1:.2f}")
        print(f"    -> avg SR: {avg_sr*100:.1f}%  (best avg: {best_avg_sr*100:.1f}%, "
              f"best PickCube: {best_pickcube_sr*100:.1f}%)")
        print(f"    elapsed: {elapsed:.1f} min, {pct:.0f}% done")

        eval_record = {
            "step": total_steps_done,
            task0: {"SR": sr_t0, "avg_R": r_t0},
            task1: {"SR": sr_t1, "avg_R": r_t1},
            "avg_SR": avg_sr,
            "elapsed_min": elapsed,
        }
        eval_history.append(eval_record)

        if avg_sr > best_avg_sr:
            best_avg_sr = avg_sr
            ckpt_best = os.path.join(CKPT_SAVE_DIR, "route_a_v7_best_avg.pt")
            torch.save(model.policy.state_dict(), ckpt_best)
            print(f"    * NEW BEST AVG! saved {ckpt_best}")

        if sr_t1 > best_pickcube_sr:
            best_pickcube_sr = sr_t1
            ckpt_best_pc = os.path.join(CKPT_SAVE_DIR, "route_a_v7_best_pickcube.pt")
            torch.save(model.policy.state_dict(), ckpt_best_pc)
            print(f"    * NEW BEST PICKCUBE! saved {ckpt_best_pc}")

        # 30% rule
        if total_steps_done >= int(TOTAL_TIMESTEPS * 0.3) and best_pickcube_sr == 0.0:
            print(f"    *** 30% RULE: PickCube SR still 0% at {total_steps_done} steps! ***")

        results.setdefault(task1, {})[f"step_{total_steps_done}"] = eval_record

        # Save partial results
        result_file = os.path.join(RESULT_DIR, "cl_results_route_a_v7_partial.json")
        with open(result_file, "w") as f:
            json.dump({"eval_history": eval_history, "best_avg_sr": best_avg_sr,
                        "best_pickcube_sr": best_pickcube_sr}, f, indent=2, default=float)

    phase1_time = (time.time() - phase1_start) / 60
    print(f"\n  Phase 1 done! Time: {phase1_time:.1f} min")

    # Load best avg
    ckpt_best = os.path.join(CKPT_SAVE_DIR, "route_a_v7_best_avg.pt")
    if os.path.isfile(ckpt_best):
        model.policy.load_state_dict(torch.load(ckpt_best, map_location="cpu"))
        print(f"  Loaded best avg checkpoint (avg_SR={best_avg_sr*100:.1f}%)")

    ckpt1_path = os.path.join(CKPT_SAVE_DIR, "route_a_v7_after_pickcube.pt")
    torch.save(model.policy.state_dict(), ckpt1_path)

    # Final eval (50 episodes)
    print(f"\n  -- Final eval (50 episodes) --")
    sr_t0_f, r_t0_f = eval_policy(model, task0, n_episodes=50)
    sr_t1_f, r_t1_f = eval_policy(model, task1, n_episodes=50)
    print(f"    {task0}: SR={sr_t0_f*100:.1f}%  avg_R={r_t0_f:.2f}")
    print(f"    {task1}: SR={sr_t1_f*100:.1f}%  avg_R={r_t1_f:.2f}")

    # ─────────────────────────────────────────────────────────────
    #  RESULTS
    # ─────────────────────────────────────────────────────────────
    forgetting_pokecube = sr0 - sr_t0_f

    all_results = {
        "route": "A_v7",
        "strategy": "transfer_feature_extractor_fresh_head",
        "task_sequence": TASK_SEQUENCE,
        "control_mode": CONTROL_MODE,
        "unified_obs_dim": UNIFIED_OBS_DIM,
        "hyperparams": {
            "lr_start": LR_START, "lr_end": LR_END, "lr_schedule": "linear",
            "ent_coef": ENT_COEF, "gamma": GAMMA,
            "n_steps": N_STEPS, "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS,
            "clip_range": CLIP_RANGE, "target_kl": TARGET_KL,
            "max_grad_norm": MAX_GRAD_NORM, "gae_lambda": GAE_LAMBDA,
            "total_timesteps": TOTAL_TIMESTEPS, "n_envs": N_ENVS,
            "head_reinit": True,
        },
        "results": {
            "phase0_official": {task0: {"SR": sr0, "avg_R": r0}},
            "phase1_final": {
                task0: {"SR": sr_t0_f, "avg_R": r_t0_f},
                task1: {"SR": sr_t1_f, "avg_R": r_t1_f},
            },
        },
        "forgetting": {
            task0: {"before": sr0, "after": sr_t0_f, "forget": forgetting_pokecube},
            task1: {"before": 0.0, "after": sr_t1_f, "forget": 0.0},
        },
        "eval_history": eval_history,
        "best_avg_sr": best_avg_sr,
        "best_pickcube_sr": best_pickcube_sr,
    }

    result_file = os.path.join(RESULT_DIR, "cl_results_route_a_v7.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'='*80}")
    print(f"  Route A v7 Results: {task0} -> {task1}")
    print(f"{'='*80}")
    print(f"  {'Task':<20s} {'Before CL':>12s} {'After CL':>12s} {'Forget':>10s}")
    print(f"  {'-'*54}")
    print(f"  {task0:<20s} {sr0*100:>10.1f}%  {sr_t0_f*100:>10.1f}%  {forgetting_pokecube*100:>+8.1f}%")
    print(f"  {task1:<20s} {'N/A':>10s}  {sr_t1_f*100:>10.1f}%  {'N/A':>10s}")
    print(f"\n  Best PickCube SR during training: {best_pickcube_sr*100:.1f}%")
    print(f"  Results saved: {result_file}")

    train_envs.close()
    print("\nDONE")
