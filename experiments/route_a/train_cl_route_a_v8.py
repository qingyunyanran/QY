"""
ManiSkill3 CL - Route A v8
  Frozen Feature Extractor + Task-Specific Heads

  Strategy:
    Phase 0: Load official PokeCube checkpoint, save its action head
    Phase 1: Freeze policy_net (zero forgetting!), train on PickCube with:
             - fresh action_net + log_std (new task head)
             - trainable value_net (value function must adapt to new reward)
             - policy_net FROZEN (preserves PokeCube representation)
    Eval:    For PokeCube eval, swap in saved PokeCube action head
             For PickCube eval, use current action head

  Why v8:
    v7 showed fresh head works (PickCube R=24.5 > from-scratch R=22.72),
    BUT unfrozen policy_net overwrote PokeCube features -> PokeCube 0%.
    v7 also early-stopped at iter 3 (target_kl=0.02 too strict).

    v8 fixes:
    1. Freeze mlp_extractor.policy_net -> PokeCube features preserved
    2. Save/restore PokeCube action head for proper multi-task eval
    3. target_kl 0.02 -> 0.08 (no premature stopping)
    4. n_epochs 8 -> 4 (less aggressive per-batch updates)
    5. gamma 0.85 -> 0.9 (PickCube needs longer horizon for grasp+lift)
    6. Slightly higher LR for head-only training (5e-4 -> 5e-5)

  Trainable params: ~148K (value_net + heads) out of ~295K total
  Frozen params: ~147K (policy_net = PokeCube feature extractor)
"""
import os, sys, time, json, copy
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

TOTAL_TIMESTEPS = 3_000_000
N_ENVS = 4
NET_ARCH_PI = [256, 256, 256]
NET_ARCH_VF = [256, 256, 256]
ACTIVATION_FN = nn.Tanh

# v8 hyperparams — head-only training can use higher LR
LR_START = 5e-4          # higher: only training heads + value_net
LR_END = 5e-5            # decay to 1/10
ENT_COEF = 0.01          # exploration for grasp
N_STEPS = 2048
BATCH_SIZE = 128
N_EPOCHS = 4             # was 8, reduced for stability
GAMMA = 0.9              # was 0.85, PickCube needs longer horizon
CLIP_RANGE = 0.2
TARGET_KL = 0.08         # was 0.02, way too strict
MAX_GRAD_NORM = 0.5
GAE_LAMBDA = 0.95

EVAL_FREQ = 100_000
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


def save_action_head(sd):
    """Extract task-specific action head weights."""
    return {
        "action_net.weight": sd["action_net.weight"].clone(),
        "action_net.bias": sd["action_net.bias"].clone(),
        "log_std": sd["log_std"].clone(),
    }


def load_action_head(sd, head):
    """Load a saved action head into state dict (returns new dict)."""
    new_sd = {k: v.clone() for k, v in sd.items()}
    new_sd["action_net.weight"] = head["action_net.weight"].clone()
    new_sd["action_net.bias"] = head["action_net.bias"].clone()
    new_sd["log_std"] = head["log_std"].clone()
    return new_sd


def reinit_action_head(sd, act_dim=8):
    """Reinitialize action head only. Keep value head as-is (will be trained)."""
    import math
    nn.init.orthogonal_(sd["action_net.weight"], gain=0.01)
    nn.init.zeros_(sd["action_net.bias"])
    sd["log_std"] = torch.zeros(act_dim)
    print("  Reinitialized action_net + log_std (fresh PickCube head)")
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
#  EVALUATION (with head swapping for proper CL eval)
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_policy(model, task_id, n_episodes=EVAL_EPS, action_head=None):
    """Evaluate policy. If action_head provided, temporarily swap it in.

    This is the CORRECT way to do multi-task CL evaluation:
    - PickCube eval: use current action head
    - PokeCube eval: swap in saved PokeCube action head
    """
    policy = model.policy
    saved_head = None

    if action_head is not None:
        # Save current head
        saved_head = save_action_head(policy.state_dict())
        # Load target task head
        new_sd = load_action_head(policy.state_dict(), action_head)
        policy.load_state_dict(new_sd, strict=True)

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

    if saved_head is not None:
        # Restore original head
        restored_sd = load_action_head(policy.state_dict(), saved_head)
        policy.load_state_dict(restored_sd, strict=True)

    return float(np.mean(successes)), float(np.mean(total_rewards))


# ═══════════════════════════════════════════════════════════════════
#  FREEZE POLICY NETWORK
# ═══════════════════════════════════════════════════════════════════
def freeze_policy_net(model):
    """Freeze mlp_extractor.policy_net to preserve PokeCube features.

    Value network remains trainable (value features must adapt to PickCube
    reward structure, and they don't affect action selection at all).
    """
    trainable = 0
    frozen = 0
    for name, param in model.policy.named_parameters():
        if name.startswith("mlp_extractor.policy_net"):
            param.requires_grad = False
            frozen += param.numel()
        else:
            trainable += param.numel()

    total = trainable + frozen
    print(f"\n  Parameter breakdown:")
    print(f"    Frozen (policy_net):  {frozen:>8d} ({frozen/total*100:.1f}%)")
    print(f"    Trainable (rest):     {trainable:>8d} ({trainable/total*100:.1f}%)")
    print(f"    Total:                {total:>8d}")

    # Rebuild optimizer with only trainable params
    trainable_params = [p for p in model.policy.parameters() if p.requires_grad]
    lr = model.lr_schedule(1.0) if callable(model.lr_schedule) else model.learning_rate

    model.policy.optimizer = torch.optim.Adam(
        trainable_params,
        lr=lr,
        eps=1e-5,  # SB3 default
    )
    print(f"  Optimizer rebuilt with {len(trainable_params)} param groups, lr={lr}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("  ManiSkill3 CL - Route A v8 (Frozen Feature Extractor + Task Heads)")
    print("=" * 80)
    print(f"  Unified obs_dim: {UNIFIED_OBS_DIM}")
    for task, cfg in TASK_CONFIGS.items():
        print(f"    {task}: raw_obs={cfg['raw_obs_dim']} -> padded={UNIFIED_OBS_DIM}, "
              f"act={cfg['act_dim']}, ctrl={CONTROL_MODE}")

    print(f"\n  Steps: {TOTAL_TIMESTEPS}  envs: {N_ENVS}")
    print(f"  v8 strategy: FREEZE policy_net, train value_net + fresh action head")
    print(f"  Phase 1: lr={LR_START}->{LR_END} linear, ent={ENT_COEF}, gamma={GAMMA}")
    print(f"    n_steps={N_STEPS}, batch={BATCH_SIZE}, epochs={N_EPOCHS}")
    print(f"    clip={CLIP_RANGE}, target_kl={TARGET_KL}, grad_norm={MAX_GRAD_NORM}")
    print(f"  Eval: every {EVAL_FREQ} steps, {EVAL_EPS} episodes")
    print(f"  Eval: PokeCube uses SAVED action head (proper multi-head CL)")
    print("=" * 80)

    results = {}

    # ─────────────────────────────────────────────────────────────
    #  PHASE 0: Load official PokeCube checkpoint
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

    # Evaluate PokeCube with official weights
    print(f"\n  -- Eval {task0} (official ckpt) --")
    sr0, r0 = eval_policy(model, task0)
    print(f"    {task0}: SR={sr0*100:.1f}%  avg_R={r0:.2f}")
    results[task0] = {"phase0_official": {"SR": sr0, "avg_R": r0}}

    if sr0 < 0.1:
        print(f"  *** WARNING: SR={sr0*100:.1f}% is very low! ***")

    # SAVE PokeCube action head for later evaluation
    pokecube_head = save_action_head(model.policy.state_dict())
    print(f"  Saved PokeCube action head (action_net + log_std)")

    # Save full phase 0 checkpoint
    ckpt0_path = os.path.join(CKPT_SAVE_DIR, "route_a_v8_phase0_pokecube.pt")
    torch.save(model.policy.state_dict(), ckpt0_path)
    print(f"  Saved Phase 0: {ckpt0_path}")

    # ─────────────────────────────────────────────────────────────
    #  PHASE 1: Freeze policy_net, fresh head, train PickCube
    # ─────────────────────────────────────────────────────────────
    task1 = "PickCube-v1"
    print(f"\n{'='*80}")
    print(f"  Phase 1: Transfer to {task1} (FROZEN policy_net + fresh head)")
    print(f"{'='*80}")

    # Get Phase 0 state dict and reinitialize ONLY action head
    phase1_sd = {k: v.clone() for k, v in model.policy.state_dict().items()}
    phase1_sd = reinit_action_head(phase1_sd, act_dim=TASK_CONFIGS[task1]["act_dim"])

    # Close old envs, create PickCube envs
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
    print(f"  Phase 1 model loaded. LR: {LR_START} -> {LR_END} (linear)")

    # FREEZE policy_net — this is the key v8 change
    freeze_policy_net(model)

    # Sanity checks
    print(f"\n  -- Sanity: fresh head on {task1} (before training) --")
    sr_pre, r_pre = eval_policy(model, task1, n_episodes=5)
    print(f"    {task1}: SR={sr_pre*100:.1f}%  avg_R={r_pre:.2f} (expected ~0)")

    print(f"  -- Sanity: PokeCube head swapped back (frozen features) --")
    sr_pre0, r_pre0 = eval_policy(model, task0, n_episodes=5, action_head=pokecube_head)
    print(f"    {task0}: SR={sr_pre0*100:.1f}%  avg_R={r_pre0:.2f} (should be ~{sr0*100:.0f}%)")

    if sr_pre0 < sr0 * 0.8:
        print(f"  *** WARNING: PokeCube dropped to {sr_pre0*100:.1f}% after head swap! "
              f"Feature freeze may not be working correctly. ***")

    phase1_start = time.time()
    best_avg_sr = 0.0
    best_pickcube_sr = 0.0
    total_steps_done = 0
    eval_history = []
    no_sr_warning_fired = False

    while total_steps_done < TOTAL_TIMESTEPS:
        steps_this_round = min(EVAL_FREQ, TOTAL_TIMESTEPS - total_steps_done)
        model.learn(total_timesteps=steps_this_round, reset_num_timesteps=False)
        total_steps_done += steps_this_round

        print(f"\n  -- [eval @ step {total_steps_done}/{TOTAL_TIMESTEPS}] --")
        # PickCube: use current (training) head
        sr_t1, r_t1 = eval_policy(model, task1)
        # PokeCube: swap in saved head (frozen features should preserve performance)
        sr_t0, r_t0 = eval_policy(model, task0, action_head=pokecube_head)
        avg_sr = (sr_t0 + sr_t1) / 2

        pct = total_steps_done / TOTAL_TIMESTEPS * 100
        elapsed = (time.time() - phase1_start) / 60
        print(f"    {task0}: SR={sr_t0*100:6.1f}%  avg_R={r_t0:.2f}  (frozen features + saved head)")
        print(f"    {task1}: SR={sr_t1*100:6.1f}%  avg_R={r_t1:.2f}  (training)")
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
            ckpt_best = os.path.join(CKPT_SAVE_DIR, "route_a_v8_best_avg.pt")
            torch.save({
                "state_dict": model.policy.state_dict(),
                "pokecube_head": pokecube_head,
            }, ckpt_best)
            print(f"    * NEW BEST AVG! saved {ckpt_best}")

        if sr_t1 > best_pickcube_sr:
            best_pickcube_sr = sr_t1
            ckpt_best_pc = os.path.join(CKPT_SAVE_DIR, "route_a_v8_best_pickcube.pt")
            torch.save({
                "state_dict": model.policy.state_dict(),
                "pokecube_head": pokecube_head,
            }, ckpt_best_pc)
            print(f"    * NEW BEST PICKCUBE! saved {ckpt_best_pc}")

        # 30% rule
        if total_steps_done >= int(TOTAL_TIMESTEPS * 0.3) and best_pickcube_sr == 0.0:
            if not no_sr_warning_fired:
                print(f"\n    *** 30% RULE WARNING: PickCube SR still 0% at {total_steps_done} steps! ***")
                print(f"    *** Frozen features may not support grasping. Check diagnostics. ***")
                no_sr_warning_fired = True
                # Save a diagnostic checkpoint
                ckpt_diag = os.path.join(CKPT_SAVE_DIR, "route_a_v8_30pct_diag.pt")
                torch.save({
                    "state_dict": model.policy.state_dict(),
                    "pokecube_head": pokecube_head,
                }, ckpt_diag)

        results.setdefault(task1, {})[f"step_{total_steps_done}"] = eval_record

        # Save partial results
        result_file = os.path.join(RESULT_DIR, "cl_results_route_a_v8_partial.json")
        with open(result_file, "w") as f:
            json.dump({"eval_history": eval_history, "best_avg_sr": best_avg_sr,
                        "best_pickcube_sr": best_pickcube_sr}, f, indent=2, default=float)

    phase1_time = (time.time() - phase1_start) / 60
    print(f"\n  Phase 1 done! Time: {phase1_time:.1f} min")

    # Load best avg checkpoint
    ckpt_best = os.path.join(CKPT_SAVE_DIR, "route_a_v8_best_avg.pt")
    if os.path.isfile(ckpt_best):
        best_data = torch.load(ckpt_best, map_location="cpu", weights_only=False)
        model.policy.load_state_dict(best_data["state_dict"])
        pokecube_head = best_data["pokecube_head"]
        print(f"  Loaded best avg checkpoint (avg_SR={best_avg_sr*100:.1f}%)")

    # Save final checkpoint (with heads)
    ckpt1_path = os.path.join(CKPT_SAVE_DIR, "route_a_v8_final.pt")
    torch.save({
        "state_dict": model.policy.state_dict(),
        "pokecube_head": pokecube_head,
    }, ckpt1_path)

    # Final eval (50 episodes)
    print(f"\n  -- Final eval (50 episodes) --")
    sr_t1_f, r_t1_f = eval_policy(model, task1, n_episodes=50)
    sr_t0_f, r_t0_f = eval_policy(model, task0, n_episodes=50, action_head=pokecube_head)
    print(f"    {task0}: SR={sr_t0_f*100:.1f}%  avg_R={r_t0_f:.2f}  (frozen + saved head)")
    print(f"    {task1}: SR={sr_t1_f*100:.1f}%  avg_R={r_t1_f:.2f}")

    # ─────────────────────────────────────────────────────────────
    #  RESULTS
    # ─────────────────────────────────────────────────────────────
    forgetting_pokecube = sr0 - sr_t0_f

    all_results = {
        "route": "A_v8",
        "strategy": "frozen_policy_net_task_specific_heads",
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
            "freeze_policy_net": True,
            "value_net_trainable": True,
            "task_specific_heads": True,
        },
        "results": {
            "phase0_official": {task0: {"SR": sr0, "avg_R": r0}},
            "phase1_final": {
                task0: {"SR": sr_t0_f, "avg_R": r_t0_f, "eval_method": "saved_action_head"},
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

    result_file = os.path.join(RESULT_DIR, "cl_results_route_a_v8.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'='*80}")
    print(f"  Route A v8 Results: {task0} -> {task1}")
    print(f"{'='*80}")
    print(f"  {'Task':<20s} {'Before CL':>12s} {'After CL':>12s} {'Forget':>10s}")
    print(f"  {'-'*54}")
    print(f"  {task0:<20s} {sr0*100:>10.1f}%  {sr_t0_f*100:>10.1f}%  {forgetting_pokecube*100:>+8.1f}%")
    print(f"  {task1:<20s} {'N/A':>10s}  {sr_t1_f*100:>10.1f}%  {'N/A':>10s}")
    print(f"\n  Best PickCube SR during training: {best_pickcube_sr*100:.1f}%")
    print(f"  PokeCube forgetting: {forgetting_pokecube*100:+.1%} (should be ~0 with frozen policy_net)")
    print(f"  Results saved: {result_file}")

    train_envs.close()
    print("\nDONE")
