"""
ManiSkill3 CL - Route A v5
  Phase 0: Load OFFICIAL PokeCube-v1 checkpoint, evaluate (no training needed)
  Phase 1: Fine-tune on PickCube-v1, measure forgetting on PokeCube

  Key fixes from v4:
  - Uses REAL official checkpoint (not user's training intermediate)
  - Unified obs_dim=54 via zero-padding (PokeCube 54, PickCube 42->54)
  - Same model architecture throughout -> no dimension mismatch on eval
  - max_episode_steps matches env default
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

# ═══════════════════════════════════════════════════════════════════
#  Fix terminal_observation shape (ManiSkill returns 1D, SB3 wants 2D)
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
UNIFIED_OBS_DIM = 54  # max of all task obs_dims
CONTROL_MODE = "pd_joint_delta_pos"

TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 4
NET_ARCH_PI = [256, 256, 256]
NET_ARCH_VF = [256, 256, 256]
ACTIVATION_FN = nn.Tanh
BASE_LR = 3e-4          # Match official LR
ENT_COEF = 0.0          # Match official (no entropy bonus)
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.8             # Match official gamma

EVAL_FREQ = 100_000
EVAL_EPS = 50

# REAL official checkpoint
OFFICIAL_CKPT = os.path.expanduser(
    r"~\.maniskill\demos\PokeCube-v1\rl\ppo_pd_joint_delta_pos_ckpt.pt"
)
CKPT_SAVE_DIR = r"E:\munichi\cl_checkpoints"
RESULT_DIR = r"E:\munichi\cl_results"
os.makedirs(CKPT_SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  Official Agent structure (for checkpoint conversion reference)
# ═══════════════════════════════════════════════════════════════════
def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class OfficialAgent(nn.Module):
    def __init__(self, obs_dim=54, act_dim=8):
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            _layer_init(nn.Linear(256, 256)), nn.Tanh(),
            _layer_init(nn.Linear(256, 256)), nn.Tanh(),
            _layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            _layer_init(nn.Linear(256, 256)), nn.Tanh(),
            _layer_init(nn.Linear(256, 256)), nn.Tanh(),
            _layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

# ═══════════════════════════════════════════════════════════════════
#  CHECKPOINT CONVERTER (Official → SB3)
# ═══════════════════════════════════════════════════════════════════
def convert_official_to_sb3(official_ckpt_path):
    """Convert official ManiSkill checkpoint → SB3 state_dict format.

    Unified obs_dim = 54. First layer weights for PickCube (42→54) are
    already the right size since official PokeCube ckpt has 54-dim input.
    """
    raw = torch.load(official_ckpt_path, map_location="cpu", weights_only=False)
    print(f"  Loaded: {official_ckpt_path}")
    print(f"  actor_logstd: {raw['actor_logstd'].flatten().tolist()}")

    sb3_sd = {}
    # Actor hidden layers
    for src, dst in [(0, 0), (2, 2), (4, 4)]:
        sb3_sd[f"mlp_extractor.policy_net.{dst}.weight"] = raw[f"actor_mean.{src}.weight"].clone()
        sb3_sd[f"mlp_extractor.policy_net.{dst}.bias"] = raw[f"actor_mean.{src}.bias"].clone()
    # Actor output
    sb3_sd["action_net.weight"] = raw["actor_mean.6.weight"].clone()
    sb3_sd["action_net.bias"] = raw["actor_mean.6.bias"].clone()
    # log_std
    sb3_sd["log_std"] = raw["actor_logstd"].clone().flatten()
    # Critic hidden layers
    for src, dst in [(0, 0), (2, 2), (4, 4)]:
        sb3_sd[f"mlp_extractor.value_net.{dst}.weight"] = raw[f"critic.{src}.weight"].clone()
        sb3_sd[f"mlp_extractor.value_net.{dst}.bias"] = raw[f"critic.{src}.bias"].clone()
    # Critic output
    sb3_sd["value_net.weight"] = raw["critic.6.weight"].clone()
    sb3_sd["value_net.bias"] = raw["critic.6.bias"].clone()

    print(f"  Converted {len(sb3_sd)} keys")
    return sb3_sd

# ═══════════════════════════════════════════════════════════════════
#  WRAPPERS
# ═══════════════════════════════════════════════════════════════════
class UnifiedObsWrapper(gym.Wrapper):
    """Flatten obs and zero-pad to UNIFIED_OBS_DIM so the same model
    can handle tasks with different observation dimensions."""
    def __init__(self, env, raw_obs_dim):
        super().__init__(env)
        self.raw_obs_dim = raw_obs_dim
        self.pad_dim = UNIFIED_OBS_DIM - raw_obs_dim
        # Override observation space to unified dim
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
        # gym.make already includes FlattenActionSpaceWrapper internally
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
#  MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("  ManiSkill3 CL - Route A v5 (Official Checkpoint + Unified Obs)")
    print("  Phase 0: Load PokeCube official ckpt, evaluate")
    print("  Phase 1: Fine-tune on PickCube, measure forgetting")
    print("=" * 80)
    print(f"  Unified obs_dim: {UNIFIED_OBS_DIM}")
    for task, cfg in TASK_CONFIGS.items():
        print(f"    {task}: raw_obs={cfg['raw_obs_dim']} -> padded={UNIFIED_OBS_DIM}, "
              f"act={cfg['act_dim']}, ctrl={CONTROL_MODE}")
    print(f"  Steps/task: {TOTAL_TIMESTEPS}  envs: {N_ENVS}")
    print(f"  Net: pi={NET_ARCH_PI}, vf={NET_ARCH_VF}, Tanh")
    print(f"  Hyper: lr={BASE_LR}, ent={ENT_COEF}, n_steps={N_STEPS}, "
          f"batch={BATCH_SIZE}, epochs={N_EPOCHS}, gamma={GAMMA}")
    print(f"  Eval: every {EVAL_FREQ} steps, {EVAL_EPS} episodes")
    print(f"  Official ckpt: {OFFICIAL_CKPT}")
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

    # Create SB3 PPO with unified obs_dim=54
    print("\n  Creating SB3 PPO model...")
    train_envs = FixedDummyVecEnv([make_env(task0, i) for i in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        train_envs,
        policy_kwargs=dict(
            net_arch=dict(pi=NET_ARCH_PI, vf=NET_ARCH_VF),
            activation_fn=ACTIVATION_FN,
        ),
        learning_rate=BASE_LR,
        ent_coef=ENT_COEF,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        verbose=1,
    )

    # Load weights with verification
    print("\n  Loading official weights...")
    model_keys = set(model.policy.state_dict().keys())
    convert_keys = set(sb3_sd.keys())
    matched = model_keys & convert_keys
    print(f"  Matched: {len(matched)}/{len(model_keys)}")
    if len(matched) != len(model_keys):
        missing = model_keys - convert_keys
        print(f"  MISSING: {sorted(missing)}")
    model.policy.load_state_dict(sb3_sd, strict=False)
    print("  Weights loaded!")

    # Evaluate Phase 0
    print(f"\n  -- Eval {task0} (official ckpt, no training) --")
    sr0, r0 = eval_policy(model, task0)
    print(f"    {task0}: SR={sr0*100:.1f}%  avg_R={r0:.2f}")
    results[task0] = {"phase0_official": {"SR": sr0, "avg_R": r0}}

    if sr0 < 0.1:
        print(f"\n  *** WARNING: SR={sr0*100:.1f}% is very low! ***")
        print(f"  Official checkpoint should give ~84% SR.")
        print(f"  Check control_mode / obs_mode compatibility.")

    # Save Phase 0 checkpoint
    ckpt0_path = os.path.join(CKPT_SAVE_DIR, "route_a_v5_phase0_pokecube.pt")
    torch.save(model.policy.state_dict(), ckpt0_path)
    print(f"  Saved Phase 0: {ckpt0_path}")

    # ─────────────────────────────────────────────────────────────
    #  PHASE 1: Fine-tune on PickCube
    # ─────────────────────────────────────────────────────────────
    task1 = "PickCube-v1"
    print(f"\n{'='*80}")
    print(f"  Phase 1: Fine-tune on {task1} ({TOTAL_TIMESTEPS} steps)")
    print(f"{'='*80}")

    # Switch environment to PickCube (same unified obs_dim=54, no arch change)
    phase0_sd = {k: v.clone() for k, v in model.policy.state_dict().items()}
    train_envs.close()
    train_envs = FixedDummyVecEnv([make_env(task1, i) for i in range(N_ENVS)])

    # Create new model with lower LR for fine-tuning, load Phase 0 weights
    finetune_lr = BASE_LR * 0.3
    model = PPO(
        "MlpPolicy",
        train_envs,
        policy_kwargs=dict(
            net_arch=dict(pi=NET_ARCH_PI, vf=NET_ARCH_VF),
            activation_fn=ACTIVATION_FN,
        ),
        learning_rate=finetune_lr,
        ent_coef=ENT_COEF,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        verbose=1,
    )
    model.policy.load_state_dict(phase0_sd, strict=True)
    print(f"  Transferred weights to new model (LR={finetune_lr})")

    phase1_start = time.time()
    best_avg_sr = 0.0

    for eval_step in range(EVAL_FREQ, TOTAL_TIMESTEPS + 1, EVAL_FREQ):
        model.learn(total_timesteps=eval_step, reset_num_timesteps=False)

        # Eval BOTH tasks with SAME model (unified obs_dim makes this possible)
        print(f"\n  -- [eval @ step {eval_step}] --")
        sr_t0, r_t0 = eval_policy(model, task0)
        sr_t1, r_t1 = eval_policy(model, task1)
        avg_sr = (sr_t0 + sr_t1) / 2

        pct = eval_step / TOTAL_TIMESTEPS * 100
        print(f"    {task0}: SR={sr_t0*100:6.1f}%  avg_R={r_t0:.2f}")
        print(f"    {task1}: SR={sr_t1*100:6.1f}%  avg_R={r_t1:.2f}")
        print(f"    -> avg SR: {avg_sr*100:.1f}%  (best: {best_avg_sr*100:.1f}%)")

        if avg_sr > best_avg_sr:
            best_avg_sr = avg_sr
            ckpt_best = os.path.join(CKPT_SAVE_DIR, "route_a_v5_best.pt")
            torch.save(model.policy.state_dict(), ckpt_best)
            print(f"    * NEW BEST! saved {ckpt_best}")

        results[task1] = results.get(task1, {})
        results[task1][f"step_{eval_step}"] = {
            task0: {"SR": sr_t0, "avg_R": r_t0},
            task1: {"SR": sr_t1, "avg_R": r_t1},
        }

    phase1_time = (time.time() - phase1_start) / 60
    print(f"\n  Phase 1 done! Time: {phase1_time:.1f} min")

    # Load best for final eval
    ckpt_best = os.path.join(CKPT_SAVE_DIR, "route_a_v5_best.pt")
    if os.path.isfile(ckpt_best):
        model.policy.load_state_dict(torch.load(ckpt_best, map_location="cpu"))
        print(f"  Loaded best checkpoint (avg_SR={best_avg_sr*100:.1f}%)")

    ckpt1_path = os.path.join(CKPT_SAVE_DIR, "route_a_v5_after_pickcube.pt")
    torch.save(model.policy.state_dict(), ckpt1_path)

    # Final eval
    print(f"\n  -- Final eval --")
    sr_t0_f, r_t0_f = eval_policy(model, task0)
    sr_t1_f, r_t1_f = eval_policy(model, task1)
    print(f"    {task0}: SR={sr_t0_f*100:.1f}%  avg_R={r_t0_f:.2f}")
    print(f"    {task1}: SR={sr_t1_f*100:.1f}%  avg_R={r_t1_f:.2f}")

    # ─────────────────────────────────────────────────────────────
    #  RESULTS
    # ─────────────────────────────────────────────────────────────
    forgetting_pokecube = sr0 - sr_t0_f

    all_results = {
        "route": "A_v5",
        "task_sequence": TASK_SEQUENCE,
        "control_mode": CONTROL_MODE,
        "unified_obs_dim": UNIFIED_OBS_DIM,
        "hyperparams": {
            "lr": BASE_LR, "ent_coef": ENT_COEF, "gamma": GAMMA,
            "n_steps": N_STEPS, "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS,
            "total_timesteps": TOTAL_TIMESTEPS, "n_envs": N_ENVS,
        },
        "results": {
            "phase0_official": {
                task0: {"SR": sr0, "avg_R": r0},
            },
            "phase1_final": {
                task0: {"SR": sr_t0_f, "avg_R": r_t0_f},
                task1: {"SR": sr_t1_f, "avg_R": r_t1_f},
            },
        },
        "forgetting": {
            task0: {"before": sr0, "after": sr_t0_f, "forget": forgetting_pokecube},
            task1: {"before": 0.0, "after": sr_t1_f, "forget": 0.0},
        },
        "training_details": results,
    }

    result_file = os.path.join(RESULT_DIR, "cl_results_route_a_v5.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'='*80}")
    print(f"  Route A v5 Results: {task0} -> {task1}")
    print(f"{'='*80}")
    print(f"  {'Task':<20s} {'Before CL':>12s} {'After CL':>12s} {'Forget':>10s}")
    print(f"  {'-'*54}")
    print(f"  {task0:<20s} {sr0*100:>10.1f}%  {sr_t0_f*100:>10.1f}%  {forgetting_pokecube*100:>+8.1f}%")
    print(f"  {task1:<20s} {'N/A':>10s}  {sr_t1_f*100:>10.1f}%  {'N/A':>10s}")
    print(f"\n  Results saved: {result_file}")

    train_envs.close()
    print("\nDONE")
