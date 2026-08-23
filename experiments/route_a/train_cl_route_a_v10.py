"""
ManiSkill3 CL - Route A v10
  Frozen Backbone + LoRA Adapters + Task-Specific Heads
  (TAIL paper adaptation for RL/PPO)

  Strategy:
    Phase 0: Load official PokeCube checkpoint, save action head
    Phase 1: Freeze policy_net original weights, add LoRA adapters,
             train LoRA params + fresh action head + value_net on PickCube
    Eval:    PokeCube -> disable LoRA + swap in saved head = zero forgetting
             PickCube -> enable LoRA + current head = new task performance

  Why v9:
    v8 froze policy_net completely -> PokeCube 95% preserved, but PickCube
    avg_R=8.64 (worse than from-scratch 22.72!). Frozen poking-centric
    features can't support grasping.

    TAIL (ICLR 2024) showed frozen backbone + LoRA (1% params) achieves
    zero forgetting while enabling new task learning in continual IL.
    v9 adapts this for PPO/RL:
    - LoRA rank=16 on each policy_net layer (~25K extra params)
    - Original weights frozen forever
    - LoRA can be toggled: off=PokeCube, on=PickCube
    - action head: task-specific (saved per task)
    - value_net: fully trainable (different reward structure per task)

  Trainable params breakdown (approx):
    LoRA A/B matrices:     ~25K  (3 layers x 2 x 256*8)
    action_net + log_std:  ~2K
    value_net full:        ~147K  (must adapt per-task reward)
    Total trainable:       ~174K
    Frozen (policy net):   ~132K
"""
import os, sys, time, json, math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
#  LoRA LINEAR LAYER
# ═══════════════════════════════════════════════════════════════════
class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation.

    output = original(x) + scale * (x @ A @ B)
    where A: [in_features, rank], B: [rank, out_features]

    Original weights are frozen. LoRA can be enabled/disabled at runtime.
    When disabled: output = original(x) (exact PokeCube behavior).
    """
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scale = alpha / rank
        self.enabled = True

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, self.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # Start with zero contribution

    def forward(self, x):
        out = self.original(x)
        if self.enabled:
            out = out + (x @ self.lora_A @ self.lora_B) * self.scale
        return out

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


def inject_lora(policy_net: nn.Sequential, rank: int = 8, alpha: float = 16.0):
    """Replace each Linear layer in policy_net with LoRALinear.

    Returns list of LoRALinear modules for toggling.
    """
    lora_layers = []
    for i, layer in enumerate(policy_net):
        if isinstance(layer, nn.Linear):
            lora_layer = LoRALinear(layer, rank=rank, alpha=alpha)
            policy_net[i] = lora_layer
            lora_layers.append(lora_layer)
            print(f"    Injected LoRA (rank={rank}, alpha={alpha}) into "
                  f"policy_net.{i}: {layer.in_features}->{layer.out_features}")
    return lora_layers


def set_lora_enabled(lora_layers, enabled: bool):
    for layer in lora_layers:
        if enabled:
            layer.enable()
        else:
            layer.disable()


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

# LoRA params
LORA_RANK = 16
LORA_ALPHA = 16.0

# v9 hyperparams
LR_START = 5e-4          # moderate: training LoRA + heads + value_net
LR_END = 5e-5
ENT_COEF = 0.02
N_STEPS = 2048
BATCH_SIZE = 128
N_EPOCHS = 4
GAMMA = 0.9
CLIP_RANGE = 0.2
TARGET_KL = 0.08
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
    """Reinitialize action head only."""
    nn.init.orthogonal_(sd["action_net.weight"], gain=0.01)
    nn.init.zeros_(sd["action_net.bias"])
    sd["log_std"] = torch.zeros(act_dim)
    print("  Reinitialized action_net + log_std (fresh PickCube head)")
    return sd


# ═══════════════════════════════════════════════════════════════════
#  WRAPPERS
# ═══════════════════════════════════════════════════════════════════
class UnifiedObsWrapper(gym.Wrapper):
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
#  EVALUATION with LoRA + head toggling
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_policy(model, task_id, n_episodes=EVAL_EPS, action_head=None,
                lora_layers=None, lora_enabled=True):
    """Evaluate with proper LoRA + head toggling.

    PokeCube eval: lora_enabled=False + PokeCube action head = exact original
    PickCube eval: lora_enabled=True  + current head = adapted
    """
    policy = model.policy
    saved_head = None
    prev_lora_states = None

    # Toggle LoRA
    if lora_layers is not None:
        prev_lora_states = [l.enabled for l in lora_layers]
        set_lora_enabled(lora_layers, lora_enabled)

    # Swap action head if needed
    if action_head is not None:
        saved_head = save_action_head(policy.state_dict())
        new_sd = load_action_head(policy.state_dict(), action_head)
        policy.load_state_dict(new_sd, strict=False)

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

    # Restore action head
    if saved_head is not None:
        restored_sd = load_action_head(policy.state_dict(), saved_head)
        policy.load_state_dict(restored_sd, strict=False)

    # Restore LoRA state
    if lora_layers is not None and prev_lora_states is not None:
        for layer, prev in zip(lora_layers, prev_lora_states):
            layer.enabled = prev

    return float(np.mean(successes)), float(np.mean(total_rewards))


# ═══════════════════════════════════════════════════════════════════
#  SETUP LoRA + FREEZE + OPTIMIZER
# ═══════════════════════════════════════════════════════════════════
def setup_lora_and_freeze(model, lora_layers):
    """Freeze original weights, rebuild optimizer.

    LoRA layers must already be injected via inject_lora().
    """
    policy = model.policy

    # Count params
    trainable = 0
    frozen = 0
    lora_params = 0
    for name, param in policy.named_parameters():
        if param.requires_grad:
            trainable += param.numel()
            if "lora_" in name:
                lora_params += param.numel()
        else:
            frozen += param.numel()

    total = trainable + frozen
    print(f"\n  Parameter breakdown after LoRA injection:")
    print(f"    LoRA params:          {lora_params:>8d} ({lora_params/total*100:.1f}%)")
    print(f"    Other trainable:      {trainable - lora_params:>8d} "
          f"({(trainable-lora_params)/total*100:.1f}%)")
    print(f"    Total trainable:      {trainable:>8d} ({trainable/total*100:.1f}%)")
    print(f"    Frozen (original):    {frozen:>8d} ({frozen/total*100:.1f}%)")

    # Rebuild optimizer with ALL trainable params
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    lr = model.lr_schedule(1.0) if callable(model.lr_schedule) else model.learning_rate

    model.policy.optimizer = torch.optim.Adam(
        trainable_params, lr=lr, eps=1e-5
    )
    print(f"  Optimizer rebuilt: {len(trainable_params)} param groups, lr={lr}")

    return lora_layers


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("  ManiSkill3 CL - Route A v10 (Frozen Backbone + LoRA + Task Heads)")
    print("=" * 80)
    print(f"  Unified obs_dim: {UNIFIED_OBS_DIM}")
    for task, cfg in TASK_CONFIGS.items():
        print(f"    {task}: raw_obs={cfg['raw_obs_dim']} -> padded={UNIFIED_OBS_DIM}, "
              f"act={cfg['act_dim']}, ctrl={CONTROL_MODE}")
    print(f"\n  LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, scale={LORA_ALPHA/LORA_RANK}")
    print(f"  Steps: {TOTAL_TIMESTEPS}  envs: {N_ENVS}")
    print(f"  v9 strategy: FROZEN original policy_net + LoRA adapters")
    print(f"    PokeCube eval: LoRA OFF + saved head = zero forgetting")
    print(f"    PickCube eval: LoRA ON + fresh head = new task")
    print(f"  Phase 1: lr={LR_START}->{LR_END} linear, ent={ENT_COEF}, gamma={GAMMA}")
    print(f"    n_epochs={N_EPOCHS}, target_kl={TARGET_KL}")
    print("=" * 80)

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

    # Eval PokeCube (no LoRA yet, plain model)
    print(f"\n  -- Eval {task0} (official ckpt) --")
    sr0, r0 = eval_policy(model, task0, lora_layers=None)
    print(f"    {task0}: SR={sr0*100:.1f}%  avg_R={r0:.2f}")
    results[task0] = {"phase0_official": {"SR": sr0, "avg_R": r0}}

    if sr0 < 0.1:
        print(f"  *** WARNING: SR={sr0*100:.1f}% is very low! ***")

    # Save PokeCube action head
    pokecube_head = save_action_head(model.policy.state_dict())
    print(f"  Saved PokeCube action head")

    ckpt0_path = os.path.join(CKPT_SAVE_DIR, "route_a_v10_phase0_pokecube.pt")
    torch.save(model.policy.state_dict(), ckpt0_path)

    # ─────────────────────────────────────────────────────────────
    #  PHASE 1: Inject LoRA, freeze, fresh head, train PickCube
    # ─────────────────────────────────────────────────────────────
    task1 = "PickCube-v1"
    print(f"\n{'='*80}")
    print(f"  Phase 1: Transfer to {task1} (Frozen + LoRA + fresh head)")
    print(f"{'='*80}")

    # Reinit action head for PickCube
    phase1_sd = {k: v.clone() for k, v in model.policy.state_dict().items()}
    phase1_sd = reinit_action_head(phase1_sd, act_dim=TASK_CONFIGS[task1]["act_dim"])

    train_envs.close()
    train_envs = FixedDummyVecEnv([make_env(task1, i) for i in range(N_ENVS)])

    lr_schedule = get_linear_fn(LR_START, LR_END, 1.0)

    # Create model FIRST
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

    # INJECT LORA FIRST (changes key names: policy_net.0.weight ->
    # policy_net.0.original.weight), THEN load weights with remapped keys
    lora_layers = inject_lora(model.policy.mlp_extractor.policy_net,
                               rank=LORA_RANK, alpha=LORA_ALPHA)

    # Remap state dict keys for LoRA-wrapped layers
    def remap_sd_for_lora(sd):
        new_sd = {}
        for k, v in sd.items():
            # policy_net.{i}.weight -> policy_net.{i}.original.weight
            # policy_net.{i}.bias   -> policy_net.{i}.original.bias
            nk = k
            for prefix in ["mlp_extractor.policy_net."]:
                if k.startswith(prefix):
                    parts = k[len(prefix):].split(".")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ("weight", "bias"):
                        nk = f"{prefix}{parts[0]}.original.{parts[1]}"
            new_sd[nk] = v.clone()
        return new_sd

    remapped_sd = remap_sd_for_lora(phase1_sd)
    missing, unexpected = model.policy.load_state_dict(remapped_sd, strict=False)
    # LoRA A/B will be missing (expected, initialized fresh)
    lora_missing = [k for k in missing if "lora_" in k]
    other_missing = [k for k in missing if "lora_" not in k]
    if other_missing:
        print(f"  WARNING: unexpected missing keys: {other_missing}")
    print(f"  Loaded weights. {len(lora_missing)} LoRA params initialized fresh, "
          f"original PokeCube weights loaded.")
    print(f"  Phase 1 model ready.")

    # FREEZE original weights + rebuild optimizer
    setup_lora_and_freeze(model, lora_layers)

    # LoRA starts at B=0, so initially output = original(x)
    # Verify: with LoRA ON but B=0, should match PokeCube behavior
    print(f"\n  -- Sanity 1: {task0} with LoRA ON (B=0, should match original) --")
    sr_on, r_on = eval_policy(model, task0, n_episodes=5,
                               lora_layers=lora_layers, lora_enabled=True)
    print(f"    {task0} (LoRA ON, B=0): SR={sr_on*100:.1f}%  avg_R={r_on:.2f}")

    print(f"  -- Sanity 2: {task0} with LoRA OFF + saved head (exact original) --")
    sr_off, r_off = eval_policy(model, task0, n_episodes=5,
                                 action_head=pokecube_head,
                                 lora_layers=lora_layers, lora_enabled=False)
    print(f"    {task0} (LoRA OFF + head): SR={sr_off*100:.1f}%  avg_R={r_off:.2f}")

    print(f"  -- Sanity 3: {task1} fresh head + LoRA ON (before training) --")
    sr_pre, r_pre = eval_policy(model, task1, n_episodes=5,
                                 lora_layers=lora_layers, lora_enabled=True)
    print(f"    {task1}: SR={sr_pre*100:.1f}%  avg_R={r_pre:.2f} (expected ~0)")

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
        # PickCube: LoRA ON + current head
        sr_t1, r_t1 = eval_policy(model, task1,
                                   lora_layers=lora_layers, lora_enabled=True)
        # PokeCube: LoRA OFF + saved head (exact original behavior)
        sr_t0, r_t0 = eval_policy(model, task0,
                                   action_head=pokecube_head,
                                   lora_layers=lora_layers, lora_enabled=False)
        avg_sr = (sr_t0 + sr_t1) / 2

        pct = total_steps_done / TOTAL_TIMESTEPS * 100
        elapsed = (time.time() - phase1_start) / 60
        print(f"    {task0}: SR={sr_t0*100:6.1f}%  avg_R={r_t0:.2f}  "
              f"(LoRA OFF + saved head)")
        print(f"    {task1}: SR={sr_t1*100:6.1f}%  avg_R={r_t1:.2f}  "
              f"(LoRA ON + train head)")
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
            ckpt_best = os.path.join(CKPT_SAVE_DIR, "route_a_v10_best_avg.pt")
            torch.save({
                "state_dict": model.policy.state_dict(),
                "pokecube_head": pokecube_head,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
            }, ckpt_best)
            print(f"    * NEW BEST AVG! saved {ckpt_best}")

        if sr_t1 > best_pickcube_sr:
            best_pickcube_sr = sr_t1
            ckpt_best_pc = os.path.join(CKPT_SAVE_DIR, "route_a_v10_best_pickcube.pt")
            torch.save({
                "state_dict": model.policy.state_dict(),
                "pokecube_head": pokecube_head,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
            }, ckpt_best_pc)
            print(f"    * NEW BEST PICKCUBE! saved {ckpt_best_pc}")

        # 30% rule
        if total_steps_done >= int(TOTAL_TIMESTEPS * 0.3) and best_pickcube_sr == 0.0:
            if not no_sr_warning_fired:
                print(f"\n    *** 30% RULE: PickCube SR still 0% at {total_steps_done} steps! ***")
                no_sr_warning_fired = True
                ckpt_diag = os.path.join(CKPT_SAVE_DIR, "route_a_v10_30pct_diag.pt")
                torch.save({
                    "state_dict": model.policy.state_dict(),
                    "pokecube_head": pokecube_head,
                }, ckpt_diag)

        results.setdefault(task1, {})[f"step_{total_steps_done}"] = eval_record

        result_file = os.path.join(RESULT_DIR, "cl_results_route_a_v10_partial.json")
        with open(result_file, "w") as f:
            json.dump({"eval_history": eval_history, "best_avg_sr": best_avg_sr,
                        "best_pickcube_sr": best_pickcube_sr}, f, indent=2, default=float)

    phase1_time = (time.time() - phase1_start) / 60
    print(f"\n  Phase 1 done! Time: {phase1_time:.1f} min")

    # Load best avg
    ckpt_best = os.path.join(CKPT_SAVE_DIR, "route_a_v10_best_avg.pt")
    if os.path.isfile(ckpt_best):
        best_data = torch.load(ckpt_best, map_location="cpu", weights_only=False)
        model.policy.load_state_dict(best_data["state_dict"])
        pokecube_head = best_data["pokecube_head"]
        # Re-inject LoRA references after loading (LoRALinear modules are in state_dict)
        lora_layers = [m for m in model.policy.mlp_extractor.policy_net
                       if isinstance(m, LoRALinear)]
        print(f"  Loaded best avg (avg_SR={best_avg_sr*100:.1f}%), "
              f"found {len(lora_layers)} LoRA layers")

    # Final eval (50 episodes)
    print(f"\n  -- Final eval (50 episodes) --")
    sr_t1_f, r_t1_f = eval_policy(model, task1, n_episodes=50,
                                   lora_layers=lora_layers, lora_enabled=True)
    sr_t0_f, r_t0_f = eval_policy(model, task0, n_episodes=50,
                                   action_head=pokecube_head,
                                   lora_layers=lora_layers, lora_enabled=False)
    print(f"    {task0}: SR={sr_t0_f*100:.1f}%  avg_R={r_t0_f:.2f}  "
          f"(LoRA OFF + saved head)")
    print(f"    {task1}: SR={sr_t1_f*100:.1f}%  avg_R={r_t1_f:.2f}")

    # Save final
    ckpt1_path = os.path.join(CKPT_SAVE_DIR, "route_a_v10_final.pt")
    torch.save({
        "state_dict": model.policy.state_dict(),
        "pokecube_head": pokecube_head,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
    }, ckpt1_path)

    # ─────────────────────────────────────────────────────────────
    #  RESULTS
    # ─────────────────────────────────────────────────────────────
    forgetting_pokecube = sr0 - sr_t0_f

    all_results = {
        "route": "A_v10",
        "strategy": "frozen_backbone_lora_adapters_task_heads",
        "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "scale": LORA_ALPHA/LORA_RANK},
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
        },
        "results": {
            "phase0_official": {task0: {"SR": sr0, "avg_R": r0}},
            "phase1_final": {
                task0: {"SR": sr_t0_f, "avg_R": r_t0_f,
                        "eval_method": "lora_off_saved_head"},
                task1: {"SR": sr_t1_f, "avg_R": r_t1_f,
                        "eval_method": "lora_on_train_head"},
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

    result_file = os.path.join(RESULT_DIR, "cl_results_route_a_v10.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'='*80}")
    print(f"  Route A v10 Results: {task0} -> {task1}")
    print(f"{'='*80}")
    print(f"  {'Task':<20s} {'Before CL':>12s} {'After CL':>12s} {'Forget':>10s}")
    print(f"  {'-'*54}")
    print(f"  {task0:<20s} {sr0*100:>10.1f}%  {sr_t0_f*100:>10.1f}%  "
          f"{forgetting_pokecube*100:>+8.1f}%")
    print(f"  {task1:<20s} {'N/A':>10s}  {sr_t1_f*100:>10.1f}%  {'N/A':>10s}")
    print(f"\n  Best PickCube SR: {best_pickcube_sr*100:.1f}%")
    print(f"  PokeCube forgetting: {forgetting_pokecube*100:+.1%} "
          f"(LoRA OFF guarantees ~0)")
    print(f"  Results saved: {result_file}")

    train_envs.close()
    print("\nDONE")
