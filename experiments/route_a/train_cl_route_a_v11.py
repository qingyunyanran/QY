"""
ManiSkill3 CL - Route A v11
  Per-Task Input Encoder + Frozen Backbone + LoRA + Task Heads

  THE KEY FIX vs v9/v10:
  v9/v10 zero-padded PickCube 42dim -> 54dim. The frozen first layer was
  trained on PokeCube's 54dim obs (with meaningful dims 42-54). Even with
  LoRA, the input representation was fundamentally mismatched.
  PickCube plateaued at avg_R~20 (reach but no grasp) across rank 8 and 16.

  v11 gives each task its OWN input encoder:
  - PokeCube: Linear(54->256) from official checkpoint, FROZEN
  - PickCube: Linear(42->256) fresh, TRAINABLE (sees real 42dim obs, no padding)
  Both output 256dim features into shared hidden layers with LoRA.

  Architecture:
    obs -> [TaskEncoder] -> 256 -> [LoRA 256->256] -> [LoRA 256->256] -> head
    PokeCube eval: PokeCube encoder + LoRA OFF + PokeCube head = zero forgetting
    PickCube train/eval: PickCube encoder + LoRA ON + PickCube head = new skill
"""
import os, sys, time, json, math
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
#  Fix terminal_observation + obs_to_tensor
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
#  LoRA
# ═══════════════════════════════════════════════════════════════════
class LoRALinear(nn.Module):
    """Linear layer with LoRA: output = original(x) + scale * x @ A @ B"""
    def __init__(self, original_linear, rank=8, alpha=16.0):
        super().__init__()
        self.original = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scale = alpha / rank
        self.enabled = True

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, self.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out = self.original(x)
        if self.enabled:
            out = out + (x @ self.lora_A @ self.lora_B) * self.scale
        return out

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


# ═══════════════════════════════════════════════════════════════════
#  DUAL INPUT LAYER — the key v11 innovation
# ═══════════════════════════════════════════════════════════════════
class DualTaskInputLayer(nn.Module):
    """Routes to task-specific input encoder.

    PokeCube: frozen Linear(54->256) with official weights.
    PickCube: fresh Linear(42->256), only sees first 42 dims (no zero-padding).
    """
    def __init__(self, pokecube_linear, pickcube_obs_dim=42, hidden_dim=256):
        super().__init__()
        # PokeCube encoder: official weights, frozen
        self.pokecube_enc = pokecube_linear
        self.pokecube_enc.weight.requires_grad = False
        if self.pokecube_enc.bias is not None:
            self.pokecube_enc.bias.requires_grad = False
        self.pokecube_obs_dim = pokecube_linear.in_features  # 54

        # PickCube encoder: fresh, trainable
        self.pickcube_enc = nn.Linear(pickcube_obs_dim, hidden_dim)
        nn.init.orthogonal_(self.pickcube_enc.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.pickcube_enc.bias)

        self.current_task = "PokeCube-v1"

    def forward(self, x):
        if self.current_task == "PokeCube-v1":
            return self.pokecube_enc(x)
        else:
            # Only use first 42 dims — the rest are zero-padding that we ignore
            return self.pickcube_enc(x[:, :self.pickcube_enc.in_features])

    def set_task(self, task):
        self.current_task = task


# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
TASK_SEQUENCE = ["PokeCube-v1", "PickCube-v1"]
TASK_CONFIGS = {
    "PokeCube-v1":  {"raw_obs_dim": 54, "act_dim": 8},
    "PickCube-v1":  {"raw_obs_dim": 42, "act_dim": 8},
}
UNIFIED_OBS_DIM = 54  # for SB3 observation space, but PickCube encoder ignores padding
CONTROL_MODE = "pd_joint_delta_pos"

TOTAL_TIMESTEPS = 3_000_000
N_ENVS = 4
NET_ARCH_PI = [256, 256, 256]
NET_ARCH_VF = [256, 256, 256]
ACTIVATION_FN = nn.Tanh

LORA_RANK = 16
LORA_ALPHA = 32.0  # higher alpha for stronger LoRA signal

LR_START = 5e-4
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
    return {
        "action_net.weight": sd["action_net.weight"].clone(),
        "action_net.bias": sd["action_net.bias"].clone(),
        "log_std": sd["log_std"].clone(),
    }


def load_action_head(sd, head):
    new_sd = {k: v.clone() for k, v in sd.items()}
    new_sd["action_net.weight"] = head["action_net.weight"].clone()
    new_sd["action_net.bias"] = head["action_net.bias"].clone()
    new_sd["log_std"] = head["log_std"].clone()
    return new_sd


def reinit_action_head(sd, act_dim=8):
    nn.init.orthogonal_(sd["action_net.weight"], gain=0.01)
    nn.init.zeros_(sd["action_net.bias"])
    sd["log_std"] = torch.zeros(act_dim)
    print("  Reinitialized action_net + log_std")
    return sd


# ═══════════════════════════════════════════════════════════════════
#  WRAPPERS
# ═══════════════════════════════════════════════════════════════════
class UnifiedObsWrapper(gym.Wrapper):
    """Zero-pad to 54 for SB3 compatibility, but v11 encoder ignores padding."""
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
def eval_policy(model, task_id, n_episodes=EVAL_EPS, action_head=None,
                lora_layers=None, lora_enabled=True, input_layer=None):
    policy = model.policy
    saved_head = None
    prev_lora = None
    prev_task = None

    if input_layer is not None:
        prev_task = input_layer.current_task
        input_layer.set_task(task_id)

    if lora_layers is not None:
        prev_lora = [l.enabled for l in lora_layers]
        for l in lora_layers:
            l.enabled = lora_enabled

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

    if saved_head is not None:
        restored = load_action_head(policy.state_dict(), saved_head)
        policy.load_state_dict(restored, strict=False)
    if lora_layers is not None and prev_lora is not None:
        for l, p in zip(lora_layers, prev_lora):
            l.enabled = p
    if input_layer is not None and prev_task is not None:
        input_layer.set_task(prev_task)

    return float(np.mean(successes)), float(np.mean(total_rewards))


# ═══════════════════════════════════════════════════════════════════
#  BUILD MULTI-TASK POLICY NETWORK
# ═══════════════════════════════════════════════════════════════════
def build_multitask_policy_net(official_first_layer_weight, official_first_layer_bias,
                                lora_rank=16, lora_alpha=32.0):
    """Build policy_net with DualTaskInputLayer as layer 0 + LoRA on layers 2,4.

    Structure:
      [0] DualTaskInputLayer (54->256 PokeCube frozen, 42->256 PickCube fresh)
      [1] Tanh
      [2] LoRALinear(256->256)
      [3] Tanh
      [4] LoRALinear(256->256)
      [5] Tanh
    """
    hidden_dim = 256

    # PokeCube input encoder with official weights
    pokecube_linear = nn.Linear(54, hidden_dim)
    pokecube_linear.weight.data = official_first_layer_weight.clone()
    pokecube_linear.bias.data = official_first_layer_bias.clone()

    # Dual input layer
    input_layer = DualTaskInputLayer(
        pokecube_linear, pickcube_obs_dim=42, hidden_dim=hidden_dim
    )

    # Hidden layers with LoRA
    hidden2 = LoRALinear(nn.Linear(hidden_dim, hidden_dim), rank=lora_rank, alpha=lora_alpha)
    hidden4 = LoRALinear(nn.Linear(hidden_dim, hidden_dim), rank=lora_rank, alpha=lora_alpha)

    policy_net = nn.Sequential(
        input_layer,   # 0
        nn.Tanh(),     # 1
        hidden2,       # 2
        nn.Tanh(),     # 3
        hidden4,       # 4
        nn.Tanh(),     # 5
    )

    lora_layers = [hidden2, hidden4]

    print(f"  Built multi-task policy_net:")
    print(f"    [0] DualTaskInputLayer: PokeCube 54->256 (frozen), PickCube 42->256 (train)")
    print(f"    [2] LoRALinear 256->256 (rank={lora_rank}, alpha={lora_alpha})")
    print(f"    [4] LoRALinear 256->256 (rank={lora_rank}, alpha={lora_alpha})")

    return policy_net, lora_layers, input_layer


def setup_multitask_policy(model, sb3_sd, lora_rank=16, lora_alpha=32.0):
    """Replace model's policy_net with multi-task version, load weights correctly."""

    # Build new policy_net
    official_w = sb3_sd["mlp_extractor.policy_net.0.weight"]
    official_b = sb3_sd["mlp_extractor.policy_net.0.bias"]

    policy_net, lora_layers, input_layer = build_multitask_policy_net(
        official_w, official_b, lora_rank=lora_rank, lora_alpha=lora_alpha
    )

    # Replace the model's policy_net
    model.policy.mlp_extractor.policy_net = policy_net

    # Load hidden layer weights into LoRA originals
    # Layer 2 in new = LoRALinear wrapping official layer 2
    policy_net[2].original.weight.data = sb3_sd["mlp_extractor.policy_net.2.weight"].clone()
    policy_net[2].original.bias.data = sb3_sd["mlp_extractor.policy_net.2.bias"].clone()
    # Layer 4
    policy_net[4].original.weight.data = sb3_sd["mlp_extractor.policy_net.4.weight"].clone()
    policy_net[4].original.bias.data = sb3_sd["mlp_extractor.policy_net.4.bias"].clone()

    # Load action head and log_std (will be reinitialized for PickCube later)
    model.policy.action_net.weight.data = sb3_sd["action_net.weight"].clone()
    model.policy.action_net.bias.data = sb3_sd["action_net.bias"].clone()
    model.policy.log_std.data = sb3_sd["log_std"].clone()

    # Freeze original LoRA weights
    for lora_layer in lora_layers:
        lora_layer.original.weight.requires_grad = False
        if lora_layer.original.bias is not None:
            lora_layer.original.bias.requires_grad = False

    # Count params
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.policy.parameters() if not p.requires_grad)
    total = trainable + frozen
    lora_p = sum(p.numel() for n, p in model.policy.named_parameters()
                 if p.requires_grad and "lora_" in n)
    pickcube_enc_p = sum(p.numel() for n, p in model.policy.named_parameters()
                         if p.requires_grad and "pickcube_enc" in n)

    print(f"\n  Parameter breakdown:")
    print(f"    PickCube encoder:     {pickcube_enc_p:>8d}")
    print(f"    LoRA params:          {lora_p:>8d}")
    print(f"    Other trainable:      {trainable - lora_p - pickcube_enc_p:>8d}")
    print(f"    Total trainable:      {trainable:>8d} ({trainable/total*100:.1f}%)")
    print(f"    Frozen:               {frozen:>8d} ({frozen/total*100:.1f}%)")

    # Rebuild optimizer
    trainable_params = [p for p in model.policy.parameters() if p.requires_grad]
    lr = model.lr_schedule(1.0) if callable(model.lr_schedule) else model.learning_rate
    model.policy.optimizer = torch.optim.Adam(trainable_params, lr=lr, eps=1e-5)
    print(f"  Optimizer rebuilt: {len(trainable_params)} groups, lr={lr}")

    return lora_layers, input_layer


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("  ManiSkill3 CL - Route A v11 (Per-Task Input Encoder + LoRA)")
    print("=" * 80)
    print(f"  LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}")
    print(f"  KEY: PickCube gets dedicated 42->256 encoder (no zero-padding issue)")
    print(f"  Steps: {TOTAL_TIMESTEPS}  envs: {N_ENVS}")
    print(f"  lr={LR_START}->{LR_END}, ent={ENT_COEF}, gamma={GAMMA}")
    print("=" * 80)

    task0 = "PokeCube-v1"
    task1 = "PickCube-v1"

    # ── Phase 0 ──
    print(f"\n{'='*80}\n  Phase 0: Load official {task0}\n{'='*80}")

    if not os.path.isfile(OFFICIAL_CKPT):
        print(f"  ERROR: {OFFICIAL_CKPT} not found"); sys.exit(1)

    sb3_sd = convert_official_to_sb3(OFFICIAL_CKPT)

    train_envs = FixedDummyVecEnv([make_env(task0, i) for i in range(N_ENVS)])
    model = PPO(
        "MlpPolicy", train_envs,
        policy_kwargs=dict(net_arch=dict(pi=NET_ARCH_PI, vf=NET_ARCH_VF),
                           activation_fn=ACTIVATION_FN),
        learning_rate=LR_START, ent_coef=ENT_COEF,
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip_range=CLIP_RANGE,
        target_kl=TARGET_KL, max_grad_norm=MAX_GRAD_NORM, verbose=1,
    )
    model.policy.load_state_dict(sb3_sd, strict=False)
    print("  Official weights loaded (standard MlpPolicy for Phase 0 eval)")

    print(f"\n  -- Eval {task0} --")
    sr0, r0 = eval_policy(model, task0)
    print(f"    {task0}: SR={sr0*100:.1f}%  avg_R={r0:.2f}")

    if sr0 < 0.1:
        print(f"  *** WARNING: SR very low! ***")

    pokecube_head = save_action_head(model.policy.state_dict())
    print("  Saved PokeCube action head")

    torch.save(model.policy.state_dict(),
               os.path.join(CKPT_SAVE_DIR, "route_a_v11_phase0.pt"))

    # ── Phase 1: Multi-task policy ──
    print(f"\n{'='*80}\n  Phase 1: {task1} with per-task encoder + LoRA\n{'='*80}")

    train_envs.close()
    train_envs = FixedDummyVecEnv([make_env(task1, i) for i in range(N_ENVS)])

    lr_schedule = get_linear_fn(LR_START, LR_END, 1.0)
    model = PPO(
        "MlpPolicy", train_envs,
        policy_kwargs=dict(net_arch=dict(pi=NET_ARCH_PI, vf=NET_ARCH_VF),
                           activation_fn=ACTIVATION_FN),
        learning_rate=lr_schedule, ent_coef=ENT_COEF,
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip_range=CLIP_RANGE,
        target_kl=TARGET_KL, max_grad_norm=MAX_GRAD_NORM, verbose=1,
    )

    # Build multi-task policy network
    lora_layers, input_layer = setup_multitask_policy(
        model, sb3_sd, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA
    )

    # Set current task to PickCube for training
    input_layer.set_task(task1)

    # Reinit action head for PickCube
    sd = model.policy.state_dict()
    sd = reinit_action_head(sd, act_dim=8)
    model.policy.load_state_dict(sd, strict=False)
    # Need to re-freeze after load_state_dict
    for lora_layer in lora_layers:
        lora_layer.original.weight.requires_grad = False
        if lora_layer.original.bias is not None:
            lora_layer.original.bias.requires_grad = False
    input_layer.pokecube_enc.weight.requires_grad = False
    input_layer.pokecube_enc.bias.requires_grad = False

    # Rebuild optimizer again after head reinit
    trainable_params = [p for p in model.policy.parameters() if p.requires_grad]
    lr = model.lr_schedule(1.0) if callable(model.lr_schedule) else model.learning_rate
    model.policy.optimizer = torch.optim.Adam(trainable_params, lr=lr, eps=1e-5)

    # Sanity checks
    print(f"\n  -- Sanity 1: {task0} (PokeCube enc + LoRA OFF + PokeCube head) --")
    sr_s1, r_s1 = eval_policy(model, task0, n_episodes=5,
                               action_head=pokecube_head,
                               lora_layers=lora_layers, lora_enabled=False,
                               input_layer=input_layer)
    print(f"    SR={sr_s1*100:.1f}%  R={r_s1:.2f} (should be ~{sr0*100:.0f}%)")

    print(f"  -- Sanity 2: {task1} (PickCube enc + LoRA ON + fresh head, pre-train) --")
    sr_s2, r_s2 = eval_policy(model, task1, n_episodes=5,
                               lora_layers=lora_layers, lora_enabled=True,
                               input_layer=input_layer)
    print(f"    SR={sr_s2*100:.1f}%  R={r_s2:.2f} (expected ~0)")

    if sr_s1 < sr0 * 0.8:
        print(f"  *** WARNING: PokeCube dropped after architecture change! ***")

    # Training loop
    phase1_start = time.time()
    best_avg_sr = 0.0
    best_pickcube_sr = 0.0
    total_steps_done = 0
    eval_history = []
    warned_30 = False

    while total_steps_done < TOTAL_TIMESTEPS:
        steps = min(EVAL_FREQ, TOTAL_TIMESTEPS - total_steps_done)
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        total_steps_done += steps

        # Ensure we're in PickCube mode for training
        input_layer.set_task(task1)
        for l in lora_layers:
            l.enabled = True

        print(f"\n  -- [eval @ {total_steps_done}/{TOTAL_TIMESTEPS}] --")
        sr_t1, r_t1 = eval_policy(model, task1,
                                   lora_layers=lora_layers, lora_enabled=True,
                                   input_layer=input_layer)
        sr_t0, r_t0 = eval_policy(model, task0,
                                   action_head=pokecube_head,
                                   lora_layers=lora_layers, lora_enabled=False,
                                   input_layer=input_layer)
        avg_sr = (sr_t0 + sr_t1) / 2

        elapsed = (time.time() - phase1_start) / 60
        pct = total_steps_done / TOTAL_TIMESTEPS * 100
        print(f"    {task0}: SR={sr_t0*100:6.1f}%  R={r_t0:.2f}  (frozen enc + LoRA OFF)")
        print(f"    {task1}: SR={sr_t1*100:6.1f}%  R={r_t1:.2f}  (PickCube enc + LoRA ON)")
        print(f"    avg={avg_sr*100:.1f}%  best_avg={best_avg_sr*100:.1f}%  "
              f"best_pc={best_pickcube_sr*100:.1f}%  {elapsed:.0f}min {pct:.0f}%")

        eval_history.append({
            "step": total_steps_done,
            task0: {"SR": sr_t0, "R": r_t0},
            task1: {"SR": sr_t1, "R": r_t1},
            "avg": avg_sr, "min": elapsed,
        })

        if avg_sr > best_avg_sr:
            best_avg_sr = avg_sr
            torch.save({"sd": model.policy.state_dict(), "head": pokecube_head,
                        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA},
                       os.path.join(CKPT_SAVE_DIR, "route_a_v11_best_avg.pt"))
            print(f"    * BEST AVG!")

        if sr_t1 > best_pickcube_sr:
            best_pickcube_sr = sr_t1
            torch.save({"sd": model.policy.state_dict(), "head": pokecube_head,
                        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA},
                       os.path.join(CKPT_SAVE_DIR, "route_a_v11_best_pc.pt"))
            print(f"    * BEST PICKCUBE!")

        if total_steps_done >= TOTAL_TIMESTEPS * 0.3 and best_pickcube_sr == 0 and not warned_30:
            print(f"\n    *** 30% RULE: PickCube still 0% at {total_steps_done} ***")
            warned_30 = True

        with open(os.path.join(RESULT_DIR, "cl_v11_partial.json"), "w") as f:
            json.dump({"history": eval_history, "best_avg": best_avg_sr,
                        "best_pc": best_pickcube_sr}, f, indent=2, default=float)

    # Final eval
    best_path = os.path.join(CKPT_SAVE_DIR, "route_a_v11_best_avg.pt")
    if os.path.isfile(best_path):
        d = torch.load(best_path, map_location="cpu", weights_only=False)
        model.policy.load_state_dict(d["sd"])
        pokecube_head = d["head"]
        # Re-find LoRA and input layers
        input_layer = model.policy.mlp_extractor.policy_net[0]
        lora_layers = [m for m in model.policy.mlp_extractor.policy_net
                       if isinstance(m, LoRALinear)]
        print(f"  Loaded best avg ({best_avg_sr*100:.1f}%)")

    print(f"\n  -- Final (50 eps) --")
    sr1f, r1f = eval_policy(model, task1, n_episodes=50,
                             lora_layers=lora_layers, lora_enabled=True,
                             input_layer=input_layer)
    sr0f, r0f = eval_policy(model, task0, n_episodes=50,
                             action_head=pokecube_head,
                             lora_layers=lora_layers, lora_enabled=False,
                             input_layer=input_layer)

    forget = sr0 - sr0f
    print(f"\n  {task0}: {sr0*100:.1f}% -> {sr0f*100:.1f}% (forget {forget*100:+.1f}%)")
    print(f"  {task1}: -> {sr1f*100:.1f}% (best {best_pickcube_sr*100:.1f}%)")

    torch.save({"sd": model.policy.state_dict(), "head": pokecube_head,
                "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA},
               os.path.join(CKPT_SAVE_DIR, "route_a_v11_final.pt"))

    results = {
        "route": "A_v11",
        "strategy": "per_task_input_encoder_lora",
        "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA},
        "results": {
            "phase0": {task0: {"SR": sr0, "R": r0}},
            "final": {task0: {"SR": sr0f, "R": r0f},
                      task1: {"SR": sr1f, "R": r1f}},
        },
        "forgetting": {task0: forget},
        "best_pickcube": best_pickcube_sr,
        "history": eval_history,
    }
    with open(os.path.join(RESULT_DIR, "cl_results_v11.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    train_envs.close()
    print("\nDONE")
