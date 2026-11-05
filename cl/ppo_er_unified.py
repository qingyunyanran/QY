"""
ManiSkill3 CL - PPO + Experience Replay  (UNIFIED 42-dim observation)

Same training logic as ppo_er.py, but observations are aligned BY PHYSICAL
MEANING into the fixed 42-dim semantic-slot interface (see unified_obs.py):
  [0:18] agent proprioception
  [18:25] tcp pose
  [25:32] object slot 1 (primary manipulated object)
  [32:39] object slot 2 (second object / tool; zero for single-object tasks)
  [39:42] goal position (zero for tasks without a goal site, e.g.
           LiftPegUpright, where success is implicit in slot 1 orientation)

NO learned mapping layer, NO task one-hot: the same input position has the
same physical meaning in every task, so the whole network is shared and
observations need no padding/normalization tricks. The redundant relative
vectors (differences of absolute poses) are dropped; absolute poses are kept.

ER strategy (canonical simple ER):
  1. Train PPO on task 1, store all rollout transitions in a PER-TASK ring
     buffer
  2. On later tasks, before each PPO update sample old transitions UNIFORMLY
     ACROSS PAST TASKS and add a behavior-cloning regularizer (keep policy
     actions on old-task states)
  3. Repeat. A per-task buffer is required because each task generates ~2M
     transitions: a single global 200k FIFO would only hold the tail of the
     most recent task, leaving earlier tasks unprotected.

Task suite (4 tasks, native episode horizons, identical to the single-task
baselines): PushCube-v1, PullCube-v1, PickCube-v1, LiftPegUpright-v1.

Usage:
    cd E:\\munichi\\maniskill-cl\\cl
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    python ppo_er_unified.py
    python ppo_er_unified.py --seqft
    python ppo_er_unified.py --task-sequence PushCube-v1,PullCube-v1 --total-steps 2000000
    # per-task budgets (PickCube needs ~5.12M steps single-task, give it 6M):
    python ppo_er_unified.py --task-steps 2000000,2000000,6000000,2000000
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_conda = r"E:\My_programs\anaconda\envs\diffcl10\Library\bin"
if _conda not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda + os.pathsep + os.environ.get("PATH", "")

import gymnasium as gym
import mani_skill.envs

from unified_obs import UNIFIED_OBS_DIM, SLOT_LAYOUT, build_unified_obs
from vec_env import GpuVecEnv, evaluate_vec, as_tensor

# ============================================================
# Config
# ============================================================

# All tasks share ONE observation space: the 42-dim semantic slots.
# control_mode: Push/Pick were tuned with pd_joint_delta_pos in ppo_er.py;
# Pull/Poke official baselines also use pd_joint_delta_pos. Keep it uniform.
TASK_INFO = {
    "PushCube-v1":       {"control_mode": "pd_joint_delta_pos"},
    "PullCube-v1":       {"control_mode": "pd_joint_delta_pos"},
    "PickCube-v1":       {"control_mode": "pd_joint_delta_pos"},
    "LiftPegUpright-v1": {"control_mode": "pd_joint_delta_pos"},
    # StackCube: grasp cubeA, stack it on cubeB, release and let it settle.
    # The 42-dim mapping already exists in unified_obs.SLOT_LAYOUT
    # (slot1=cubeA_pose, slot2=cubeB_pose, goal=None -> [39:42] zero-padded):
    # the raw 48-dim obs loses exactly the three relative vectors
    # tcp_to_cubeA_pos / tcp_to_cubeB_pos / cubeA_to_cubeB_pos, which are linear
    # differences of poses we keep.
    # `max_episode_steps=100` overrides the registered default (50). StackCube
    # must grasp cubeA, carry it, place it on cubeB, release and let the stack
    # settle -- that sequence does not fit in 50 control steps. The four easy
    # tasks keep their native 50.
    "StackCube-v1":      {"control_mode": "pd_joint_delta_pos",
                          "max_episode_steps": 100},
    # ---- COMPLEX-SKILL SEQUENCE (probe 2026-11-03) ------------------------
    # Same embodiment (Panda, 8-dim pd_joint_delta_pos) and the same 48-dim
    # unified interface; only the horizon differs. Horizons are the values the
    # environments are registered with (StackPyramid 250, PegInsertionSide 100,
    # PlugCharger 200, PokeCube 50); they are listed explicitly so a registry
    # change cannot silently rescale the replay buffer's episode statistics.
    "StackPyramid-v1":   {"control_mode": "pd_joint_delta_pos",
                          "max_episode_steps": 250},
    "PegInsertionSide-v1": {"control_mode": "pd_joint_delta_pos",
                            "max_episode_steps": 100},
    "PlugCharger-v1":    {"control_mode": "pd_joint_delta_pos",
                          "max_episode_steps": 200},
    # PokeCube needs the peg (a loose 24 cm tool lying on the table) grasped
    # first and then 2-3 separate pushes on the cube; 50 steps only fits the
    # reach. 8M steps @ H=50 plateaued at 65-70%, so the budget is doubled to
    # 100 like StackCube. Success criterion is untouched.
    "PokeCube-v1":       {"control_mode": "pd_joint_delta_pos",
                          "max_episode_steps": 100},
    # ---- REPLACEMENT TASKS (reward audit 2026-11-03) --------------------
    # StackPyramid (H=250) and PlugCharger (H=200) were dropped from the
    # sequence: both declare SUPPORTED_REWARD_MODES = {"none","sparse"}, so the
    # only signal is the 0/1 success flag (measured reward == 0.000 exactly
    # over 490k steps on StackPyramid). These two replace them and both expose
    # a dense reward, so no shaping is invented on our side.
    "PlaceSphere-v1":    {"control_mode": "pd_joint_delta_pos",
                          "max_episode_steps": 50},
    "PullCubeTool-v1":   {"control_mode": "pd_joint_delta_pos",
                          "max_episode_steps": 100},
    "RollBall-v1":       {"control_mode": "pd_joint_delta_pos",
                          "max_episode_steps": 80},
    # PickSingleYCB: grasp one randomly sampled YCB object and carry it to a goal
    # point, ending with the arm static. Native horizon is 50; it is raised to 75
    # because the sampled objects differ in size (some need a two-sided grasp and
    # a longer carry), the same reasoning that doubled PokeCube's horizon.
    "PickSingleYCB-v1":  {"control_mode": "pd_joint_delta_pos",
                          "max_episode_steps": 75},
}

# Two hyperparameter sets. The discount/gain terms are IDENTICAL (official
# ManiSkill PPO: gamma=0.8, gae_lambda=0.9, lr 3e-4, zero entropy bonus, 4
# epochs with target-KL early stop, no value clipping). Only the rollout shape
# differs, because it is dictated by how the environments can be vectorised:
#
#   CPU backend -- ManiSkill forbids num_envs>1 with sim_backend="cpu" inside
#   one process, so N single-env sims are stepped in a Python loop. 16 x 512
#   keeps the batch (8192) comparable while amortising Python overhead.
#
#   GPU backend -- one env object holds 512 parallel sims on the GPU. This is
#   the official setup (512 x 50 = 25,600 batch, 32 minibatches of 800) and the
#   only regime in which PPO has a chance on StackCube: the earlier failure ran
#   8 envs x 5M steps and collapsed (see StackCube_PPO_失败分析报告.md).
PPO_CONFIG_CPU = {
    "learning_rate": 3e-4,
    "num_envs": 16,
    "num_steps": 512,
    "num_eval_envs": 2,
    "gamma": 0.8,
    "gae_lambda": 0.9,
    "num_minibatches": 16,
    "update_epochs": 4,
    "norm_adv": True,
    "clip_coef": 0.2,
    "clip_vloss": False,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "anneal_lr": False,
    "reward_scale": 1.0,
    "bc_coef": 0.05,
    "target_kl": 0.1,
}

PPO_CONFIG_GPU = dict(PPO_CONFIG_CPU)
PPO_CONFIG_GPU.update({
    "num_envs": 512,
    "num_steps": 50,
    "num_minibatches": 32,
})

# Default kept at the CPU profile so the original reproduction command still
# behaves identically; the CLI switches it via --sim-backend.
PPO_CONFIG = PPO_CONFIG_CPU

ER_CONFIG = {
    "buffer_per_task": 200_000,   # per-task ring-buffer capacity
    "replay_ratio": 0.5,
    "sample_strategy": "uniform_across_tasks",
}

# Exploration std (log) applied after a policy-only warm start. exp(-1.5)=0.22
# keeps rollouts close to the seeded expert while still leaving exploration
# room; the stock -0.5 (std 0.6) is far too wide for a converged policy.
PRETRAIN_LOGSTD = -1.0
# Minimum per-sample log-prob accepted by the ER behaviour-cloning regulariser.
# Replay actions were sampled by the policy of the task that produced them; if
# the shared actor has since moved (or its std has been re-tightened), those
# actions fall tens of std away and -logp explodes (O(100) per sample), which
# makes the 0.05-weighted BC term dominate the PPO loss and drags the whole
# network into a compromise that solves nothing. Clamping is the Huber-isation
# of that term: it still pulls the policy towards old actions when they are
# plausible under the current policy, and stops caring once they are hopeless.
BC_LOGPROB_CLAMP = 10.0
# When the actor is frozen for the whole warm-started phase the rollout is only
# there to fit the critic, so track the expert tightly: std 0.135 keeps the
# collected returns close to the greedy policy the critic has to evaluate.
PRETRAIN_LOGSTD_FROZEN = -2.0

# A warm-started task begins from an expert policy, so the stock optimiser
# schedule is far too aggressive for it: the value head has just been
# re-initialised, so the first advantage batch is enormous, and target_kl=0.1
# happily accepts a 0.12-KL jump inside a single epoch. Measured on the 5-task
# sequence: the seeded StackCube expert (97%) was gone within 10 updates
# (train_SR 4% -> 0%) and the blast radius took PickCube 95% -> 0% and
# LiftPeg 95% -> 10% with it. Freeze the actor long enough for the critic to
# fit the new reward scale, then adapt at a small step size.
PRETRAIN_FREEZE_ITERS = 25
PRETRAIN_LR = 1e-4
PRETRAIN_TARGET_KL = 0.02

DEFAULT_TASK_SEQUENCE = ["PushCube-v1", "PullCube-v1", "PickCube-v1", "LiftPegUpright-v1"]
# Target benchmark: the same four tasks with StackCube appended LAST. The order
# matters -- ER keeps every earlier task in its buffer, so appending the hard
# task tests exactly the claim we want to make: a new hard skill can be added
# without forgetting the four easy ones.
FIVE_TASK_SEQUENCE = DEFAULT_TASK_SEQUENCE + ["StackCube-v1"]

# Target benchmark round 2: five COMPLEX skills only (no easy tasks), ordered
# by increasing difficulty. StackCube stays first because its 48-dim mapping
# is already the reference implementation the other four are aligned to.
COMPLEX_TASK_SEQUENCE = ["StackCube-v1", "PlaceSphere-v1", "PokeCube-v1",
                         "PullCubeTool-v1", "PegInsertionSide-v1"]

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ppo_er_unified")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts", "ppo_er_unified")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ============================================================
# Agent (same architecture as official ManiSkill PPO)
# ============================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """3-layer Tanh MLP, shared actor-critic (separate heads).

    MULTI-HEAD MODE (n_heads > 1):  SHARED BASE + PER-TASK ADAPTER
    -------------------------------------------------------------
    The first two layers (obs -> 256 -> 256) are shared by every task; the last
    hidden layer plus the output layer are private per task. Sharing only the
    base and letting each task own the top of the network is what keeps two
    tasks with opposite feature needs from fighting over one representation.

    Measured, per-task head on a fully shared trunk (c6v3_mh, 100 episodes):
    StackCube 78% -> 27% while PegInsertionSide only reached 49% (35% right
    after its own phase) -- the interference runs both ways. Lowering the trunk
    lr to 0.1 did not help, and 0.3 was worse for Peg. Re-fitting just the heads
    with the trunk frozen recovered only 56/58.

    Layout, so that head 0 keeps the historical `actor_mean` keys:
      actor_mean[0:4]  = shared base  (Linear, Tanh, Linear, Tanh)
      actor_mean[4:7]  = task 0 adapter (Linear, Tanh, Linear->action)
      actor_heads_extra[k-1] = same adapter for task k
    This is the same 7-entry shape as the single-head net, so a single-head
    expert checkpoint loads into task 0 unchanged and reproduces the expert
    exactly (base = its first two layers, adapter = its last hidden layer plus
    output layer).
    """
    def __init__(self, obs_dim, action_dim, hidden=256, n_heads=1,
                 private_base_ids=()):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)
        self.action_dim = action_dim
        self.n_heads = max(1, int(n_heads))
        # Plain lists (NOT nn.ModuleList) so the pieces are not re-registered as
        # separate submodules: same tensors, same updates, single optimiser
        # entry, and single-head checkpoints keep loading with strict=True.
        # SHARED_BASE_LAYERS = how many leading entries of actor_mean every task
        # sees; the rest (last hidden layer + output layer) is per task.
        if self.n_heads > 1:
            self._trunk = list(self.actor_mean[:Agent.SHARED_BASE_LAYERS])
            self._adapter0 = list(self.actor_mean[Agent.SHARED_BASE_LAYERS:])
            self.actor_heads_extra = nn.ModuleList([
                nn.Sequential(
                    layer_init(nn.Linear(hidden, hidden)),
                    nn.Tanh(),
                    layer_init(nn.Linear(hidden, action_dim), std=0.01 * np.sqrt(2)),
                ) for _ in range(self.n_heads - 1)])
        else:
            self._trunk = None
            self._adapter0 = list(self.actor_mean)
        # Outlier tasks (PegInsertionSide) get a private base as well: measured,
        # the two requirements are opposite -- with the base plastic for
        # StackCube (mult 0.5) Peg drops to 60%, with it held still (0.1) Stack
        # never leaves 0%. Isolation is per task, so only the outlier pays for
        # it; the other tasks still share one base and transfer between them.
        self.private_base_ids = {int(i) for i in private_base_ids}
        if self.n_heads > 1 and self.private_base_ids:
            self.private_nets = nn.ModuleDict({
                str(tid): nn.Sequential(
                    layer_init(nn.Linear(obs_dim, hidden)),
                    nn.Tanh(),
                    layer_init(nn.Linear(hidden, hidden)),
                    nn.Tanh(),
                    layer_init(nn.Linear(hidden, hidden)),
                    nn.Tanh(),
                    layer_init(nn.Linear(hidden, action_dim), std=0.01 * np.sqrt(2)),
                ) for tid in sorted(self.private_base_ids)})

    # How many leading actor_mean entries are the shared base: the first Linear
    # and the second Linear with their Tanh. The remaining 3 entries form the
    # per-task adapter (Linear, Tanh, Linear->action).
    SHARED_BASE_LAYERS = 4

    def adapter_seq(self, tid):
        """The per-task adapter block (Sequential) for task `tid`."""
        return self.actor_mean[Agent.SHARED_BASE_LAYERS:] if int(tid) <= 0 \
            else self.actor_heads_extra[int(tid) - 1]

    def head_mean(self, x, tid=0):
        """Actor mean from the head of task `tid`.

        `tid` may be an int (whole batch belongs to one task) or a LongTensor
        with one task index per row, which is what the replay-based terms need
        (a minibatch mixes rows from several tasks).
        """
        if self._trunk is None:
            return self.actor_mean(x)
        if torch.is_tensor(tid):
            if tid.numel() == 1:
                return self.head_mean(x, int(tid.reshape(-1)[0].item()))
            out = torch.zeros((x.shape[0], self.action_dim),
                              dtype=x.dtype, device=x.device)
            for t in torch.unique(tid).tolist():
                sel = tid == t
                out[sel] = self.head_mean(x[sel], int(t))
            return out
        tid = int(tid)
        _priv = getattr(self, "private_nets", None)
        if _priv is not None and str(tid) in _priv:
            return _priv[str(tid)](x)
        h = x
        for layer in self._trunk:
            h = layer(h)
        return self.adapter_seq(tid)(h)

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def trunk_parameters(self):
        """Parameters of the shared trunk (empty in single-head mode)."""
        if self._trunk is None:
            return []
        return [p for layer in self._trunk for p in layer.parameters()]

    def get_action(self, x, deterministic=False, tid=0):
        mean = self.head_mean(x, tid)
        if deterministic:
            return mean
        std = torch.exp(self.actor_logstd.expand_as(mean))
        return Normal(mean, std).sample()

    def get_action_and_value(self, x, action=None, tid=0):
        mean = self.head_mean(x, tid)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x).squeeze(-1)

# ============================================================
# Running Mean/Std for reward normalization
# ============================================================

class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    """One ring buffer PER TASK; sampling draws uniformly across past tasks.

    Each task generates ~2M transitions over training, far larger than the
    per-task capacity, so every buffer is a FIFO ring. Keeping buffers
    separate guarantees earlier tasks are never displaced by later ones (a
    single global FIFO would only retain the tail of the latest task and
    leave old tasks unprotected).
    """
    def __init__(self, per_task_capacity, obs_dim, action_dim):
        self.per_task = per_task_capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buf = {}   # task_id -> dict of ring arrays + pos/size

    def _init_task(self, task_id):
        c = self.per_task
        self.buf[task_id] = {
            "obs": np.zeros((c, self.obs_dim), dtype=np.float32),
            "actions": np.zeros((c, self.action_dim), dtype=np.float32),
            "rewards": np.zeros(c, dtype=np.float32),
            "next_obs": np.zeros((c, self.obs_dim), dtype=np.float32),
            "dones": np.zeros(c, dtype=np.float32),
            "pos": 0,
            "size": 0,
        }

    def add_batch(self, obs, actions, rewards, next_obs, dones, task_id):
        if task_id not in self.buf:
            self._init_task(task_id)
        b = self.buf[task_id]
        n = len(obs)
        c = self.per_task
        if n >= c:   # batch bigger than whole buffer: keep only tail
            obs = obs[-c:]; actions = actions[-c:]; rewards = rewards[-c:]
            next_obs = next_obs[-c:]; dones = dones[-c:]
            b["obs"] = obs; b["actions"] = actions; b["rewards"] = rewards
            b["next_obs"] = next_obs; b["dones"] = dones
            b["pos"] = 0; b["size"] = c
            return
        start = b["pos"]
        end = start + n
        if end <= c:
            b["obs"][start:end] = obs
            b["actions"][start:end] = actions
            b["rewards"][start:end] = rewards
            b["next_obs"][start:end] = next_obs
            b["dones"][start:end] = dones
        else:
            first = c - start
            b["obs"][start:] = obs[:first]
            b["actions"][start:] = actions[:first]
            b["rewards"][start:] = rewards[:first]
            b["next_obs"][start:] = next_obs[:first]
            b["dones"][start:] = dones[:first]
            second = n - first
            b["obs"][:second] = obs[first:]
            b["actions"][:second] = actions[first:]
            b["rewards"][:second] = rewards[first:]
            b["next_obs"][:second] = next_obs[first:]
            b["dones"][:second] = dones[first:]
            end = second
        b["pos"] = end
        b["size"] = min(b["size"] + n, c)

    def sample(self, n, exclude_task=None):
        """Sample n transitions uniformly across PAST tasks. exclude_task (the
        currently training task) is skipped: ER regularizes on old tasks only.
        """
        task_ids = [t for t in sorted(self.buf.keys())
                    if self.buf[t]["size"] > 0 and t != exclude_task]
        if not task_ids:
            return None
        per = n // len(task_ids)
        rem = n - per * len(task_ids)
        out = {k: [] for k in ("obs", "actions", "rewards", "next_obs", "dones", "task_ids")}
        for j, tid in enumerate(task_ids):
            b = self.buf[tid]
            m = per + (1 if j < rem else 0)
            idx = np.random.randint(0, b["size"], size=m)
            out["obs"].append(b["obs"][idx])
            out["actions"].append(b["actions"][idx])
            out["rewards"].append(b["rewards"][idx])
            out["next_obs"].append(b["next_obs"][idx])
            out["dones"].append(b["dones"][idx])
            out["task_ids"].append(np.full(m, tid, dtype=np.int32))
        return {k: np.concatenate(v, axis=0) for k, v in out.items()}

    def num_tasks(self):
        return sum(1 for b in self.buf.values() if b["size"] > 0)

    def __len__(self):
        return sum(b["size"] for b in self.buf.values())

# ============================================================
# Environment creation (unified 42-dim obs)
# ============================================================

def make_env(env_id, num_envs, seed=0, eval_mode=False, backend="cpu", device="cpu"):
    """Create vectorized ManiSkill envs wrapped to the unified 42-dim obs.

    backend="gpu" -> ONE env holding `num_envs` parallel sims on the GPU
                     (official setup: a step is one batched simulator call)
    backend="cpu" -> `num_envs` independent single-env CPU sims stepped in a
                     Python loop (the original port, kept as a fallback for
                     machines without CUDA)
    """
    task_info = TASK_INFO[env_id]
    cfg = SLOT_LAYOUT[env_id]
    max_eps = task_info.get("max_episode_steps")

    if backend == "gpu":
        return GpuVecEnv(
            env_id, num_envs=num_envs, cfg=cfg,
            control_mode=task_info["control_mode"], device=device,
            seed=seed, max_episode_steps=max_eps,
        )

    def _make():
        kwargs = dict(
            num_envs=1,
            obs_mode="state_dict",
            control_mode=task_info["control_mode"],
            sim_backend="cpu",
            # 见 vec_env.GpuVecEnv：状态观测不需要渲染
            render_backend="none",
        )
        if max_eps is not None:
            kwargs["max_episode_steps"] = int(max_eps)
        return gym.make(env_id, **kwargs)

    envs = [_make() for _ in range(num_envs)]
    return MultiEnvWrapper(envs, env_id, cfg, seed)


class MultiEnvWrapper:
    """Multi-env wrapper: maps every task's raw obs to the unified 42-dim vector."""
    def __init__(self, envs, env_id, cfg, seed=0):
        self.envs = envs
        self.env_id = env_id
        self.cfg = cfg
        self.obs_dim = UNIFIED_OBS_DIM
        self.num_envs = len(envs)
        # action_space is batched (num_envs, action_dim); take last dim
        self.action_dim = envs[0].action_space.shape[-1]

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

        np.random.seed(seed)

    def _obs(self, raw_obs):
        return build_unified_obs(raw_obs, self.cfg)

    def reset(self, seed=None):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(self._obs(obs))
        return np.stack(obs_list), {}

    def step(self, actions):
        obs_list, rew_list, done_list, truncated_list, info_list = [], [], [], [], []
        final_obs_list = [None] * len(self.envs)
        # The trainer may hand us a device tensor when the previous task ran on
        # the GPU; the CPU backend needs a host ndarray.
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        for i, env in enumerate(self.envs):
            obs, rew, done, truncated, info = env.step(actions[i][None, :])  # (8,) -> (1,8)
            final_obs_list[i] = self._obs(obs)
            rew_list.append(float(rew))
            done_list.append(bool(done))
            truncated_list.append(bool(truncated))
            info_list.append(info)
            if done or truncated:
                obs2, _ = env.reset()
                obs = obs2
            obs_list.append(self._obs(obs))
        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=np.bool_),
            np.array(truncated_list, dtype=np.bool_),
            info_list,
            np.stack(final_obs_list),
        )

    def close(self):
        for env in self.envs:
            env.close()


# ============================================================
# Evaluation
# ============================================================

def evaluate_task(agent, env_id, num_eps=20, max_steps=None, backend="cpu", device="cpu",
                  tid=0):
    """Evaluate agent on a single task with the unified 42-dim obs.
    Uses the task's NATIVE episode horizon (identical to the single-task
    baselines); max_steps is only a safety cap on the policy loop, so it is
    derived from the horizon when not given (a hard 200 would cut every
    StackPyramid episode, horizon 250, short).

    backend="gpu" delegates to vec_env.evaluate_vec, which runs all `num_eps`
    episodes as ONE batched simulation: 20 x 100 sequential GPU steps would
    otherwise dominate evaluation time (each tiny Sim.step is latency-bound).
    """
    task_info = TASK_INFO[env_id]
    cfg = SLOT_LAYOUT[env_id]
    max_eps = task_info.get("max_episode_steps")
    if max_steps is None:
        max_steps = (int(max_eps) if max_eps is not None else 200) + 50

    if backend == "gpu":
        return evaluate_vec(agent, env_id, cfg, task_info["control_mode"],
                            device=device, num_eps=num_eps,
                            max_episode_steps=max_eps, tid=tid)

    kwargs = dict(
        num_envs=1, obs_mode="state_dict",
        control_mode=task_info["control_mode"],
        sim_backend="cpu",
        # 见 vec_env.GpuVecEnv：状态观测不需要渲染
        render_backend="none",
        # Full scene reconfiguration on every reset, identical to the
        # single-task official baseline eval protocol (eval_reconfiguration_
        # freq=1), so CL eval numbers are comparable to the baselines.
        reconfiguration_freq=1,
    )
    if max_eps is not None:
        kwargs["max_episode_steps"] = int(max_eps)
    env = gym.make(env_id, **kwargs)

    successes = []
    total_rewards = []

    with torch.no_grad():
        for _ in range(num_eps):
            obs_raw, _ = env.reset()
            obs = build_unified_obs(obs_raw, cfg)
            ep_reward = 0.0
            done = False
            truncated = False
            steps = 0
            ep_success = False
            while not (done or truncated) and steps < max_steps:
                obs_t = as_tensor(obs, device).unsqueeze(0)
                action = agent.get_action(obs_t, deterministic=True,
                                          tid=tid).cpu().numpy()[0]
                action = np.clip(action, -1.0, 1.0)
                obs_raw, rew, done, truncated, info = env.step(action[None, :])  # (8,) -> (1,8)
                obs = build_unified_obs(obs_raw, cfg)
                ep_reward += float(rew)
                steps += 1

                sinfo = None
                if info is not None:
                    sinfo = info.get("success", info.get("is_success", None))
                if sinfo is not None:
                    try:
                        if bool(torch.as_tensor(sinfo).any()):
                            ep_success = True
                    except Exception:
                        try:
                            if bool(np.asarray(sinfo).any()):
                                ep_success = True
                        except Exception:
                            pass

            successes.append(1.0 if ep_success else 0.0)
            total_rewards.append(ep_reward)

    env.close()
    sr = np.mean(successes) * 100
    avg_r = np.mean(total_rewards)
    return sr, avg_r


def evaluate_all_tasks(agent, task_sequence, num_eps=20, backend="cpu", device="cpu"):
    results = {}
    for tid, env_id in enumerate(task_sequence):
        sr, avg_r = evaluate_task(agent, env_id, num_eps,
                                  backend=backend, device=device, tid=tid)
        results[env_id] = {"success_rate": sr, "avg_reward": avg_r}
    return results


# ============================================================
# PPO Training with Experience Replay
# ============================================================

def count_rollout_ends(dones, truncateds, infos):
    """Count finished episodes and successful ones inside one rollout step.

    Handles both env interfaces: the CPU wrapper returns a list of per-env
    info dicts, the GPU wrapper returns ONE dict of batched tensors.
    """
    if isinstance(infos, dict):
        done = as_tensor(dones, "cpu").reshape(-1).bool()
        trunc = as_tensor(truncateds, "cpu").reshape(-1).bool()
        ended = done | trunc
        n_done = int(ended.sum().item())
        succ = infos.get("success", None)
        if succ is not None and n_done:
            s = as_tensor(succ, "cpu").reshape(-1).bool() & ended
            return int(s.sum().item()), n_done
        return 0, n_done

    n_succ, n_done = 0, 0
    for i, inf in enumerate(infos):
        # Count each finished episode once (success flag persists until reset,
        # so only check at episode end).
        if dones[i] or truncateds[i]:
            n_done += 1
            if inf is not None and inf.get("success") is not None:
                try:
                    if bool(torch.as_tensor(inf["success"]).any()):
                        n_succ += 1
                except Exception:
                    pass
    return n_succ, n_done


def train_ppo_er(task_sequence, total_timesteps_per_task=3_000_000, er_config=None,
                 ppo_config=None, eval_freq=50, eval_eps=20, seed=42, verbose=True,
                 backend="cpu", device=None, run_tag=None, pretrain_ckpts=None,
                 pretrain_freeze_task=False, expert_anchors=None, anchor_coef=2.0,
                 stop_sr=0.0, mtd_coef=0.0, init_ckpt="", n_heads=1,
                 trunk_lr_mult=1.0, private_base_tasks=()):
    er_config = er_config or ER_CONFIG
    ppo_config = ppo_config or PPO_CONFIG
    n_tasks = len(task_sequence)
    # Indices (in task_sequence order) of tasks that get a private base network
    # instead of the shared one. Indices, not names, because the checkpoint
    # layout -- and therefore the evaluation script -- is positional.
    _private_ids = tuple(
        i for i, t in enumerate(task_sequence)
        if t in set(private_base_tasks or ()))
    obs_dim = UNIFIED_OBS_DIM          # 48, identical for every task
    action_dim = 8                     # pd_joint_delta_pos: 7 joints + 1 gripper

    # Suffix for result/checkpoint filenames. Empty -> the original names, so
    # the archived 12M CPU run stays reproducible byte-for-byte.
    _tag = f"_{run_tag}" if run_tag else ""

    # -- device / backend resolution -------------------------------------
    # "gpu" = CUDA-backed SAPIEN + all tensors on the GPU (official setup);
    # "cpu" = the original ported path (16 separate CPU sims in a Python loop).
    # Fail loudly instead of silently falling back: SAPIEN's GPU backend is
    # CUDA-only, so on an AMD/ROCm box the physics silently stays on the CPU
    # and the run looks mysteriously slow rather than broken.
    if device is None:
        device = "cuda" if (backend == "gpu" and torch.cuda.is_available()) else "cpu"
    device_obj = torch.device(device)
    if backend == "gpu" and device_obj.type != "cuda":
        raise RuntimeError(
            "backend='gpu' requires CUDA: SAPIEN has no ROCm backend, so the "
            "simulator cannot run on this GPU. Use --sim-backend cpu instead.")

    # Per-task step budgets: a scalar gives every task the same budget; a
    # list/tuple gives one budget per task (harder tasks need more -- e.g.
    # PickCube needs ~5.12M steps single-task vs ~2M for Push/Pull/LiftPeg).
    if isinstance(total_timesteps_per_task, (list, tuple)):
        assert len(total_timesteps_per_task) == n_tasks, (
            f"got {len(total_timesteps_per_task)} step budgets for {n_tasks} tasks")
        steps_per_task = [int(x) for x in total_timesteps_per_task]
    else:
        steps_per_task = [int(total_timesteps_per_task)] * n_tasks

    torch.manual_seed(seed)
    np.random.seed(seed)

    replay_buffer = ReplayBuffer(er_config["buffer_per_task"], obs_dim, action_dim)

    all_results = {}
    best_avg_sr = {}
    eval_log = []
    # task_idx -> frozen expert actor for that task (the --anchor-task table).
    # Kept for the whole run so old tasks can be distilled from on the replay
    # batch (see the MTD block in the update loop).
    teacher_nets = {}

    for task_idx, env_id in enumerate(task_sequence):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Task {task_idx+1}/{n_tasks}: {env_id}")
            print(f"  unified_obs_dim={obs_dim} (no one-hot, no padding)")
            print(f"  replay_buffer size: {len(replay_buffer)}")
            print(f"{'='*70}")

        num_envs = ppo_config["num_envs"]
        num_steps = ppo_config["num_steps"]
        batch_size = num_envs * num_steps
        num_minibatches = ppo_config["num_minibatches"]
        minibatch_size = batch_size // num_minibatches
        update_epochs = ppo_config["update_epochs"]

        envs = make_env(env_id, num_envs, seed=seed+task_idx,
                        backend=backend, device=device)

        # The action head is shared across all tasks, so every task must expose
        # the same action width. Catch a control_mode mismatch here instead of
        # silently writing wrongly-shaped actions into the replay buffer.
        env_action_dim = int(getattr(envs, "action_dim", action_dim))
        assert env_action_dim == action_dim, (
            f"{env_id}: env action dim {env_action_dim} != shared head {action_dim}; "
            f"check control_mode in TASK_INFO")

        if task_idx == 0:
            agent = Agent(obs_dim, action_dim, n_heads=n_heads,
                          private_base_ids=_private_ids).to(device_obj)
            # Continuation of a previous CL run (the "consolidation" pass).
            # A FULL state dict, unlike the policy-only warm start below: the
            # consolidation pass continues the same CL trajectory, so the
            # critic and the shared trunk must come along unchanged.
            if init_ckpt:
                agent.load_state_dict(torch.load(init_ckpt, map_location=device_obj),
                                      strict=True)
                if verbose:
                    print(f"  [init] agent <- {init_ckpt} (full state dict)")
        # else: reuse agent from previous task (this is the point of CL!)
        elif next(agent.parameters()).device != device_obj:
            agent = agent.to(device_obj)

        # Optional per-task warm start, applied when the sequence REACHES that
        # task. All earlier tasks still train from scratch exactly as in the
        # baseline, so the only difference from a vanilla ER run is the
        # initialisation of the named task.
        #
        # POLICY-ONLY: only actor_mean is copied. The critic stays as inherited
        # from the previous task, which is the whole point of CL (a shared
        # value function carried across tasks). Loading the whole state dict
        # also dragged in a value head trained on a different reward scale.
        #
        # The exploration scale is re-tightened because the seeded actor is a
        # near-deterministic expert: keeping the inherited std of exp(-0.5)=0.6
        # made the first update exceed the KL budget (approx_kl=0.134) and
        # destroyed the policy within one iteration.
        _warm = bool(pretrain_ckpts and env_id in pretrain_ckpts)
        if _warm:
            _path = pretrain_ckpts[env_id]
            _sd = torch.load(_path, map_location=device_obj)
            _actor = {k[len("actor_mean."):]: v for k, v in _sd.items()
                      if k.startswith("actor_mean.")}
            assert _actor, f"no actor_mean.* weights in {_path}"
            if agent._trunk is not None and task_idx in agent.private_base_ids:
                # Outlier task: the whole expert IS this task's network, so it
                # loads verbatim -- base and adapter both, no sharing involved.
                agent.private_nets[str(task_idx)].load_state_dict(
                    _actor, strict=True)
            elif agent._trunk is not None and task_idx > 0:
                # Multi-head: the expert's base is shared (that is the point),
                # but its last hidden layer and output layer belong to THIS
                # task's adapter. Writing them into actor_mean would overwrite
                # task 0's adapter.
                _nb = Agent.SHARED_BASE_LAYERS
                _body = {k: v for k, v in _actor.items()
                         if int(k.split(".")[0]) < _nb}
                # teacher adapter ("4.*", "6.*") -> local adapter ("0.*", "2.*")
                _adapt = {str(int(k.split(".")[0]) - _nb) + k[k.index("."):]: v
                          for k, v in _actor.items()
                          if int(k.split(".")[0]) >= _nb}
                assert _adapt, f"no adapter weights in {_path}"
                agent.actor_mean.load_state_dict(_body, strict=False)
                agent.actor_heads_extra[task_idx - 1].load_state_dict(
                    _adapt, strict=True)
            else:
                agent.actor_mean.load_state_dict(_actor, strict=True)
            with torch.no_grad():
                agent.actor_logstd.fill_(PRETRAIN_LOGSTD_FROZEN if pretrain_freeze_task else PRETRAIN_LOGSTD)
            # The value HEAD is task-specific: mean episodic reward differs by
            # ~10x across these tasks (Push ~4, LiftPeg ~15, StackCube ~9 over a
            # much longer horizon). An inherited head therefore predicts values
            # off by an order of magnitude, the advantages it produces have the
            # wrong scale and sign, and the first few updates wipe out the
            # seeded expert even though the actor update is clipped. Re-init
            # only the output layer (values ~0 -> advantage ~ return); the
            # shared trunk, which is the transferable part, is kept.
            layer_init(agent.critic[6], std=1.0)
            if verbose:
                print(f"  [warm start] {env_id}: actor_mean <- {_path}")
                print(f"  [warm start] critic.6 re-initialised, trunk kept, "
                      f"actor_logstd reset to {PRETRAIN_LOGSTD}")
                print(f"  [warm start] actor frozen for {PRETRAIN_FREEZE_ITERS} "
                      f"iterations, then lr={PRETRAIN_LR} target_kl={PRETRAIN_TARGET_KL}")

        # Frozen teacher for a task whose policy comes from a separate
        # pretraining stage. The actor is SHARED by every task, so copying an
        # expert into it (hard warm start) erases the earlier policies
        # instantly -- measured on the 5-task sequence: StackCube 95% while
        # Push/Pull/Pick/LiftPeg all fell to 0%. Distillation imports the same
        # knowledge without evicting the others: the shared trunk is pulled
        # towards the expert on THIS task's observations while ER keeps it on
        # the earlier tasks' observations.
        anchor_net = None
        if expert_anchors and env_id in expert_anchors:
            _apath = expert_anchors[env_id]
            _ateacher = Agent(obs_dim, action_dim).to(device_obj)
            _asd = torch.load(_apath, map_location=device_obj)
            _missing, _unexpected = _ateacher.load_state_dict(_asd, strict=False)
            assert not [k for k in _missing if k.startswith("actor_mean.")], _missing
            for _p in _ateacher.parameters():
                _p.requires_grad_(False)
            _ateacher.eval()
            anchor_net = _ateacher.actor_mean
            teacher_nets[task_idx] = anchor_net
            if verbose:
                print(f"  [anchor] {env_id}: distilling from {_apath} "
                      f"(coef={anchor_coef})")

        # A finished outlier task keeps its own network, so nothing should touch
        # it again: MTD still distils every old task, and on a private net that
        # distillation alone moved PegInsertionSide 85% -> 62% over the other
        # five tasks. Freezing it keeps the finished policy bit-exact.
        for _ptid in sorted(agent.private_base_ids):
            if _ptid < task_idx:
                for _p in agent.private_nets[str(_ptid)].parameters():
                    _p.requires_grad_(False)

        task_lr = PRETRAIN_LR if _warm else ppo_config["learning_rate"]
        task_target_kl = PRETRAIN_TARGET_KL if _warm else ppo_config.get("target_kl")
        # Multi-head: the heads are free to move at full speed, but the shared
        # trunk is what a finished task's frozen head reads from, so its drift
        # is the remaining leak (measured: with heads frozen per task, Peg still
        # went 95% -> 20% in 164k steps purely from trunk updates). Give the
        # trunk its own, much smaller step size instead of freezing it outright:
        # freezing would block all representation transfer between tasks.
        if agent._trunk is not None and trunk_lr_mult != 1.0:
            _trunk_ids = {id(p) for p in agent.trunk_parameters()}
            _trunk = [p for p in agent.parameters() if id(p) in _trunk_ids]
            _rest = [p for p in agent.parameters() if id(p) not in _trunk_ids]
            optimizer = optim.Adam([
                {"params": _trunk, "lr": task_lr * trunk_lr_mult},
                {"params": _rest, "lr": task_lr},
            ], eps=1e-5)
        else:
            optimizer = optim.Adam(agent.parameters(), lr=task_lr, eps=1e-5)
        critic_warmup_iters = 5 if task_idx > 0 else 0
        if _warm:
            critic_warmup_iters = max(critic_warmup_iters, PRETRAIN_FREEZE_ITERS)

        # Rollout tensors stay on the training device: on the GPU path the env
        # already hands back GPU tensors, so staging them on the host would add
        # two PCIe round-trips per environment step.
        obs_storage = torch.zeros((num_steps, num_envs, obs_dim), device=device_obj)
        act_storage = torch.zeros((num_steps, num_envs, action_dim), device=device_obj)
        rew_storage = torch.zeros((num_steps, num_envs), device=device_obj)
        done_storage = torch.zeros((num_steps, num_envs), device=device_obj)
        val_storage = torch.zeros((num_steps, num_envs), device=device_obj)
        logp_storage = torch.zeros((num_steps, num_envs), device=device_obj)
        final_obs_storage = torch.zeros((num_steps, num_envs, obs_dim), device=device_obj)

        task_total_steps = steps_per_task[task_idx]
        num_iterations = task_total_steps // batch_size
        if _warm and pretrain_freeze_task:
            # Full-task freeze: the policy for this task is supplied by the
            # standalone pretraining stage, so the CL run only adapts the value
            # network. Co-training the actor was measured to be self-defeating:
            # the ER behaviour-cloning term on earlier tasks is saturated
            # against a freshly seeded actor, so the shared trunk drifts to a
            # compromise that solves neither the new task nor the old ones
            # (StackCube 97% -> 0%, PickCube 95% -> 0%, LiftPeg 95% -> 10%).
            # Freezing via requires_grad keeps the ordinary training path, so
            # buffer bookkeeping and end-of-phase evaluation still run.
            for _p in agent.actor_mean.parameters():
                _p.requires_grad_(False)
            agent.actor_logstd.requires_grad_(False)
        global_step = 0
        start_time = time.time()
        if verbose:
            print(f"  step budget: {task_total_steps:,} ({num_iterations} iterations)")

        obs, _ = envs.reset()
        obs_tensor = as_tensor(obs, device_obj)
        bc_log = []
        anchor_log = []
        mtd_log = []
        # Consecutive evaluations at/above stop_sr. Two in a row (not one) so a
        # lucky 20-episode sample cannot end a task that is still improving.
        # Budgets are competitive with the 8M-step cap, so a task that is
        # already solved should hand the GPU to the next one.
        _sr_hits = 0
        # Best SR seen for the current task, used to keep its best weights (see
        # the eval block below).
        _best_env_sr = 0.0

        for iteration in range(1, num_iterations + 1):

            if ppo_config["anneal_lr"]:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lr = frac * task_lr
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            rollout_success = 0
            rollout_episodes = 0
            for step in range(num_steps):
                global_step += num_envs
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(
                        obs_tensor, tid=task_idx)

                next_obs, rewards, dones, truncateds, infos, final_obs = envs.step(action)
                next_obs_tensor = as_tensor(next_obs, device_obj)
                final_obs_tensor = as_tensor(final_obs, device_obj)

                n_succ, n_done = count_rollout_ends(dones, truncateds, infos)
                rollout_success += n_succ
                rollout_episodes += n_done

                obs_storage[step] = obs_tensor
                act_storage[step] = action
                rew_storage[step] = as_tensor(rewards, device_obj).reshape(-1)
                done_storage[step] = as_tensor(dones | truncateds, device_obj).reshape(-1)
                val_storage[step] = value.squeeze(-1)
                logp_storage[step] = logprob
                final_obs_storage[step] = final_obs_tensor
                obs_tensor = next_obs_tensor

            with torch.no_grad():
                # Bootstrap values for truncation/termination events INSIDE the
                # rollout using the final observation of the episode (as in the
                # official ManiSkill PPO: final_values[t, done envs]). Without
                # this, every mid-rollout episode end loses its terminal value
                # estimate (episodes ~100 steps, rollout 512 steps -> ~5 ends
                # per env per rollout), which badly biases long-horizon actions.
                next_value = agent.get_value(obs_tensor).reshape(1, -1)
                final_values = torch.zeros((num_steps, num_envs), device=device_obj)
                for t in range(num_steps):
                    done_mask = done_storage[t].bool()
                    if done_mask.any():
                        final_values[t, done_mask] = agent.get_value(
                            final_obs_storage[t, done_mask]).view(-1)

                advantages = torch.zeros_like(rew_storage)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    # done_storage[t] = "episode ended AFTER step t" (this
                    # script stores the post-step flag, unlike the official
                    # CleanRL convention which stores the pre-step flag).
                    # The GAE bootstrap mask at t must use the POST-step flag:
                    # if the episode ended at t, bootstrap from final_values[t]
                    # (terminal obs) and cut the lambda chain -- do NOT read
                    # done_storage[t+1], which belongs to the next episode and
                    # leaks its bootstrap across the episode boundary.
                    if t == num_steps - 1:
                        next_not_done = 1.0 - done_storage[t]
                        nextvalues = next_value
                    else:
                        next_not_done = 1.0 - done_storage[t]
                        nextvalues = val_storage[t + 1]
                    real_next_values = next_not_done * nextvalues + final_values[t]
                    delta = rew_storage[t] + ppo_config["gamma"] * real_next_values - val_storage[t]
                    advantages[t] = lastgaelam = delta + ppo_config["gamma"] * ppo_config["gae_lambda"] * next_not_done * lastgaelam
                returns = advantages + val_storage

            b_obs = obs_storage.reshape(-1, obs_dim)
            b_actions = act_storage.reshape(-1, action_dim)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = val_storage.reshape(-1)
            b_logprobs = logp_storage.reshape(-1)

            replay_obs_all = None
            replay_actions_all = None
            replay_tids_all = None
            if task_idx > 0 and er_config["replay_ratio"] > 0:
                # ER regularizes on PAST tasks only: exclude the current task's
                # buffer (self-replay would need on-policy corrections).
                n_replay = int(batch_size * er_config["replay_ratio"])
                replay = replay_buffer.sample(n_replay, exclude_task=task_idx)
                if replay is not None:
                    replay_obs_all = torch.as_tensor(
                        replay["obs"], dtype=torch.float32, device=device_obj)
                    replay_actions_all = torch.as_tensor(
                        replay["actions"], dtype=torch.float32, device=device_obj)
                    # Which task each replay row came from -- needed to pick the
                    # matching teacher in the MTD term below.
                    replay_tids_all = torch.as_tensor(
                        replay["task_ids"], dtype=torch.long, device=device_obj)

            if iteration <= critic_warmup_iters:
                if verbose:
                    print(f"  [WARMUP] critic-only iteration {iteration}/{critic_warmup_iters}")
                for _ in range(update_epochs):
                    b_inds = torch.randperm(batch_size, device=device_obj)
                    for start in range(0, batch_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        mb_obs_w = b_obs[mb_inds]
                        mb_returns_w = b_returns[mb_inds]
                        mb_old_values_w = b_values[mb_inds]

                        new_values_w = agent.get_value(mb_obs_w)
                        if ppo_config["clip_vloss"]:
                            v_loss_unclipped = (new_values_w - mb_returns_w) ** 2
                            v_clipped = mb_old_values_w + torch.clamp(
                                new_values_w - mb_old_values_w, -ppo_config["clip_coef"], ppo_config["clip_coef"])
                            v_loss_clipped = (v_clipped - mb_returns_w) ** 2
                            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        else:
                            v_loss = 0.5 * ((new_values_w - mb_returns_w) ** 2).mean()

                        optimizer.zero_grad()
                        v_loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), ppo_config["max_grad_norm"])
                        optimizer.step()
                if iteration % 10 == 0 or iteration == critic_warmup_iters:
                    if verbose:
                        elapsed = time.time() - start_time
                        sps = int(global_step / elapsed)
                        print(f"  [{iteration}/{num_iterations}] step={global_step:,} "
                              f"reward={rew_storage.mean().item():.2f} buffer={len(replay_buffer):,} "
                              f"sps={sps} elapsed={elapsed/60:.1f}min")
                continue

            clipfracs = []
            for epoch in range(update_epochs):
                b_inds = torch.randperm(batch_size, device=device_obj)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds], tid=task_idx)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > ppo_config["clip_coef"]).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if ppo_config["norm_adv"]:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - ppo_config["clip_coef"], 1 + ppo_config["clip_coef"])
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    if anchor_net is not None:
                        # Distillation replaces the policy gradient here.
                        pg_loss = pg_loss * 0.0

                    newvalue = newvalue.view(-1)
                    if ppo_config["clip_vloss"]:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -ppo_config["clip_coef"], ppo_config["clip_coef"],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ppo_config["ent_coef"] * entropy_loss + v_loss * ppo_config["vf_coef"]

                    if replay_obs_all is not None:
                        mb_size = mb_inds.shape[0]
                        rp_idx = torch.randint(0, replay_obs_all.shape[0], (mb_size,),
                                               device=device_obj)
                        _, rp_logprob, _, _ = agent.get_action_and_value(
                            replay_obs_all[rp_idx], replay_actions_all[rp_idx],
                            tid=replay_tids_all[rp_idx]
                            if replay_tids_all is not None else 0)
                        bc_loss = -torch.clamp(rp_logprob, min=-BC_LOGPROB_CLAMP).mean()
                        loss = loss + ppo_config.get("bc_coef", 0.05) * bc_loss
                        bc_log.append(float(bc_loss))

                    if anchor_net is not None:
                        with torch.no_grad():
                            teacher_mu = anchor_net(b_obs[mb_inds])
                        anchor_loss = torch.nn.functional.mse_loss(
                            agent.head_mean(b_obs[mb_inds], task_idx), teacher_mu)
                        loss = loss + anchor_coef * anchor_loss
                        anchor_log.append(float(anchor_loss))

                    # Multi-teacher distillation (MTD). The anchor term above
                    # only keeps the CURRENT task's expert; these expert
                    # policies are otherwise forgotten, and replay+BC on stale
                    # actions is a weak substitute (measured: StackCube 87% ->
                    # 25% over a 5-task ER run). Here every replay row is
                    # distilled from the expert of the task it came from, which
                    # is the same supervision the anchor term uses, applied to
                    # the old tasks' own observations.
                    if mtd_coef > 0 and replay_tids_all is not None:
                        mb_size = mb_inds.shape[0]
                        rp_idx = torch.randint(0, replay_obs_all.shape[0],
                                               (mb_size,), device=device_obj)
                        rp_tids = replay_tids_all[rp_idx]
                        mtd_loss = torch.zeros((), device=device_obj)
                        for tid in torch.unique(rp_tids).tolist():
                            net = teacher_nets.get(int(tid))
                            if net is None:
                                continue
                            sel = rp_tids == tid
                            obs_sel = replay_obs_all[rp_idx[sel]]
                            with torch.no_grad():
                                t_mu = net(obs_sel)
                            mtd_loss = mtd_loss + torch.nn.functional.mse_loss(
                                agent.head_mean(obs_sel, int(tid)), t_mu) * (
                                    obs_sel.shape[0] / mb_size)
                        loss = loss + mtd_coef * mtd_loss
                        mtd_log.append(float(mtd_loss))

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), ppo_config["max_grad_norm"])
                    optimizer.step()

                # target-KL early stop (official PPO): halt further epochs once
                # the policy moved too far from the rollout distribution.
                kl_gate = task_target_kl
                if anchor_net is not None and kl_gate is not None:
                    # Distillation must move the policy far from the rollout
                    # distribution; the PPO gate would cut every epoch short.
                    kl_gate = max(kl_gate, 0.5)
                if kl_gate is not None and approx_kl > kl_gate:
                    if verbose and iteration % 10 == 0:
                        print(f"  [KL early-stop] approx_kl={approx_kl.item():.4f} at epoch {epoch+1}")
                    break

                if verbose and iteration % 10 == 0:
                    print(f"  [INFO] epoch={epoch+1}/{update_epochs} KL={approx_kl:.4f}")

            # The replay buffer lives in host (numpy) memory on purpose: it holds
            # every past task's transitions for the whole run, so keeping it on
            # the GPU would blow up VRAM once a few tasks are stored.
            original_obs = obs_storage.reshape(-1, obs_dim).detach().cpu().numpy()
            original_actions = act_storage.reshape(-1, action_dim).detach().cpu().numpy()
            original_rewards = rew_storage.reshape(-1).detach().cpu().numpy()
            original_dones = done_storage.reshape(-1).detach().cpu().numpy()
            original_next_obs = final_obs_storage.reshape(-1, obs_dim).detach().cpu().numpy()

            replay_buffer.add_batch(
                original_obs, original_actions, original_rewards,
                original_next_obs, original_dones, task_id=task_idx,
            )

            if verbose and (iteration % 10 == 0 or iteration == 1):
                elapsed = time.time() - start_time
                sps = int(global_step / elapsed)
                avg_reward = rew_storage.mean().item()
                train_sr = 100.0 * rollout_success / max(1, rollout_episodes)
                bc_str = f"bc={sum(bc_log)/len(bc_log):.1f} " if bc_log else ""
                if anchor_log:
                    bc_str += f"anchor={sum(anchor_log)/len(anchor_log):.4f} "
                if mtd_log:
                    bc_str += f"mtd={sum(mtd_log)/len(mtd_log):.4f} "
                print(f"  [{iteration}/{num_iterations}] step={global_step:,} "
                      f"reward={avg_reward:.3f} train_SR={train_sr:.0f}% "
                      f"{bc_str}buffer={len(replay_buffer):,} "
                      f"sps={sps} elapsed={elapsed/60:.1f}min")

            if iteration % eval_freq == 0 or iteration == num_iterations:
                if verbose:
                    print(f"\n  -- Eval @ step {global_step:,}/{task_total_steps} --")

                eval_results = {}
                for eval_tid, eval_env_id in enumerate(task_sequence[:task_idx+1]):
                    sr, avg_r = evaluate_task(agent, eval_env_id, num_eps=eval_eps,
                                              backend=backend, device=device,
                                              tid=eval_tid)
                    eval_results[eval_env_id] = {"success_rate": sr, "avg_reward": avg_r}
                    if verbose:
                        print(f"    {eval_env_id}: SR={sr:.1f}% avg_R={avg_r:.2f}")

                avg_sr = np.mean([eval_results[t]["success_rate"] for t in eval_results])
                eval_log.append({
                    "global_step": global_step,
                    "task": env_id,
                    "task_idx": task_idx,
                    "eval_results": eval_results,
                    "avg_success_rate": avg_sr,
                })

                if verbose:
                    print(f"    avg SR: {avg_sr:.1f}%")

                # Keep the weights of the best evaluation of THIS task. The last
                # weights are not always the best ones (a 20-episode estimate is
                # noisy and late training can regress), and the CL stage downstream
                # wants the strongest possible teacher for each task.
                cur_sr = eval_results[env_id]["success_rate"]
                if cur_sr > _best_env_sr:
                    _best_env_sr = cur_sr
                    torch.save(agent.state_dict(), os.path.join(
                        CKPT_DIR,
                        f"ppo_er_unified_task{task_idx}_{env_id}{_tag}_best.pt"))
                    if verbose:
                        print(f"  [best] {env_id} SR={cur_sr:.1f}% -> best ckpt kept")

                # -- stop_sr: hand the GPU to the next task once this one is
                # solved twice in a row (see _sr_hits). Only the CURRENT task
                # is checked: an old task's SR is not a reason to stop.
                if stop_sr:
                    if eval_results[env_id]["success_rate"] >= stop_sr:
                        _sr_hits += 1
                    else:
                        _sr_hits = 0
                    if _sr_hits >= 2:
                        if verbose:
                            print(f"  [early stop] {env_id} SR="
                                  f"{eval_results[env_id]['success_rate']:.1f}% "
                                  f">= {stop_sr:.0f}% twice at step "
                                  f"{global_step:,}")
                        break

        envs.close()

        final_results = evaluate_all_tasks(agent, task_sequence[:task_idx+1],
                                           num_eps=eval_eps,
                                           backend=backend, device=device)
        task_key = f"task_{task_idx}_{env_id}"
        all_results[task_key] = {
            "final_eval": final_results,
            "eval_log": eval_log,
        }
        best_avg_sr[task_idx] = max(e["avg_success_rate"] for e in eval_log) if eval_log else 0.0

        if verbose:
            print(f"\n  === Task {task_idx+1} ({env_id}) complete ===")
            for t, r in final_results.items():
                print(f"    {t}: SR={r['success_rate']:.1f}% avg_R={r['avg_reward']:.2f}")

        # Tag the artefacts so a new run (e.g. the 5-task GPU run) never
        # overwrites the 12M CPU evidence that is already submitted.
        ckpt_path = os.path.join(
            CKPT_DIR, f"ppo_er_unified_task{task_idx}_{env_id}{_tag}.pt")
        torch.save(agent.state_dict(), ckpt_path)
        if verbose:
            print(f"  Checkpoint saved: {ckpt_path}")

    results_path = os.path.join(
        RESULT_DIR, f"ppo_er_unified_results{_tag}.json" if run_tag
        else "ppo_er_unified_results.json")
    output = {
        "obs_interface": "unified_42_semantic_slots",
        "sim_backend": backend,
        "device": str(device_obj),
        "run_tag": run_tag,
        "task_sequence": task_sequence,
        "total_timesteps_per_task": steps_per_task,
        "er_config": er_config,
        "anchor_coef": anchor_coef,
        "mtd_coef": mtd_coef,
        "init_ckpt": init_ckpt,
        "ppo_config": {k: v for k, v in ppo_config.items() if isinstance(v, (int, float, bool, str))},
        "per_task_results": all_results,
        "best_avg_sr": {str(k): v for k, v in best_avg_sr.items()},
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    if verbose:
        print(f"\nResults saved: {results_path}")

    return output


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO + ER on unified 42-dim ManiSkill3 obs")
    parser.add_argument("--task-sequence", type=str,
                        default=",".join(DEFAULT_TASK_SEQUENCE),
                        help="Comma-separated task sequence")
    parser.add_argument("--total-steps", type=int, default=2_000_000,
                        help="Total PPO steps per task (same budget for every task)")
    parser.add_argument("--task-steps", type=str, default=None,
                        help="Per-task step budgets, comma-separated, e.g. "
                             "'2000000,2000000,6000000,2000000'. Overrides --total-steps.")
    parser.add_argument("--replay-ratio", type=float, default=0.5,
                        help="Fraction of replay data in each PPO update batch")
    parser.add_argument("--buffer-per-task", type=int, default=200_000,
                        help="Replay buffer capacity per task")
    parser.add_argument("--eval-freq", type=int, default=50,
                        help="Evaluate every N iterations")
    parser.add_argument("--eval-eps", type=int, default=20,
                        help="Evaluation episodes per task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seqft", action="store_true",
                        help="Run SeqFT baseline (no replay, replay_ratio=0)")
    parser.add_argument("--sim-backend", choices=["auto", "gpu", "cpu"], default="auto",
                        help="Physics backend. 'gpu' = one env holding N parallel CUDA "
                             "sims (official PPO setup); 'cpu' = N separate CPU sims "
                             "(the original port, ~30x slower). 'auto' picks gpu when "
                             "CUDA is available. NOTE: SAPIEN has no ROCm backend, so an "
                             "AMD GPU always falls back to cpu.")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override rollout envs (default: 512 on gpu, 16 on cpu)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Override rollout steps per env (default: 50 on gpu, 512 on cpu)")
    parser.add_argument("--num-minibatches", type=int, default=None,
                        help="Override minibatches per update. With --num-envs 128 "
                             "--num-steps 64 (batch 8,192) set this to 16 to keep the "
                             "archived baseline's optimizer schedule (gradient steps, "
                             "not just samples, drive how fast a task converges).")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override discount factor (default 0.8). Official PPO "
                             "raises it to 0.99 for long-horizon tasks such as "
                             "PegInsertionSide where 0.8 cannot assign credit.")
    parser.add_argument("--gae-lambda", type=float, default=None,
                        help="Override GAE lambda (default 0.9).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override the shared actor-critic learning rate "
                             "(default 3e-4, which is too aggressive for a "
                             "precision task sharing the trunk).")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="Override the target-KL early-stop threshold "
                             "(default 0.1). Lower keeps each update closer to "
                             "the rollout distribution, which is what protects "
                             "a brittle previously-learned task.")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Suffix for result/checkpoint filenames, e.g. 'gpu5task'. "
                             "Omit to reuse the original names (the 12M CPU evidence).")
    parser.add_argument("--pretrain-ckpt", type=str, default="",
                        help="Per-task warm start, TASK=PATH[,TASK=PATH...]. The "
                             "checkpoint is loaded when the sequence reaches that "
                             "task, e.g. StackCube-v1 from its official PPO ckpt.")
    parser.add_argument("--init-ckpt", type=str, default="",
                        help="Full agent state dict to CONTINUE a previous CL run "
                             "from (used by the consolidation pass). Unlike "
                             "--pretrain-ckpt this keeps the critic and starts the "
                             "first task of the sequence from those weights.")
    parser.add_argument("--anchor-task", type=str, default="",
                        help="Distil a frozen expert into the shared actor for "
                             "TASK=PATH[,TASK=PATH...] instead of copying its "
                             "weights in (a hard copy evicts the other tasks).")
    parser.add_argument("--anchor-coef", type=float, default=2.0,
                        help="Weight of the mean-matching distillation loss.")
    parser.add_argument("--mtd-coef", type=float, default=0.0,
                        help="Weight of the multi-teacher distillation term: every "
                             "ER replay sample is distilled from the expert of the "
                             "task it came from. 0 disables it (plain ER).")
    parser.add_argument("--multi-head", action="store_true",
                        help="Task-incremental heads: one shared trunk plus one final "
                             "actor layer per task, selected by task index at rollout, "
                             "evaluation and on every replay row. Makes overwriting a "
                             "finished task impossible, which is what a precision task "
                             "such as PegInsertionSide needs (single head: 95%% -> 0%% "
                             "in 200k steps, at any learning rate).")
    parser.add_argument("--trunk-lr-mult", type=float, default=1.0,
                        help="Multiplier on the shared trunk's learning rate "
                             "(multi-head only). 1.0 = same as the heads; 0.1 keeps "
                             "the representation nearly still so a frozen task head "
                             "keeps seeing the features it was learned on.")
    parser.add_argument("--private-base", type=str, default="",
                        help="Comma-separated outlier tasks (names from "
                             "--task-sequence) that get a private copy of the "
                             "whole actor instead of the shared base + adapter. "
                             "Measured conflict: PegInsertionSide and StackCube "
                             "need opposite base plasticity (Peg 60%% at "
                             "trunk-lr-mult 0.5 where Stack reaches 75%%, Stack "
                             "0%% at 0.1 where Peg holds 85-95%%).")
    parser.add_argument("--stop-sr", type=float, default=0.0,
                        help="Per-task early stop: after two consecutive evals at "
                             "or above this success rate (%%), move on to the next "
                             "task. 0 disables it (full budget always runs).")
    parser.add_argument("--pretrain-freeze-task", action="store_true",
                        help="For warm-started tasks, freeze the actor for the "
                             "whole phase and only fit the critic. Use when the "
                             "policy comes from a separate pretraining stage.")
    args = parser.parse_args()

    task_seq = [t.strip() for t in args.task_sequence.split(",")]

    for t in task_seq:
        if t not in TASK_INFO:
            print(f"ERROR: Unknown task '{t}'. Available: {list(TASK_INFO.keys())}")
            sys.exit(1)

    # -- backend / device ------------------------------------------------
    sim_backend = args.sim_backend
    if sim_backend == "auto":
        sim_backend = "gpu" if torch.cuda.is_available() else "cpu"
    if sim_backend == "gpu" and not torch.cuda.is_available():
        print("ERROR: --sim-backend gpu requires CUDA. SAPIEN cannot run physics on "
              "a non-NVIDIA GPU (no ROCm backend), so use --sim-backend cpu.")
        sys.exit(1)
    device = "cuda" if sim_backend == "gpu" else "cpu"

    ppo_cfg = dict(PPO_CONFIG_GPU if sim_backend == "gpu" else PPO_CONFIG_CPU)
    if args.num_envs is not None:
        ppo_cfg["num_envs"] = args.num_envs
    if args.num_steps is not None:
        ppo_cfg["num_steps"] = args.num_steps
    if args.num_minibatches is not None:
        ppo_cfg["num_minibatches"] = args.num_minibatches
    # Official PPO raises the discount up to 0.99 for tasks where the default 0.8
    # fails (long-horizon credit assignment, e.g. PegInsertionSide). Exposed here
    # so a per-task override does not require editing the config dict.
    if args.gamma is not None:
        ppo_cfg["gamma"] = args.gamma
    if args.gae_lambda is not None:
        ppo_cfg["gae_lambda"] = args.gae_lambda
    # Global optimiser overrides. Needed when a sequence carries a precision
    # task (PegInsertionSide): the stock 3e-4 / target_kl 0.1 moves the shared
    # actor far enough in a handful of updates to take insertion success from
    # 95% to 0% (measured), so the whole sequence has to run at the gentler
    # schedule that warm starts already use.
    if args.lr is not None:
        ppo_cfg["learning_rate"] = args.lr
    if args.target_kl is not None:
        ppo_cfg["target_kl"] = args.target_kl
    # Defaults already match: 16x512 -> 16 minibatches of 512 (CPU),
    # 512x50 -> 32 minibatches of 800 (GPU). Only recompute when the user
    # overrides the rollout shape, keeping minibatches at ~800 samples.
    # An explicit --num-minibatches always wins: re-deriving it here would
    # silently undo a deliberate choice to mirror the baseline schedule.
    if (args.num_envs is not None or args.num_steps is not None) \
            and args.num_minibatches is None:
        ppo_cfg["num_minibatches"] = max(
            1, (ppo_cfg["num_envs"] * ppo_cfg["num_steps"]) // 800)

    er_cfg = ER_CONFIG.copy()
    er_cfg["replay_ratio"] = 0.0 if args.seqft else args.replay_ratio
    er_cfg["buffer_per_task"] = args.buffer_per_task

    pretrain_ckpts = {}
    for item in args.pretrain_ckpt.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            print(f"ERROR: --pretrain-ckpt entry '{item}' is not TASK=PATH")
            sys.exit(1)
        k, _, v = item.partition("=")
        pretrain_ckpts[k.strip()] = v.strip()
    for k, v in pretrain_ckpts.items():
        if k not in task_seq:
            print(f"ERROR: --pretrain-ckpt task '{k}' is not in the task sequence")
            sys.exit(1)
        if not os.path.exists(v):
            print(f"ERROR: --pretrain-ckpt path does not exist: {v}")
            sys.exit(1)

    expert_anchors = {}
    for item in args.anchor_task.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            print(f"ERROR: --anchor-task entry '{item}' is not TASK=PATH")
            sys.exit(1)
        k, _, v = item.partition("=")
        expert_anchors[k.strip()] = v.strip()
    for k, v in expert_anchors.items():
        if k not in task_seq:
            print(f"ERROR: --anchor-task task '{k}' is not in the task sequence")
            sys.exit(1)
        if not os.path.exists(v):
            print(f"ERROR: --anchor-task path does not exist: {v}")
            sys.exit(1)

    print(f"{'='*70}")
    print(f"ManiSkill3 CL - PPO + {'SeqFT' if args.seqft else 'ER'}  (unified 48-dim obs)")
    print(f"{'='*70}")
    print(f"Task sequence: {task_seq}")
    print(f"Observation: {UNIFIED_OBS_DIM}-dim semantic slots (no mapping layer, no one-hot)")
    print(f"Sim backend: {sim_backend}  device={device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if sim_backend == "gpu" else ""))
    print(f"Rollout: {ppo_cfg['num_envs']} envs x {ppo_cfg['num_steps']} steps "
          f"= {ppo_cfg['num_envs']*ppo_cfg['num_steps']:,} per update, "
          f"{ppo_cfg['num_minibatches']} minibatches")
    if args.task_steps is not None:
        step_budgets = [int(x.strip()) for x in args.task_steps.split(",")]
        if len(step_budgets) != len(task_seq):
            print(f"ERROR: --task-steps has {len(step_budgets)} budgets for {len(task_seq)} tasks")
            sys.exit(1)
    else:
        step_budgets = args.total_steps
    print(f"Steps per task: {step_budgets}")
    print(f"Replay ratio: {er_cfg['replay_ratio']}")
    print(f"Replay buffer: {er_cfg['buffer_per_task']:,} transitions per task")
    print(f"Eval: every {args.eval_freq} iterations, {args.eval_eps} episodes")
    if args.stop_sr:
        print(f"Early stop: current task >= {args.stop_sr:.0f}% twice -> next task")
    print(f"{'='*70}")

    results = train_ppo_er(
        task_sequence=task_seq,
        total_timesteps_per_task=step_budgets,
        er_config=er_cfg,
        ppo_config=ppo_cfg,
        eval_freq=args.eval_freq,
        eval_eps=args.eval_eps,
        seed=args.seed,
        backend=sim_backend,
        device=device,
        run_tag=args.run_tag,
        pretrain_ckpts=pretrain_ckpts,
        pretrain_freeze_task=args.pretrain_freeze_task,
        expert_anchors=expert_anchors,
        anchor_coef=args.anchor_coef,
        mtd_coef=args.mtd_coef,
        init_ckpt=args.init_ckpt,
        stop_sr=args.stop_sr,
        n_heads=len(task_seq) if args.multi_head else 1,
        trunk_lr_mult=args.trunk_lr_mult,
        private_base_tasks=[t.strip() for t in args.private_base.split(",")
                            if t.strip()],
    )
