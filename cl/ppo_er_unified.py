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
}

# PPO hyperparameters -- aligned with official ManiSkill PPO baseline
# (examples/baselines/ppo/ppo.py): gamma=0.8 short-sighted discount for dense
# reward, zero entropy bonus, 4 epochs with target-KL early stop, lr 3e-4
# constant, no value clipping. num_envs/num_steps differ (16x512 vs 512x50)
# because we run on CPU, but batch/minibatch sizes stay comparable.
PPO_CONFIG = {
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

ER_CONFIG = {
    "buffer_per_task": 200_000,   # per-task ring-buffer capacity
    "replay_ratio": 0.5,
    "sample_strategy": "uniform_across_tasks",
}

DEFAULT_TASK_SEQUENCE = ["PushCube-v1", "PullCube-v1", "PickCube-v1", "LiftPegUpright-v1"]

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
    """3-layer Tanh MLP, shared actor-critic (separate heads)."""
    def __init__(self, obs_dim, action_dim, hidden=256):
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

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action(self, x, deterministic=False):
        mean = self.actor_mean(x)
        if deterministic:
            return mean
        std = torch.exp(self.actor_logstd.expand_as(mean))
        return Normal(mean, std).sample()

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
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

def make_env(env_id, num_envs, seed=0, eval_mode=False):
    """Create vectorized ManiSkill envs wrapped to the unified 42-dim obs."""
    task_info = TASK_INFO[env_id]
    cfg = SLOT_LAYOUT[env_id]

    def _make():
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="state_dict",
            control_mode=task_info["control_mode"],
            sim_backend="cpu",
        )
        return env

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
        actions = np.clip(actions, -1.0, 1.0)
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

def evaluate_task(agent, env_id, num_eps=20, max_steps=200):
    """Evaluate agent on a single task with the unified 42-dim obs.
    Uses the task's NATIVE episode horizon (identical to the single-task
    baselines); max_steps is only a safety cap on the policy loop."""
    task_info = TASK_INFO[env_id]
    cfg = SLOT_LAYOUT[env_id]
    env = gym.make(
        env_id, num_envs=1, obs_mode="state_dict",
        control_mode=task_info["control_mode"],
        sim_backend="cpu",
        # Full scene reconfiguration on every reset, identical to the
        # single-task official baseline eval protocol (eval_reconfiguration_
        # freq=1), so CL eval numbers are comparable to the baselines.
        reconfiguration_freq=1,
    )

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
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = agent.get_action(obs_t, deterministic=True).cpu().numpy()[0]
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


def evaluate_all_tasks(agent, task_sequence, num_eps=20):
    results = {}
    for env_id in task_sequence:
        sr, avg_r = evaluate_task(agent, env_id, num_eps)
        results[env_id] = {"success_rate": sr, "avg_reward": avg_r}
    return results


# ============================================================
# PPO Training with Experience Replay
# ============================================================

def train_ppo_er(task_sequence, total_timesteps_per_task=3_000_000, er_config=None,
                 ppo_config=None, eval_freq=50, eval_eps=20, seed=42, verbose=True):
    er_config = er_config or ER_CONFIG
    ppo_config = ppo_config or PPO_CONFIG
    n_tasks = len(task_sequence)
    obs_dim = UNIFIED_OBS_DIM          # 42, identical for every task
    action_dim = 8                     # pd_joint_delta_pos: 7 joints + 1 gripper

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

        envs = make_env(env_id, num_envs, seed=seed+task_idx)

        if task_idx == 0:
            agent = Agent(obs_dim, action_dim).to(torch.device("cpu"))
        # else: reuse agent from previous task (this is the point of CL!)

        optimizer = optim.Adam(agent.parameters(), lr=ppo_config["learning_rate"], eps=1e-5)
        critic_warmup_iters = 5 if task_idx > 0 else 0

        obs_storage = torch.zeros((num_steps, num_envs, obs_dim))
        act_storage = torch.zeros((num_steps, num_envs, action_dim))
        rew_storage = torch.zeros((num_steps, num_envs))
        done_storage = torch.zeros((num_steps, num_envs))
        val_storage = torch.zeros((num_steps, num_envs))
        logp_storage = torch.zeros((num_steps, num_envs))
        final_obs_storage = torch.zeros((num_steps, num_envs, obs_dim))

        task_total_steps = steps_per_task[task_idx]
        num_iterations = task_total_steps // batch_size
        global_step = 0
        start_time = time.time()
        if verbose:
            print(f"  step budget: {task_total_steps:,} ({num_iterations} iterations)")

        obs, _ = envs.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        for iteration in range(1, num_iterations + 1):

            if ppo_config["anneal_lr"]:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lr = frac * ppo_config["learning_rate"]
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            rollout_success = 0
            rollout_episodes = 0
            for step in range(num_steps):
                global_step += num_envs
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs_tensor)

                next_obs, rewards, dones, truncateds, infos, final_obs = envs.step(action.numpy())
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
                final_obs_tensor = torch.tensor(final_obs, dtype=torch.float32)

                for i, inf in enumerate(infos):
                    # Count each finished episode once (success flag persists
                    # until reset, so only check at episode end).
                    if dones[i] or truncateds[i]:
                        rollout_episodes += 1
                        if inf is not None and inf.get("success") is not None:
                            try:
                                if bool(torch.as_tensor(inf["success"]).any()):
                                    rollout_success += 1
                            except Exception:
                                pass

                obs_storage[step] = obs_tensor
                act_storage[step] = action
                rew_storage[step] = torch.tensor(rewards, dtype=torch.float32).squeeze(-1)
                done_storage[step] = torch.tensor(dones | truncateds, dtype=torch.float32)
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
                final_values = torch.zeros((num_steps, num_envs))
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
            if task_idx > 0 and er_config["replay_ratio"] > 0:
                # ER regularizes on PAST tasks only: exclude the current task's
                # buffer (self-replay would need on-policy corrections).
                n_replay = int(batch_size * er_config["replay_ratio"])
                replay = replay_buffer.sample(n_replay, exclude_task=task_idx)
                if replay is not None:
                    replay_obs_all = torch.tensor(replay["obs"], dtype=torch.float32)
                    replay_actions_all = torch.tensor(replay["actions"], dtype=torch.float32)

            if iteration <= critic_warmup_iters:
                if verbose:
                    print(f"  [WARMUP] critic-only iteration {iteration}/{critic_warmup_iters}")
                for _ in range(update_epochs):
                    b_inds = torch.randperm(batch_size)
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
                b_inds = torch.randperm(batch_size)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds])
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
                        rp_idx = torch.randint(0, replay_obs_all.shape[0], (mb_size,))
                        _, rp_logprob, _, _ = agent.get_action_and_value(
                            replay_obs_all[rp_idx], replay_actions_all[rp_idx])
                        bc_loss = -rp_logprob.mean()
                        loss = loss + ppo_config.get("bc_coef", 0.05) * bc_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), ppo_config["max_grad_norm"])
                    optimizer.step()

                # target-KL early stop (official PPO): halt further epochs once
                # the policy moved too far from the rollout distribution.
                if ppo_config.get("target_kl") is not None and approx_kl > ppo_config["target_kl"]:
                    if verbose and iteration % 10 == 0:
                        print(f"  [KL early-stop] approx_kl={approx_kl.item():.4f} at epoch {epoch+1}")
                    break

                if verbose and iteration % 10 == 0:
                    print(f"  [INFO] epoch={epoch+1}/{update_epochs} KL={approx_kl:.4f}")

            original_obs = obs_storage.reshape(-1, obs_dim).numpy()
            original_actions = act_storage.reshape(-1, action_dim).numpy()
            original_rewards = rew_storage.reshape(-1).numpy()
            original_dones = done_storage.reshape(-1).numpy()
            original_next_obs = final_obs_storage.reshape(-1, obs_dim).numpy()

            replay_buffer.add_batch(
                original_obs, original_actions, original_rewards,
                original_next_obs, original_dones, task_id=task_idx,
            )

            if verbose and (iteration % 10 == 0 or iteration == 1):
                elapsed = time.time() - start_time
                sps = int(global_step / elapsed)
                avg_reward = rew_storage.mean().item()
                train_sr = 100.0 * rollout_success / max(1, rollout_episodes)
                print(f"  [{iteration}/{num_iterations}] step={global_step:,} "
                      f"reward={avg_reward:.3f} train_SR={train_sr:.0f}% "
                      f"buffer={len(replay_buffer):,} "
                      f"sps={sps} elapsed={elapsed/60:.1f}min")

            if iteration % eval_freq == 0 or iteration == num_iterations:
                if verbose:
                    print(f"\n  -- Eval @ step {global_step:,}/{task_total_steps} --")

                eval_results = {}
                for eval_env_id in task_sequence[:task_idx+1]:
                    sr, avg_r = evaluate_task(agent, eval_env_id, num_eps=eval_eps)
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

        envs.close()

        final_results = evaluate_all_tasks(agent, task_sequence[:task_idx+1], num_eps=eval_eps)
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

        ckpt_path = os.path.join(CKPT_DIR, f"ppo_er_unified_task{task_idx}_{env_id}.pt")
        torch.save(agent.state_dict(), ckpt_path)
        if verbose:
            print(f"  Checkpoint saved: {ckpt_path}")

    results_path = os.path.join(RESULT_DIR, "ppo_er_unified_results.json")
    output = {
        "obs_interface": "unified_42_semantic_slots",
        "task_sequence": task_sequence,
        "total_timesteps_per_task": steps_per_task,
        "er_config": er_config,
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
    args = parser.parse_args()

    task_seq = [t.strip() for t in args.task_sequence.split(",")]

    for t in task_seq:
        if t not in TASK_INFO:
            print(f"ERROR: Unknown task '{t}'. Available: {list(TASK_INFO.keys())}")
            sys.exit(1)

    er_cfg = ER_CONFIG.copy()
    er_cfg["replay_ratio"] = 0.0 if args.seqft else args.replay_ratio
    er_cfg["buffer_per_task"] = args.buffer_per_task

    print(f"{'='*70}")
    print(f"ManiSkill3 CL - PPO + {'SeqFT' if args.seqft else 'ER'}  (unified 42-dim obs)")
    print(f"{'='*70}")
    print(f"Task sequence: {task_seq}")
    print(f"Observation: {UNIFIED_OBS_DIM}-dim semantic slots (no mapping layer, no one-hot)")
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
    print(f"{'='*70}")

    results = train_ppo_er(
        task_sequence=task_seq,
        total_timesteps_per_task=step_budgets,
        er_config=er_cfg,
        eval_freq=args.eval_freq,
        eval_eps=args.eval_eps,
        seed=args.seed,
    )
