"""
cl_comparison.py - Continual Learning comparison for ManiSkill BC experts
  with DAgger refinement per phase

Strategies:
  1. SeqFT  - Sequential fine-tuning (naive, catastrophic forgetting baseline)
  2. ER     - Experience Replay (buffer of previous task data)
  3. EWC    - Elastic Weight Consolidation (Fisher-based regularization)

Tasks: PushCube-v1, PickCube-v1, StackCube-v1
  - All use pd_ee_delta_pose, action_dim=7
  - obs_dim varies (35/42/48), zero-padded to max_obs_dim

DAgger: each CL phase includes DAgger refinement rounds after initial BC training.

Usage:
    cd E:\\munichi\\continual_learning_v2
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" cl_comparison.py
    & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" cl_comparison.py --eval-eps 20 --dagger-rounds 3
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium
import mani_skill.envs

from scripted_experts import PushCubeExpert, PickCubeExpert, StackCubeExpert
from train_bc_experts import flatten_obs
from stable_baselines3 import PPO as SB3_PPO

# ============================================================
# Config
# ============================================================

CL_TASKS = ["PushCube-v1", "PickCube-v1", "StackCube-v1"]

TASK_CFG = {
    "PushCube-v1": {
        "max_episode_steps": 200,
        "control_mode": "pd_ee_delta_pose",
        "expert_cls": PushCubeExpert,
        "n_demos": 200,
    },
    "PickCube-v1": {
        "max_episode_steps": 200,
        "control_mode": "pd_ee_delta_pose",
        "expert_cls": PickCubeExpert,
        "n_demos": 200,
    },
    "StackCube-v1": {
        "max_episode_steps": 300,
        "control_mode": "pd_ee_delta_pose",
        "expert_cls": StackCubeExpert,
        "n_demos": 200,
    },
}

# RL expert model paths (SB3 PPO)
# NOTE: Disabled - PPO obs_space (35) doesn't match VecNormalize stats dim (54).
# The stale vec_normalize.pkl was from an older training run with different obs config.
# Using scripted experts as fallback until PPO can be retrained with matching normalization.
RL_EXPERT_PATHS = {
    "PushCube-v1": "expert_models/PushCube-v1/PushCube-v1_sb3_ppo.zip",
    # "PickCube-v1": "expert_models/PickCube-v1/PickCube-v1_sb3_ppo.zip",
}

# VecNormalize stats paths (must match training config)
RL_VEC_NORMALIZE_PATHS = {
    "PushCube-v1": os.path.join(os.path.dirname(__file__), "expert_models", "PushCube-v1", "vec_normalize.pkl"),
    "PickCube-v1": os.path.join(os.path.dirname(__file__), "expert_models", "PickCube-v1", "vec_normalize.pkl"),
}

_rl_expert_cache = {}

def _load_rl_expert(task_id):
    """Load SB3 PPO model + VecNormalize as expert.
    Returns (model, vec_normalize) or (None, None) if unavailable.
    """
    if task_id not in RL_EXPERT_PATHS:
        return None, None
    if task_id in _rl_expert_cache:
        return _rl_expert_cache[task_id]
    model_path = RL_EXPERT_PATHS[task_id]
    if not os.path.exists(model_path):
        print(f"  WARNING: RL expert not found at {model_path}")
        return None, None
    print(f"  Loading RL expert for {task_id}: {model_path}")
    model = SB3_PPO.load(model_path)

    # Load VecNormalize stats only if obs dim matches the model's obs space
    vec_norm = None
    vn_path = RL_VEC_NORMALIZE_PATHS.get(task_id)
    if vn_path and os.path.exists(vn_path):
        with open(vn_path, 'rb') as f:
            vn_candidate = pickle.load(f)
        # Handle dict format (regen script) or VecNormalize object
        if isinstance(vn_candidate, dict):
            obs_rms = vn_candidate["obs_rms"]
            clip_obs = vn_candidate.get("clip_obs", 10.0)
        else:
            obs_rms = vn_candidate.obs_rms
            clip_obs = getattr(vn_candidate, 'clip_obs', 10.0)
        # Only use if the observation dimension matches (avoid stale stats)
        model_obs_dim = model.observation_space.shape[0]
        vn_obs_dim = obs_rms.mean.shape[0]
        if vn_obs_dim == model_obs_dim:
            vec_norm = {"obs_rms": obs_rms, "clip_obs": clip_obs}
            print(f"  Loaded VecNormalize stats from {vn_path} (dim={vn_obs_dim})")
        else:
            print(f"  WARNING: VecNormalize dim mismatch: stats={vn_obs_dim}, model={model_obs_dim}. Skipping normalization.")
    else:
        print(f"  WARNING: VecNormalize stats not found at {vn_path}")

    _rl_expert_cache[task_id] = (model, vec_norm)
    return model, vec_norm

MAX_OBS_DIM = 48
TASK_ID_DIM = 3
ACTION_DIM = 7
HIDDEN_DIM = 512
CL_EPOCHS = 200
CL_BATCH_SIZE = 256
CL_LR = 1e-3
ER_BUFFER_SIZE = 5000
EWC_LAMBDA = 100.0
DEMO_DIR = os.path.join(os.path.dirname(__file__), "cl_demos")
DAGGER_ROUNDS = 2
DAGGER_EPISODES = 50
DAGGER_EPOCHS = 50


# ============================================================
# Normalizer
# ============================================================

class ZScoreNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim, dtype=np.float32)
        self.std = np.ones(dim, dtype=np.float32)

    def fit(self, data):
        self.mean = np.mean(data, axis=0).astype(np.float32)
        self.std = np.std(data, axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-6)

    def inv(self, x):
        return x * (self.std + 1e-6) + self.mean


class MinMaxNorm:
    def __init__(self, dim):
        self.min = np.zeros(dim, dtype=np.float32)
        self.max = np.ones(dim, dtype=np.float32)
        self.scale = np.ones(dim, dtype=np.float32)

    def fit(self, data):
        self.min = np.min(data, axis=0).astype(np.float32)
        self.max = np.max(data, axis=0).astype(np.float32)
        r = self.max - self.min
        self.scale = np.where(r > 1e-6, 2.0 / (r + 1e-8), 1.0)

    def __call__(self, x):
        return (x - self.min) * self.scale - 1.0

    def inv(self, x):
        return (x + 1.0) / (self.scale + 1e-8) + self.min


# ============================================================
# Policy Network
# ============================================================

class CLPolicy(nn.Module):
    def __init__(self, input_dim=MAX_OBS_DIM + TASK_ID_DIM,
                 action_dim=ACTION_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Helpers
# ============================================================

def _get_success(info):
    """Extract success flag from info dict."""
    s = info.get("success", False)
    if hasattr(s, "item"):
        return bool(s.item())
    if isinstance(s, np.ndarray):
        return bool(s.flat[0])
    return bool(s)


def pad_obs(obs, target_dim=MAX_OBS_DIM):
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.shape[0] == 0:
        return obs
    if obs.shape[1] >= target_dim:
        return obs[:, :target_dim]
    return np.pad(obs, ((0, 0), (0, target_dim - obs.shape[1])))


# ============================================================
# Demo Collection (success-only)
# ============================================================

def collect_demos(task_id, n_demos, verbose=True):
    """Collect expert demos. ONLY keeps successful episodes."""
    cfg = TASK_CFG[task_id]
    # Try loading RL expert (SB3 PPO), fallback to scripted expert
    rl_expert, vec_norm = _load_rl_expert(task_id)
    use_scripted = rl_expert is None
    if use_scripted:
        expert = cfg["expert_cls"]()

    # Always use state_dict; RL expert will flatten + normalize internally
    obs_mode = "state_dict"
    env = gymnasium.make(
        task_id, num_envs=1, obs_mode=obs_mode,
        control_mode=cfg["control_mode"],
        max_episode_steps=cfg["max_episode_steps"],
    )

    obs_list, act_list = [], []
    successes = 0
    attempts = 0
    max_attempts = n_demos * 10

    while successes < n_demos and attempts < max_attempts:
        obs, info = env.reset()
        if use_scripted:
            expert.reset()
        ep_obs, ep_acts = [], []
        ep_ok = False
        attempts += 1

        for _ in range(cfg["max_episode_steps"]):
            if use_scripted:
                flat = flatten_obs(obs)
                action = expert.get_action(obs)
            else:
                flat = flatten_obs(obs)
                # Apply VecNormalize (PPO was trained with normalized obs)
                norm_obs = flat.copy()
                if vec_norm is not None:
                    rms = vec_norm["obs_rms"]
                    co = vec_norm["clip_obs"]
                    norm_obs = np.clip(
                        (flat - rms.mean) / np.sqrt(rms.var + rms.epsilon),
                        -co, co
                    )
                action, _ = rl_expert.predict(norm_obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).flatten()
            ep_obs.append(flat)
            ep_acts.append(action.copy())

            obs, rew, term, trunc, info = env.step(
                torch.tensor(action[None], dtype=torch.float32))
            if _get_success(info):
                ep_ok = True
                break
            if term or trunc:
                break

        if ep_ok:
            successes += 1
            obs_list.extend(ep_obs)
            act_list.extend(ep_acts)

        if verbose and (attempts % 100 == 0 or successes == n_demos):
            print(f"    {task_id}: {successes}/{n_demos} success, "
                  f"{attempts} attempts (SR={100*successes/attempts:.0f}%)")

    env.close()
    print(f"    {task_id}: collected {successes} successful episodes "
          f"({len(obs_list)} transitions) from {attempts} attempts")
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def get_or_collect_demos(task_id):
    """Load cached successful demos or collect new ones."""
    os.makedirs(DEMO_DIR, exist_ok=True)
    obs_path = os.path.join(DEMO_DIR, f"{task_id}_v4_obs.npy")
    act_path = os.path.join(DEMO_DIR, f"{task_id}_v4_act.npy")

    if os.path.exists(obs_path) and os.path.exists(act_path):
        obs = np.load(obs_path)
        act = np.load(act_path)
        if len(obs) > 0:
            print(f"  Loaded {len(obs)} successful demos for {task_id}")
            return obs, act
        else:
            print(f"  Cache for {task_id} is empty, re-collecting...")

    n = TASK_CFG[task_id]["n_demos"]
    print(f"  Collecting {n} successful demos for {task_id}...")
    obs, act = collect_demos(task_id, n)
    np.save(obs_path, obs)
    np.save(act_path, act)
    print(f"  Saved {len(obs)} successful demos for {task_id}")
    return obs, act


# ============================================================
# Policy Evaluation
# ============================================================

def eval_policy(model, obs_norm, act_norm, task_id, n_episodes=20):
    """Evaluate learned policy on a task. Returns success rate."""
    cfg = TASK_CFG[task_id]
    ti = CL_TASKS.index(task_id)
    max_steps = cfg["max_episode_steps"]

    env = gymnasium.make(
        task_id, num_envs=1, obs_mode="state_dict",
        control_mode=cfg["control_mode"],
        max_episode_steps=max_steps,
    )

    model.eval()
    successes = 0

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, info = env.reset()
            ep_ok = False

            for _ in range(max_steps):
                flat = flatten_obs(obs)
                padded = pad_obs(flat.reshape(1, -1))
                tid = np.zeros(TASK_ID_DIM, dtype=np.float32)
                tid[ti] = 1.0
                inp = np.concatenate([padded[0], tid])

                obs_n = obs_norm(inp)
                obs_t = torch.from_numpy(obs_n[None]).float()
                pred = model(obs_t).numpy()[0]
                action = act_norm.inv(pred)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

                obs, rew, term, trunc, info = env.step(
                    torch.tensor(action[None], dtype=torch.float32))
                if _get_success(info):
                    ep_ok = True
                    break
                if term or trunc:
                    break

            if ep_ok:
                successes += 1

    env.close()
    return successes / n_episodes


# ============================================================
# DAgger Data Collection
# ============================================================

def collect_dagger_data(model, obs_norm, act_norm, expert, env,
                        task_id, n_episodes, max_steps):
    """DAgger: run model, expert corrects failed episodes.

    For each episode:
    1. Run model. If success -> keep (obs, model_action) pairs.
    2. If failure -> new episode with expert collecting correction data.
    """
    ti = CL_TASKS.index(task_id)
    obs_list, act_list = [], []
    success_eps = 0
    failed_eps = 0

    # Try loading RL expert, fallback to scripted
    rl_expert, vec_norm = _load_rl_expert(task_id)
    use_scripted = rl_expert is None
    if use_scripted:
        expert = TASK_CFG[task_id]["expert_cls"]()

    for ep in range(n_episodes):
        obs, info = env.reset()
        if use_scripted:
            expert.reset()
        ep_obs, ep_acts = [], []
        ep_ok = False

        # --- Run model ---
        for _ in range(max_steps):
            flat = flatten_obs(obs)  # always state_dict mode
            padded = pad_obs(flat.reshape(1, -1))
            tid_vec = np.zeros(TASK_ID_DIM, dtype=np.float32)
            tid_vec[ti] = 1.0
            inp = np.concatenate([padded[0], tid_vec])

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_norm(inp)[None]).float()
                pred = model(obs_t).numpy()[0]
            model_action = act_norm.inv(pred)
            model_action = np.clip(model_action, -1.0, 1.0).astype(np.float32)

            ep_obs.append(flat)
            ep_acts.append(model_action.copy())

            obs, rew, term, trunc, info = env.step(
                torch.tensor(model_action[None], dtype=torch.float32))
            if _get_success(info):
                ep_ok = True
                break
            if term or trunc:
                break

        if ep_ok:
            success_eps += 1
            obs_list.extend(ep_obs)
            act_list.extend(ep_acts)
        else:
            failed_eps += 1
            # --- Expert correction episode ---
            obs2, _ = env.reset()
            if use_scripted:
                expert.reset()
            for _ in range(max_steps):
                if use_scripted:
                    flat2 = flatten_obs(obs2)
                    expert_action = expert.get_action(obs2)
                else:
                    flat2 = flatten_obs(obs2)
                    norm_obs2 = flat2.copy()
                    if vec_norm is not None:
                        rms = vec_norm["obs_rms"]
                        co = vec_norm["clip_obs"]
                        norm_obs2 = np.clip(
                            (flat2 - rms.mean) / np.sqrt(rms.var + rms.epsilon),
                            -co, co
                        )
                    expert_action, _ = rl_expert.predict(norm_obs2, deterministic=True)
                    expert_action = np.asarray(expert_action, dtype=np.float32).flatten()
                obs_list.append(flat2)
                act_list.append(expert_action.copy())

                obs2, rew2, term2, trunc2, info2 = env.step(
                    torch.tensor(expert_action[None], dtype=torch.float32))
                if _get_success(info2) or term2 or trunc2:
                    break

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.float32)
    print(f"      DAgger data: {success_eps} success eps, "
          f"{failed_eps} failed->corrected, "
          f"{len(obs_list)} transitions")
    return obs_arr, act_arr


# ============================================================
# Training
# ============================================================

def _do_training(model, all_obs, all_act, obs_norm, act_norm,
                 epochs, lr, fisher, prev_params, verbose, prefix=""):
    """Internal: train model on given data for given epochs."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-5)
    criterion = nn.MSELoss()

    obs_n = obs_norm(all_obs)
    act_n = act_norm(all_act)
    dataset = TensorDataset(
        torch.from_numpy(obs_n), torch.from_numpy(act_n))
    loader = DataLoader(dataset, batch_size=CL_BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_b = 0
        for bo, ba in loader:
            bo = bo + torch.randn_like(bo) * 0.01
            optimizer.zero_grad()
            pred = model(bo)
            loss = criterion(pred, ba)
            if fisher is not None and prev_params is not None:
                loss = loss + ewc_loss(model, fisher, prev_params)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        scheduler.step()
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"      {prefix}Ep {epoch:3d}: "
                  f"loss={total_loss/max(n_b,1):.6f}")


def train_model(train_obs, train_act, obs_norm, act_norm,
                epochs=CL_EPOCHS, lr=CL_LR,
                replay_obs=None, replay_act=None,
                fisher=None, prev_params=None,
                task_id_idx=None,
                dagger_rounds=0, dagger_episodes=50,
                dagger_epochs=DAGGER_EPOCHS,
                expert=None, env=None,
                verbose=True):
    """Train CLPolicy with optional ER/EWC and DAgger refinement."""
    model = CLPolicy()
    ti = task_id_idx if task_id_idx is not None else 0

    # Add task_id one-hot
    n_train = len(train_obs)
    tid = np.zeros((n_train, TASK_ID_DIM), dtype=np.float32)
    tid[:, ti] = 1.0
    cur_obs = np.concatenate([train_obs, tid], axis=1)
    cur_act = train_act.copy()

    # Add replay buffer data
    if replay_obs is not None and len(replay_obs) > 0:
        cur_obs = np.concatenate([cur_obs, replay_obs])
        cur_act = np.concatenate([cur_act, replay_act])

    # --- Initial BC training ---
    if verbose:
        print(f"      Initial BC ({epochs} ep, {len(cur_obs)} samples)...")
    _do_training(model, cur_obs, cur_act, obs_norm, act_norm,
                 epochs, lr, fisher, prev_params, verbose, prefix="")

    # --- DAgger refinement rounds ---
    if dagger_rounds > 0 and expert is not None and env is not None:
        max_steps = TASK_CFG[CL_TASKS[ti]]["max_episode_steps"]

        for dr in range(dagger_rounds):
            if verbose:
                print(f"      DAgger round {dr+1}/{dagger_rounds}...")

            # Collect correction data (env is reused, reset inside)
            new_obs, new_act = collect_dagger_data(
                model, obs_norm, act_norm, expert, env,
                CL_TASKS[ti], dagger_episodes, max_steps)

            if len(new_obs) == 0:
                if verbose:
                    print(f"      No new data collected, skipping")
                break

            # Pad new obs to MAX_OBS_DIM, then add task_id
            new_obs_padded = pad_obs(new_obs)  # (N, MAX_OBS_DIM)
            new_tid = np.zeros((len(new_obs), TASK_ID_DIM), dtype=np.float32)
            new_tid[:, ti] = 1.0
            new_obs_tid = np.concatenate([new_obs_padded, new_tid], axis=1)

            mix_obs = np.concatenate([cur_obs, new_obs_tid])
            mix_act = np.concatenate([cur_act, new_act])

            # Continue training with mixed data
            if verbose:
                print(f"      Retraining ({dagger_epochs} ep, "
                      f"{len(mix_obs)} samples)...")
            _do_training(model, mix_obs, mix_act, obs_norm, act_norm,
                         dagger_epochs, lr, fisher, prev_params,
                         verbose, prefix=f"D{dr+1} ")

            # Update cur_obs/cur_act for next DAgger round
            cur_obs = mix_obs
            cur_act = mix_act

    return model


def compute_fim(model, obs_t, act_t, n_samples=200):
    """Compute diagonal Fisher Information Matrix."""
    n = min(n_samples, len(obs_t))
    idx = np.random.choice(len(obs_t), n, replace=False)
    fim = {}
    for name, p in model.named_parameters():
        fim[name] = torch.zeros_like(p)

    model.eval()
    criterion = nn.MSELoss()
    for i in idx:
        model.zero_grad()
        out = model(obs_t[i:i+1])
        loss = criterion(out, act_t[i:i+1])
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fim[name] += p.grad.data.clone() ** 2 / n

    return fim


def ewc_loss(model, fisher, prev_params):
    """EWC regularization loss."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if name in fisher:
            loss += (fisher[name].to(p.device) *
                     (p - prev_params[name].to(p.device)) ** 2).sum()
    return EWC_LAMBDA * loss


def snapshot_params(model):
    return {n: p.data.clone() for n, p in model.named_parameters()}


# ============================================================
# CL Strategies
# ============================================================

def run_seqft(datasets, obs_norm, act_norm, eval_eps,
              epochs=CL_EPOCHS, dagger_rounds=0,
              dagger_episodes=50, dagger_epochs=DAGGER_EPOCHS):
    """Strategy 1: Sequential Fine-Tuning with DAgger."""
    print("\n" + "="*60)
    print(f"Strategy 1: SeqFT (DAgger rounds={dagger_rounds})")
    print("="*60)

    results = {}
    t0 = time.time()

    for phase, task_id in enumerate(CL_TASKS):
        print(f"\n  --- Phase {phase+1}: {task_id} ---")
        ti = CL_TASKS.index(task_id)
        cfg = TASK_CFG[task_id]
        train_o, train_a = datasets[task_id]

        # Create env for DAgger (reused across rounds)
        env = None
        expert = None
        if dagger_rounds > 0:
            rl_exp, _ = _load_rl_expert(task_id)
            obs_mode = "state_dict"  # always state_dict; RL expert handles flatten+normalize internally
            expert = cfg["expert_cls"]() if rl_exp is None else None
            env = gymnasium.make(
                task_id, num_envs=1, obs_mode=obs_mode,
                control_mode=cfg["control_mode"],
                max_episode_steps=cfg["max_episode_steps"],
            )

        model = train_model(
            train_o, train_a, obs_norm, act_norm,
            epochs=epochs, task_id_idx=ti, verbose=True,
            dagger_rounds=dagger_rounds, dagger_episodes=dagger_episodes,
            dagger_epochs=dagger_epochs, expert=expert, env=env,
        )

        if env is not None:
            env.close()

        for eval_task in CL_TASKS[:phase+1]:
            sr = eval_policy(model, obs_norm, act_norm, eval_task, eval_eps)
            results.setdefault(task_id, {})[eval_task] = sr
            print(f"    {eval_task}: {sr*100:.0f}%")

    mins = (time.time() - t0) / 60
    print(f"  SeqFT done in {mins:.1f} min")
    return results


def run_er(datasets, obs_norm, act_norm, eval_eps,
           epochs=CL_EPOCHS, dagger_rounds=0,
           dagger_episodes=50, dagger_epochs=DAGGER_EPOCHS):
    """Strategy 2: Experience Replay with DAgger."""
    print("\n" + "="*60)
    print(f"Strategy 2: ER (DAgger rounds={dagger_rounds})")
    print("="*60)

    results = {}
    buffer_obs, buffer_act = [], []
    t0 = time.time()

    for phase, task_id in enumerate(CL_TASKS):
        print(f"\n  --- Phase {phase+1}: {task_id} ---")
        ti = CL_TASKS.index(task_id)
        cfg = TASK_CFG[task_id]
        train_o, train_a = datasets[task_id]

        r_obs, r_act = None, None
        if len(buffer_obs) > 0:
            r_obs = np.concatenate(buffer_obs)
            r_act = np.concatenate(buffer_act)

        env = None
        expert = None
        if dagger_rounds > 0:
            rl_exp, _ = _load_rl_expert(task_id)
            obs_mode = "state_dict" if rl_exp is None else "state"
            expert = cfg["expert_cls"]() if rl_exp is None else None
            env = gymnasium.make(
                task_id, num_envs=1, obs_mode=obs_mode,
                control_mode=cfg["control_mode"],
                max_episode_steps=cfg["max_episode_steps"],
            )

        model = train_model(
            train_o, train_a, obs_norm, act_norm,
            epochs=epochs, replay_obs=r_obs, replay_act=r_act,
            task_id_idx=ti, verbose=True,
            dagger_rounds=dagger_rounds, dagger_episodes=dagger_episodes,
            dagger_epochs=dagger_epochs, expert=expert, env=env,
        )

        if env is not None:
            env.close()

        for eval_task in CL_TASKS[:phase+1]:
            sr = eval_policy(model, obs_norm, act_norm, eval_task, eval_eps)
            results.setdefault(task_id, {})[eval_task] = sr
            print(f"    {eval_task}: {sr*100:.0f}%")

        # Add to buffer with task_id baked in (51D)
        n = len(train_o)
        tid = np.zeros((n, TASK_ID_DIM), dtype=np.float32)
        tid[:, ti] = 1.0
        buffer_obs.append(np.concatenate([train_o, tid], axis=1))
        buffer_act.append(train_a)

        # Cap buffer
        all_o = np.concatenate(buffer_obs)
        all_a = np.concatenate(buffer_act)
        if len(all_o) > ER_BUFFER_SIZE:
            idx = np.random.choice(len(all_o), ER_BUFFER_SIZE, replace=False)
            buffer_obs = [all_o[idx]]
            buffer_act = [all_a[idx]]

    mins = (time.time() - t0) / 60
    print(f"  ER done in {mins:.1f} min")
    return results


def run_ewc(datasets, obs_norm, act_norm, eval_eps,
            epochs=CL_EPOCHS, dagger_rounds=0,
            dagger_episodes=50, dagger_epochs=DAGGER_EPOCHS):
    """Strategy 3: Elastic Weight Consolidation with DAgger."""
    print("\n" + "="*60)
    print(f"Strategy 3: EWC (lambda={EWC_LAMBDA}, DAgger rounds={dagger_rounds})")
    print("="*60)

    results = {}
    fisher = None
    prev_params = None
    model = None
    t0 = time.time()

    for phase, task_id in enumerate(CL_TASKS):
        print(f"\n  --- Phase {phase+1}: {task_id} ---")
        ti = CL_TASKS.index(task_id)
        cfg = TASK_CFG[task_id]
        train_o, train_a = datasets[task_id]

        env = None
        expert = None
        if dagger_rounds > 0:
            rl_exp, _ = _load_rl_expert(task_id)
            obs_mode = "state_dict" if rl_exp is None else "state"
            expert = cfg["expert_cls"]() if rl_exp is None else None
            env = gymnasium.make(
                task_id, num_envs=1, obs_mode=obs_mode,
                control_mode=cfg["control_mode"],
                max_episode_steps=cfg["max_episode_steps"],
            )

        model = train_model(
            train_o, train_a, obs_norm, act_norm,
            epochs=epochs, fisher=fisher, prev_params=prev_params,
            task_id_idx=ti, verbose=True,
            dagger_rounds=dagger_rounds, dagger_episodes=dagger_episodes,
            dagger_epochs=dagger_epochs, expert=expert, env=env,
        )

        if env is not None:
            env.close()

        for eval_task in CL_TASKS[:phase+1]:
            sr = eval_policy(model, obs_norm, act_norm, eval_task, eval_eps)
            results.setdefault(task_id, {})[eval_task] = sr
            print(f"    {eval_task}: {sr*100:.0f}%")

        # Compute FIM
        print(f"    Computing Fisher ({task_id})...")
        n = len(train_o)
        tid = np.zeros((n, TASK_ID_DIM), dtype=np.float32)
        tid[:, ti] = 1.0
        obs_with_tid = np.concatenate([train_o, tid], axis=1)
        obs_n = obs_norm(obs_with_tid)
        act_n = act_norm(train_a)
        fisher = compute_fim(
            model,
            torch.from_numpy(obs_n).float(),
            torch.from_numpy(act_n).float())
        prev_params = snapshot_params(model)
        print(f"    FIM computed.")

    mins = (time.time() - t0) / 60
    print(f"  EWC done in {mins:.1f} min")
    return results


# ============================================================
# Results Display
# ============================================================

def print_results(all_results, eval_eps):
    print("\n" + "="*70)
    print("CONTINUAL LEARNING COMPARISON RESULTS")
    print("="*70)

    header = f"{'Strategy':<12}"
    for t in CL_TASKS:
        header += f" {t:<20}"
    header += f" {'Average':<10} {'Forget':<10}"
    print(header)
    print("-"*70)

    for strategy in ["SeqFT", "ER", "EWC"]:
        r = all_results[strategy]
        final_task = CL_TASKS[-1]

        perfs = []
        for t in CL_TASKS:
            p = r.get(final_task, {}).get(t, 0)
            perfs.append(p)

        avg = np.mean(perfs)
        forgets = []
        for i, t in enumerate(CL_TASKS[:-1]):
            learned = r.get(t, {}).get(t, 0)
            final = r.get(final_task, {}).get(t, 0)
            forgets.append(max(0, learned - final))
        avg_forget = np.mean(forgets) if forgets else 0

        row = f"{strategy:<12}"
        for p in perfs:
            row += f" {p*100:6.1f}%{'':13}"
        row += f" {avg*100:6.1f}%   {avg_forget*100:5.1f}%"
        print(row)

    print("-"*70)

    print("\nDetailed per-phase results (task accuracy after each phase):")
    for strategy in ["SeqFT", "ER", "EWC"]:
        r = all_results[strategy]
        print(f"\n  {strategy}:")
        print(f"  {'Phase':<20}", end="")
        for t in CL_TASKS:
            print(f" {t:<16}", end="")
        print()

        for phase_task in CL_TASKS:
            print(f"  After {phase_task:<12}", end="")
            for eval_task in CL_TASKS:
                v = r.get(phase_task, {}).get(eval_task, None)
                if v is not None:
                    print(f" {v*100:6.1f}%{'':8}", end="")
                else:
                    print(f" {'--':<14}", end="")
            print()

    # Save results
    out = {
        "tasks": CL_TASKS,
        "eval_episodes": eval_eps,
        "cl_epochs": CL_EPOCHS,
        "hidden_dim": HIDDEN_DIM,
        "er_buffer_size": ER_BUFFER_SIZE,
        "ewc_lambda": EWC_LAMBDA,
        "dagger_rounds": DAGGER_ROUNDS,
        "dagger_episodes": DAGGER_EPISODES,
        "dagger_epochs": DAGGER_EPOCHS,
    }
    for strategy in ["SeqFT", "ER", "EWC"]:
        out[strategy] = {}
        for phase_task, evals in all_results[strategy].items():
            out[strategy][phase_task] = {
                t: float(v) for t, v in evals.items()}

    path = os.path.join(os.path.dirname(__file__), "cl_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-eps", type=int, default=20,
                        help="Evaluation episodes per task")
    parser.add_argument("--epochs", type=int, default=CL_EPOCHS,
                        help="Initial BC epochs per task")
    parser.add_argument("--dagger-rounds", type=int, default=DAGGER_ROUNDS,
                        help="DAgger refinement rounds per phase (0=disable)")
    parser.add_argument("--dagger-episodes", type=int, default=DAGGER_EPISODES,
                        help="Episodes per DAgger round")
    parser.add_argument("--dagger-epochs", type=int, default=DAGGER_EPOCHS,
                        help="Training epochs per DAgger round")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip demo collection, use cached files")
    args = parser.parse_args()

    print("="*70)
    print("Continual Learning Comparison: SeqFT vs ER vs EWC (+DAgger)")
    print("="*70)
    print(f"  Tasks: {CL_TASKS}")
    print(f"  Obs: zero-pad to {MAX_OBS_DIM} + task_id({TASK_ID_DIM}) = "
          f"{MAX_OBS_DIM + TASK_ID_DIM}")
    print(f"  Action: {ACTION_DIM}D, Hidden: {HIDDEN_DIM}")
    print(f"  Initial BC epochs: {args.epochs}")
    print(f"  DAgger: rounds={args.dagger_rounds}, "
          f"episodes={args.dagger_episodes}, "
          f"retrain_epochs={args.dagger_epochs}")
    print(f"  ER buffer: {ER_BUFFER_SIZE}")
    print(f"  EWC lambda: {EWC_LAMBDA}")
    print(f"  Eval episodes: {args.eval_eps}")
    print("="*70)

    # ---- Step 1: Collect / Load demos ----
    print("\n[Step 1] Demo collection...")
    datasets = {}
    all_padded_obs = []
    all_norm_act = []

    for task_id in CL_TASKS:
        obs, act = get_or_collect_demos(task_id)
        if len(obs) == 0:
            raise RuntimeError(f"No demos collected for {task_id}! Cannot continue.")
        datasets[task_id] = (obs, act)
        all_padded_obs.append(pad_obs(obs))
        all_norm_act.append(act)
        obs_2d = obs.reshape(len(obs), -1)
        print(f"  {task_id}: {len(obs)} demos, "
              f"obs_dim={obs_2d.shape[1]}, act_dim={act.shape[1]}")

    # ---- Step 2: Fit shared normalizers ----
    print("\n[Step 2] Fitting shared normalizers...")
    all_51d = []
    for i, task_id in enumerate(CL_TASKS):
        o = pad_obs(datasets[task_id][0])
        n = len(o)
        tid = np.zeros((n, TASK_ID_DIM), dtype=np.float32)
        tid[:, i] = 1.0
        all_51d.append(np.concatenate([o, tid], axis=1))
    combined_51d = np.concatenate(all_51d, axis=0)

    obs_norm = ZScoreNorm(MAX_OBS_DIM + TASK_ID_DIM)
    obs_norm.fit(combined_51d)

    combined_act = np.concatenate(all_norm_act)
    act_norm = MinMaxNorm(ACTION_DIM)
    act_norm.fit(combined_act)

    norm_datasets = {}
    for task_id in CL_TASKS:
        o, a = datasets[task_id]
        norm_datasets[task_id] = (pad_obs(o), a)
    print(f"  Done. Combined: {len(combined_51d)} transitions")

    # ---- Step 3: Run CL strategies ----
    print("\n[Step 3] Running CL strategies...")
    t0 = time.time()
    all_results = {}

    strategy_kwargs = dict(
        epochs=args.epochs,
        dagger_rounds=args.dagger_rounds,
        dagger_episodes=args.dagger_episodes,
        dagger_epochs=args.dagger_epochs,
    )

    all_results["SeqFT"] = run_seqft(
        norm_datasets, obs_norm, act_norm, args.eval_eps, **strategy_kwargs)
    all_results["ER"] = run_er(
        norm_datasets, obs_norm, act_norm, args.eval_eps, **strategy_kwargs)
    all_results["EWC"] = run_ewc(
        norm_datasets, obs_norm, act_norm, args.eval_eps, **strategy_kwargs)

    total = (time.time() - t0) / 60
    print(f"\nTotal time: {total:.1f} min")

    # ---- Step 4: Results ----
    print_results(all_results, args.eval_eps)


if __name__ == "__main__":
    main()

