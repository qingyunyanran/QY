"""
train_bc_experts.py - Train BC expert models (v4 - zero-variance action masking)

Key improvements over v3:
  1. Detect and mask zero-variance action dims (e.g. rotation dims 3-5)
  2. Only train on effective action dims → less gradient noise
  3. Longer training (300 epochs) + smaller batch (256) for better convergence
  4. Eval fills masked dims with constant value from demo data

Usage:
    cd E:\\munichi\\continual_learning_v2
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" train_bc_experts.py --task PickCube-v1
    & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" train_bc_experts.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import gymnasium
import mani_skill.envs

from scripted_experts import (
    PushCubeExpert, PickCubeExpert, StackCubeExpert, PegInsertionSideExpert,
    _extract_extra,
)
from peg_ik_expert import PegInsertionIKExpertForBC


# ============================================================
# Config (v2: more demos, bigger net, longer training)
# ============================================================

TASK_CONFIG = {
    "PushCube-v1": {
        "expert_cls": PushCubeExpert,
        "max_episode_steps": 200,
        "n_demos": 500,
        "target_success": 0.90,
        "bc_epochs": 150,
        "bc_batch_size": 512,
        "bc_lr": 1e-3,
        "bc_hidden": 512,
        "dagger_rounds": 0,
        "dagger_rollouts": 0,
        "control_mode": "pd_ee_delta_pose",
        "action_dim": 7,
    },
    "PickCube-v1": {
        "expert_cls": PickCubeExpert,
        "max_episode_steps": 200,
        "n_demos": 1000,
        "target_success": 0.90,
        "bc_epochs": 300,
        "bc_batch_size": 256,
        "bc_lr": 1e-3,
        "bc_hidden": 1024,
        "dagger_rounds": 3,
        "dagger_rollouts": 200,
        "control_mode": "pd_ee_delta_pose",
        "action_dim": 7,
    },
    "StackCube-v1": {
        "expert_cls": StackCubeExpert,
        "max_episode_steps": 300,
        "n_demos": 1000,
        "target_success": 0.70,
        "bc_epochs": 200,
        "bc_batch_size": 512,
        "bc_lr": 1e-3,
        "bc_hidden": 512,
        "dagger_rounds": 0,
        "dagger_rollouts": 0,
        "control_mode": "pd_ee_delta_pose",
        "action_dim": 7,
    },
    "PegInsertionSide-v1": {
        "expert_cls": PegInsertionIKExpertForBC,
        "max_episode_steps": 500,
        "n_demos": 500,
        "target_success": 0.50,
        "bc_epochs": 300,
        "bc_batch_size": 512,
        "bc_lr": 1e-3,
        "bc_hidden": 1024,
        "dagger_rounds": 0,
        "dagger_rollouts": 0,
        "control_mode": "pd_joint_pos",
        "action_dim": 8,
        "action_clip_low": -3.5,
        "action_clip_high": 3.5,
    },
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "expert_models")


# ============================================================
# Obs flattening: dict -> 1D vector
# ============================================================

def flatten_obs(obs_dict):
    """Flatten state_dict obs into 1D numpy array."""
    parts = []
    agent = obs_dict.get("agent", {})
    if "qpos" in agent:
        v = agent["qpos"]
        if hasattr(v, "cpu"):
            v = v.cpu().numpy()
        v = np.asarray(v, dtype=np.float32).flatten()
        parts.append(v)
    if "qvel" in agent:
        v = agent["qvel"]
        if hasattr(v, "cpu"):
            v = v.cpu().numpy()
        v = np.asarray(v, dtype=np.float32).flatten()
        parts.append(v)
    extra = obs_dict.get("extra", {})
    for key in sorted(extra.keys()):
        v = extra[key]
        if hasattr(v, "cpu"):
            v = v.cpu().numpy()
        v = np.asarray(v, dtype=np.float32).flatten()
        parts.append(v)
    return np.concatenate(parts)


def get_obs_dim(task_id, config=None):
    """Determine obs dim by creating a test env."""
    control_mode = config.get("control_mode", "pd_ee_delta_pose") if config else "pd_ee_delta_pose"
    env = gymnasium.make(
        task_id, num_envs=1, obs_mode="state_dict",
        control_mode=control_mode,
    )
    obs, _ = env.reset()
    flat = flatten_obs(obs)
    env.close()
    return flat.shape[0]


# ============================================================
# Normalization (v3: obs z-score + action min-max)
# ============================================================

class ZScoreNormalizer:
    """Z-score for observations: x_norm = (x - mean) / (std + eps)"""
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)
        self.eps = 1e-6

    def fit(self, data):
        self.mean = np.mean(data, axis=0).astype(np.float32)
        self.std = np.std(data, axis=0).astype(np.float32)
        self.std[self.std < self.eps] = 1.0

    def normalize(self, x):
        return (x - self.mean) / (self.std + self.eps)


class MinMaxNormalizer:
    """Min-max normalization for actions: maps [min, max] -> [-1, 1]
    
    This is critical for BC: z-score centers actions at 0 which causes
    the Tanh network to collapse to near-zero outputs (mode collapse).
    Min-max preserves the actual action distribution shape.
    """
    def __init__(self, shape):
        self.data_min = np.zeros(shape, dtype=np.float32)
        self.data_max = np.ones(shape, dtype=np.float32)
        self.scale = np.ones(shape, dtype=np.float32)

    def fit(self, data):
        self.data_min = np.min(data, axis=0).astype(np.float32)
        self.data_max = np.max(data, axis=0).astype(np.float32)
        data_range = self.data_max - self.data_min
        # For constant features (range=0), use scale=1 (output stays 0)
        self.scale = np.where(data_range > 1e-6, 2.0 / (data_range + 1e-8), 1.0)

    def normalize(self, x):
        # Map [min, max] -> [-1, 1]
        return (x - self.data_min) * self.scale - 1.0

    def denormalize(self, x_norm):
        # Map [-1, 1] -> [min, max]
        return (x_norm + 1.0) / (self.scale + 1e-8) + self.data_min


# ============================================================
# BC Network (v2: 3-layer 512-dim)
# ============================================================

class BCMLP(nn.Module):
    """3-layer MLP: normalized_obs -> action, Tanh output.
    
    No dropout - BC doesn't need regularization via dropout,
    and dropout at eval time adds unwanted stochasticity.
    """

    def __init__(self, obs_dim, action_dim=7, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Demo Collection
# ============================================================

def collect_demos(task_id, config, verbose=True):
    """Collect expert demonstrations. Returns obs_list, act_list, stats."""
    expert_cls = config["expert_cls"]
    max_steps = config["max_episode_steps"]
    n_demos = config["n_demos"]

    control_mode = config.get("control_mode", "pd_ee_delta_pose")

    env = gymnasium.make(
        task_id, num_envs=1, obs_mode="state_dict",
        control_mode=control_mode, render_mode="rgb_array",
        max_episode_steps=max_steps,
    )
    expert = expert_cls()
    if hasattr(expert, 'set_env'):
        expert.set_env(env.unwrapped)

    obs_list, act_list = [], []
    successes = 0
    total_steps = 0
    ep_lengths = []

    for ep in range(n_demos):
        obs, info = env.reset()
        expert.reset()
        ep_success = False
        ep_steps = 0

        for step_i in range(max_steps):
            flat_obs = flatten_obs(obs)
            if isinstance(obs, dict):
                action = expert.get_action(obs)
            else:
                action = np.zeros(config.get("action_dim", 7), dtype=np.float32)

            obs_list.append(flat_obs)
            act_list.append(action.copy())

            action_tensor = torch.tensor(action[np.newaxis, ...], dtype=torch.float32)
            obs, reward, terminated, truncated, info = env.step(action_tensor)
            ep_steps += 1

            success_val = info.get("success", False) if isinstance(info, dict) else False
            if hasattr(success_val, "cpu"):
                success_val = success_val.cpu().numpy()
            if isinstance(success_val, np.ndarray):
                success_val = success_val.item() if success_val.size == 1 else success_val.any()

            if success_val:
                ep_success = True
                break
            if terminated or truncated:
                break

        total_steps += ep_steps
        ep_lengths.append(ep_steps)
        if ep_success:
            successes += 1

        if verbose and (ep % 50 == 0 or ep == n_demos - 1):
            print(f"  Ep {ep+1}/{n_demos}: steps={ep_steps}, "
                  f"success={'YES' if ep_success else 'no'}, "
                  f"running SR={successes}/{ep+1} ({100*successes/(ep+1):.0f}%)")

    env.close()

    stats = {
        "n_episodes": n_demos,
        "n_success": successes,
        "success_rate": successes / n_demos,
        "total_transitions": len(obs_list),
        "mean_ep_length": np.mean(ep_lengths),
    }
    return obs_list, act_list, stats


# ============================================================
# BC Training (v2: with obs+action normalization)
# ============================================================

def train_bc(obs_list, act_list, obs_dim, config, verbose=True):
    """Train BC MLP with obs z-score + action min-max normalization.
    v4: Detect and mask zero-variance action dims, only train on effective dims.
    """
    epochs = config["bc_epochs"]
    batch_size = config["bc_batch_size"]
    lr = config["bc_lr"]
    hidden_dim = config["bc_hidden"]

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.float32)

    # ---- Detect zero-variance action dims ----
    act_std = np.std(act_arr, axis=0)
    act_mean_full = np.mean(act_arr, axis=0)
    active_mask = act_std > 1e-6  # True = this dim varies, needs learning
    active_indices = np.where(active_mask)[0]
    masked_indices = np.where(~active_mask)[0]
    n_active = len(active_indices)

    print(f"  Action std: {act_std.round(6)}")
    print(f"  Active dims (vary): {active_indices.tolist()} -> {n_active} dims to learn")
    print(f"  Masked dims (constant): {masked_indices.tolist()} -> values={act_mean_full[masked_indices].round(4)}")

    # ---- Obs: z-score normalization ----
    obs_norm = ZScoreNormalizer(obs_arr.shape[1])
    obs_norm.fit(obs_arr)
    obs_normalized = obs_norm.normalize(obs_arr)

    # ---- Action: min-max normalization ONLY on active dims ----
    act_active = act_arr[:, active_mask]  # shape: (N, n_active)
    act_norm = MinMaxNormalizer(act_active.shape[1])
    act_norm.fit(act_active)
    act_active_normalized = act_norm.normalize(act_active)
    act_active_normalized = np.clip(act_active_normalized, -1.0, 1.0)

    print(f"  Active action range: min={np.min(act_active, axis=0).round(4)}, "
          f"max={np.max(act_active, axis=0).round(4)}")
    print(f"  Normalized active action mean: {np.mean(act_active_normalized, axis=0).round(3)}")
    print(f"  Data: {obs_normalized.shape[0]} transitions, {obs_normalized.shape[1]} obs dim, "
          f"{n_active} active action dims")
    print(f"  Building DataLoader (batch_size={batch_size})...")

    obs_tensor = torch.from_numpy(obs_normalized)
    act_tensor = torch.from_numpy(act_active_normalized)

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"  DataLoader ready. {len(loader)} batches per epoch.")

    # Build model - output only active dims
    print(f"  Building BCMLP (obs_dim={obs_dim}, action_dim={n_active}, hidden={hidden_dim})...")
    model = BCMLP(obs_dim, action_dim=n_active, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    print(f"  Starting training for {epochs} epochs...")

    epoch_losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_obs, batch_act in loader:
            # Observation noise augmentation
            noise = torch.randn_like(batch_obs) * 0.01
            batch_obs = batch_obs + noise

            optimizer.zero_grad()
            pred = model(batch_obs)
            loss = criterion(pred, batch_act)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 30 == 0 or epoch == epochs - 1:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.6f}, lr={cur_lr:.2e}, best={best_loss:.6f}")
            import sys; sys.stdout.flush()

    print(f"  Final loss: {epoch_losses[-1]:.6f}, Best loss: {best_loss:.6f}")

    # Store active/masked info for eval
    model._active_indices = active_indices
    model._masked_indices = masked_indices
    model._active_mask = active_mask
    model._const_values = act_mean_full.copy()  # constant values for masked dims

    metrics = {
        "final_loss": epoch_losses[-1],
        "best_loss": best_loss,
        "epoch_losses": epoch_losses,
        "active_indices": active_indices.tolist(),
        "masked_indices": masked_indices.tolist(),
        "n_active": n_active,
    }
    return model, obs_norm, act_norm, metrics


# ============================================================
# Evaluation (v2: with normalization)
# ============================================================

def evaluate_bc(model, obs_norm, act_norm, task_id, config, n_episodes=20):
    """Evaluate BC model. v4: reconstruct full action from active dims + masked constants."""
    max_steps = config["max_episode_steps"]
    active_mask = model._active_mask
    active_indices = model._active_indices
    masked_indices = model._masked_indices
    const_values = model._const_values
    control_mode = config.get("control_mode", "pd_ee_delta_pose")
    clip_lo = config.get("action_clip_low", -1.0)
    clip_hi = config.get("action_clip_high", 1.0)

    env = gymnasium.make(
        task_id, num_envs=1, obs_mode="state_dict",
        control_mode=control_mode, render_mode="rgb_array",
        max_episode_steps=max_steps,
    )

    model.eval()
    successes = 0
    details = []

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, info = env.reset()
            ep_success = False
            ep_steps = 0
            ep_reward = 0.0

            for step_i in range(max_steps):
                flat_obs = flatten_obs(obs)

                # Normalize obs
                obs_normalized = obs_norm.normalize(flat_obs)
                obs_t = torch.from_numpy(obs_normalized[np.newaxis, :]).float()

                # Network output in [-1, 1] for active dims only
                pred_norm_active = model(obs_t).numpy()[0]

                # Denormalize active dims to real action space
                action_active = act_norm.denormalize(pred_norm_active)

                # Reconstruct full action
                action = const_values.copy()
                action[active_mask] = action_active
                action = np.clip(action, clip_lo, clip_hi).astype(np.float32)

                action_tensor = torch.tensor(action[np.newaxis, ...], dtype=torch.float32)
                obs, reward, terminated, truncated, info = env.step(action_tensor)
                ep_steps += 1
                ep_reward += reward.item() if hasattr(reward, 'item') else float(reward)

                success_val = info.get("success", False) if isinstance(info, dict) else False
                if hasattr(success_val, "cpu"):
                    success_val = success_val.cpu().numpy()
                if isinstance(success_val, np.ndarray):
                    success_val = success_val.item() if success_val.size == 1 else success_val.any()

                if success_val:
                    ep_success = True
                    break
                if terminated or truncated:
                    break

            if ep_success:
                successes += 1

            details.append({
                "ep": ep, "steps": ep_steps,
                "success": ep_success, "reward": ep_reward,
            })

            status = "SUCCESS" if ep_success else "FAIL"
            print(f"  Ep {ep}: {ep_steps} steps, reward={ep_reward:.2f}, {status}")

    env.close()
    sr = successes / n_episodes
    print(f"\n  BC Eval: {successes}/{n_episodes} ({100*sr:.0f}%)")
    return sr, details


# ============================================================
# DAgger: iterative rollout + expert relabeling
# ============================================================

def dagger_rollout_and_relabel(model, obs_norm, act_norm, expert, task_id, config,
                                n_episodes, verbose=True):
    """Rollout BC model to collect obs states, then relabel with scripted expert actions.
    
    This addresses compounding error: BC is trained on expert-distribution obs,
    but at eval time encounters obs from BC-distribution (which drifts over time).
    DAgger fixes this by training on obs from the BC model's actual distribution.
    
    Returns: new_obs_list, new_act_list (expert-labeled)
    """
    max_steps = config["max_episode_steps"]
    active_mask = model._active_mask
    const_values = model._const_values
    control_mode = config.get("control_mode", "pd_ee_delta_pose")
    clip_lo = config.get("action_clip_low", -1.0)
    clip_hi = config.get("action_clip_high", 1.0)

    env = gymnasium.make(
        task_id, num_envs=1, obs_mode="state_dict",
        control_mode=control_mode, render_mode="rgb_array",
        max_episode_steps=max_steps,
    )

    # Inject env for IK-based experts
    if hasattr(expert, 'set_env'):
        expert.set_env(env.unwrapped)

    model.eval()
    new_obs_list = []
    new_act_list = []
    total_steps = 0
    ep_successes = 0

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, info = env.reset()
            expert.reset()
            ep_steps = 0

            for step_i in range(max_steps):
                flat_obs = flatten_obs(obs)

                # BC action (for stepping the env)
                obs_normalized = obs_norm.normalize(flat_obs)
                obs_t = torch.from_numpy(obs_normalized[np.newaxis, :]).float()
                pred_norm_active = model(obs_t).numpy()[0]
                action_active = act_norm.denormalize(pred_norm_active)
                bc_action = const_values.copy()
                bc_action[active_mask] = action_active
                bc_action = np.clip(bc_action, clip_lo, clip_hi).astype(np.float32)

                # Expert action (for relabeling) - uses raw obs dict
                expert_action = expert.get_action(obs)

                # Store (obs, expert_action) for training
                new_obs_list.append(flat_obs)
                new_act_list.append(expert_action.copy())

                # Step with BC action
                action_tensor = torch.tensor(bc_action[np.newaxis, ...], dtype=torch.float32)
                obs, reward, terminated, truncated, info = env.step(action_tensor)
                ep_steps += 1
                total_steps += 1

                success_val = info.get("success", False) if isinstance(info, dict) else False
                if hasattr(success_val, "cpu"):
                    success_val = success_val.cpu().numpy()
                if isinstance(success_val, np.ndarray):
                    success_val = success_val.item() if success_val.size == 1 else success_val.any()

                if success_val:
                    ep_successes += 1
                    break
                if terminated or truncated:
                    break

            if verbose and (ep % 50 == 0 or ep == n_episodes - 1):
                print(f"    DAgger ep {ep+1}/{n_episodes}: BC steps={ep_steps}, "
                      f"running SR={ep_successes}/{ep+1} ({100*ep_successes/(ep+1):.0f}%)")

    env.close()
    bc_sr = ep_successes / n_episodes
    print(f"    DAgger rollout done: {len(new_obs_list)} transitions, "
          f"BC model SR={bc_sr*100:.0f}%, {n_episodes} episodes")
    return new_obs_list, new_act_list


def retrain_with_dagger(model, obs_norm, act_norm, initial_obs, initial_act,
                         dagger_obs, dagger_act, obs_dim, config, verbose=True):
    """Retrain BC model on combined initial + DAgger data.
    
    Re-detects active dims from the combined dataset (should be same as initial).
    Uses the same active_mask from the existing model for consistency.
    """
    epochs = config["bc_epochs"]
    batch_size = config["bc_batch_size"]
    lr = config["bc_lr"] * 0.5  # Lower LR for fine-tuning
    hidden_dim = config["bc_hidden"]

    # Combine data
    all_obs = np.array(initial_obs + dagger_obs, dtype=np.float32)
    all_act = np.array(initial_act + dagger_act, dtype=np.float32)

    print(f"    Combined data: {len(initial_obs)} initial + {len(dagger_obs)} DAgger = {len(all_obs)} total")

    # Use existing active mask (from initial training)
    active_mask = model._active_mask
    active_indices = model._active_indices
    n_active = len(active_indices)

    # Re-fit obs normalizer on all data
    obs_norm.fit(all_obs)
    obs_normalized = obs_norm.normalize(all_obs)

    # Normalize active action dims with existing act_norm
    act_active = all_act[:, active_mask]
    act_norm.fit(act_active)
    act_active_normalized = act_norm.normalize(act_active)
    act_active_normalized = np.clip(act_active_normalized, -1.0, 1.0)

    print(f"    Active action mean (normalized): {np.mean(act_active_normalized, axis=0).round(3)}")

    obs_tensor = torch.from_numpy(obs_normalized)
    act_tensor = torch.from_numpy(act_active_normalized)
    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Rebuild model with same architecture
    new_model = BCMLP(obs_dim, action_dim=n_active, hidden_dim=hidden_dim)
    optimizer = optim.Adam(new_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    print(f"    Retraining for {epochs} epochs on {len(loader)} batches/epoch...")

    epoch_losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        new_model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_obs, batch_act in loader:
            noise = torch.randn_like(batch_obs) * 0.01
            batch_obs = batch_obs + noise

            optimizer.zero_grad()
            pred = new_model(batch_obs)
            loss = criterion(pred, batch_act)
            loss.backward()
            nn.utils.clip_grad_norm_(new_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 50 == 0 or epoch == epochs - 1:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch}/{epochs}: loss={avg_loss:.6f}, lr={cur_lr:.2e}, best={best_loss:.6f}")
            import sys; sys.stdout.flush()

    print(f"    Final loss: {epoch_losses[-1]:.6f}, Best loss: {best_loss:.6f}")

    # Carry over active/masked info
    new_model._active_indices = active_indices
    new_model._masked_indices = model._masked_indices
    new_model._active_mask = active_mask
    new_model._const_values = model._const_values.copy()

    metrics = {
        "final_loss": epoch_losses[-1],
        "best_loss": best_loss,
        "epoch_losses": epoch_losses,
    }
    return new_model, obs_norm, act_norm, metrics


# ============================================================
# Save helpers
# ============================================================

def save_model(model, obs_norm, act_norm, task_id, config, metrics, eval_sr):
    """Save model + normalization stats + metadata (v4 with active/masked dims)."""
    task_dir = os.path.join(OUTPUT_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    model_path = os.path.join(task_dir, f"{task_id}_bc_expert.pt")
    torch.save(model.state_dict(), model_path)

    norm_path = os.path.join(task_dir, f"{task_id}_norm_stats.pt")
    torch.save({
        "obs_mean": obs_norm.mean,
        "obs_std": obs_norm.std,
        "act_min": act_norm.data_min,
        "act_max": act_norm.data_max,
        "act_scale": act_norm.scale,
        "active_indices": model._active_indices.tolist(),
        "masked_indices": model._masked_indices.tolist(),
        "active_mask": model._active_mask.tolist(),
        "const_values": model._const_values.tolist(),
    }, norm_path)

    metadata = {
        "task_id": task_id,
        "model_type": "bc_expert_v4",
        "model_file": f"{task_id}_bc_expert.pt",
        "norm_file": f"{task_id}_norm_stats.pt",
        "obs_dim": int(obs_norm.mean.shape[0]),
        "action_dim": 7,
        "network_output_dim": int(model._active_mask.sum()),
        "active_indices": model._active_indices.tolist(),
        "masked_indices": model._masked_indices.tolist(),
        "hidden_dim": config["bc_hidden"],
        "n_layers": 3,
        "n_demos": config["n_demos"],
        "bc_epochs": config["bc_epochs"],
        "bc_lr": config["bc_lr"],
        "bc_batch_size": config["bc_batch_size"],
        "final_loss": float(metrics["final_loss"]),
        "best_loss": float(metrics["best_loss"]),
        "eval_success_rate": float(eval_sr),
        "control_mode": config.get("control_mode", "pd_ee_delta_pose"),
        "obs_mode": "state_dict",
    }
    meta_path = os.path.join(task_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved model: {model_path}")
    print(f"  Saved norm stats: {norm_path}")
    print(f"  Saved metadata: {meta_path}")
    return task_dir


# ============================================================
# Main
# ============================================================

def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Train BC Expert Models (v4)")
    parser.add_argument("--task", type=str, default=None,
                        help="Single task to train (e.g. PickCube-v1)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    OUTPUT_DIR = args.output

    if args.task:
        tasks = [args.task] if args.task in TASK_CONFIG else list(TASK_CONFIG.keys())
    else:
        tasks = ["PickCube-v1", "StackCube-v1"]

    print("=" * 70)
    print("BC Expert Training Pipeline v5 (zero-var masking + DAgger)")
    print("=" * 70)
    print(f"  Tasks: {tasks}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Eval episodes: {args.eval_episodes}")
    print("=" * 70)

    results = {}

    for task_id in tasks:
        config = TASK_CONFIG[task_id]
        print(f"\n{'='*60}")
        print(f"BC Expert Training: {task_id}")
        print(f"{'='*60}")

        print(f"\n[STEP 1] Determining obs dimension...")
        obs_dim = get_obs_dim(task_id, config)
        print(f"  Obs dim: {obs_dim}")

        n_demos = config["n_demos"]
        print(f"\n[STEP 2] Collecting {n_demos} expert demonstrations...")
        t0 = time.time()
        obs_list, act_list, demo_stats = collect_demos(task_id, config)
        demo_time = (time.time() - t0) / 60
        print(f"  Demo collection done in {demo_time:.1f} min")
        print(f"  Transitions: {demo_stats['total_transitions']}")
        print(f"  Expert SR: {demo_stats['success_rate']*100:.0f}%")
        print(f"  Mean ep length: {demo_stats['mean_ep_length']:.1f}")

        if demo_stats["success_rate"] < 0.8:
            print(f"  [WARNING] Expert SR only {demo_stats['success_rate']*100:.0f}%")

        print(f"\n[STEP 3] Training BC MLP ({config['bc_epochs']} epochs, "
              f"hidden={config['bc_hidden']}, demos={n_demos})...")
        t0 = time.time()
        model, obs_norm, act_norm, train_metrics = train_bc(
            obs_list, act_list, obs_dim, config
        )
        train_time = (time.time() - t0) / 60
        print(f"  Initial training done in {train_time:.1f} min")

        # ---- DAgger iterative refinement (if enabled) ----
        dagger_rounds = config.get("dagger_rounds", 0)
        dagger_rollouts = config.get("dagger_rollouts", 0)

        if dagger_rounds > 0 and dagger_rollouts > 0:
            print(f"\n[STEP 3b] DAgger iterative refinement ({dagger_rounds} rounds, "
                  f"{dagger_rollouts} rollouts/round)...")
            
            # Create scripted expert for relabeling
            expert_cls = config["expert_cls"]
            scripted_expert = expert_cls()
            
            # Keep initial data for combined retraining
            initial_obs = list(obs_list)
            initial_act = list(act_list)
            all_dagger_obs = []
            all_dagger_act = []

            for round_i in range(dagger_rounds):
                print(f"\n  --- DAgger Round {round_i+1}/{dagger_rounds} ---")
                print(f"  Rolling out BC model ({dagger_rollouts} episodes)...")
                
                new_obs, new_act = dagger_rollout_and_relabel(
                    model, obs_norm, act_norm, scripted_expert,
                    task_id, config, n_episodes=dagger_rollouts
                )
                
                all_dagger_obs.extend(new_obs)
                all_dagger_act.extend(new_act)
                
                print(f"  Retraining on combined data...")
                t0 = time.time()
                model, obs_norm, act_norm, train_metrics = retrain_with_dagger(
                    model, obs_norm, act_norm,
                    initial_obs, initial_act,
                    all_dagger_obs, all_dagger_act,
                    obs_dim, config
                )
                round_time = (time.time() - t0) / 60
                print(f"  Round {round_i+1} done in {round_time:.1f} min")

                # Quick eval to check progress
                quick_sr, _ = evaluate_bc(
                    model, obs_norm, act_norm, task_id, config,
                    n_episodes=10
                )
                print(f"  Quick eval after round {round_i+1}: {quick_sr*100:.0f}%")
                
                # Stop early if target reached
                if quick_sr >= config["target_success"]:
                    print(f"  Target {config['target_success']*100:.0f}% reached! Stopping DAgger.")
                    break
            
            # Free initial data
            del initial_obs, initial_act, all_dagger_obs, all_dagger_act

        del obs_list, act_list

        print(f"\n[STEP 4] Evaluating BC model ({args.eval_episodes} episodes)...")
        eval_sr, eval_details = evaluate_bc(
            model, obs_norm, act_norm, task_id, config,
            n_episodes=args.eval_episodes
        )

        task_dir = save_model(model, obs_norm, act_norm, task_id, config,
                              train_metrics, eval_sr)

        with open(os.path.join(task_dir, "demo_stats.json"), "w") as f:
            json.dump(demo_stats, f, indent=2)

        results[task_id] = {
            "success_rate": eval_sr,
            "target": config["target_success"],
            "status": "OK" if eval_sr >= config["target_success"] else "BELOW_TARGET",
            "dir": task_dir,
        }

        target = config["target_success"]
        if eval_sr >= target:
            print(f"\n  [OK] {task_id}: {eval_sr*100:.0f}% >= {target*100:.0f}%")
        else:
            print(f"\n  [WARNING] {task_id}: {eval_sr*100:.0f}% < {target*100:.0f}%")

    print(f"\n{'='*70}")
    print("ALL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Output directory: {OUTPUT_DIR}")
    for task_id, r in results.items():
        status_icon = "OK" if r["status"] == "OK" else "BELOW TARGET"
        print(f"  {task_id}: {r['success_rate']*100:.0f}% "
              f"(target: {r['target']*100:.0f}%) [{status_icon}]")


if __name__ == "__main__":
    main()
