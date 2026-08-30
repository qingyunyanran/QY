# -*- coding: utf-8 -*-
"""
Unified 42-dim observation interface for the ManiSkill3 CL benchmark.

WHY
---
The five Franka Panda tabletop tasks ship with different state-obs dims
(Push/Pull 35, Pick 42, Stack 48, Poke 54). The difference comes ONLY from
the number of scene objects and from convenience relative vectors
(tcp_to_obj, obj_to_goal, ...). The relative vectors are linear differences
of absolute poses (zero new information); PushCube and PullCube contain ZERO
relative vectors and both train to 100% SR. Instead of adding a learned
mapping layer (which cost ~30 percentage points of single-task SR), we align
observations BY PHYSICAL MEANING into fixed semantic slots.

SLOT LAYOUT (42 dims, identical physical meaning in every task)
--------------------------------------------------------------
  [ 0:18] agent proprioception (Panda qpos/qvel/...)   - 5 tasks identical
  [18:25] tcp_pose          (7: xyz + quaternion)
  [25:32] object SLOT 1     (7): PRIMARY manipulated object
           Push/Pull/Pick: the cube ; Stack: cubeA ; Poke: the cube
  [32:39] object SLOT 2     (7): second object / tool
           Stack: cubeB ; Poke: the peg ; single-object tasks: zeros
  [39:42] goal_pos          (3: xyz of goal region)
           Stack: zeros (goal = on top of cubeB, implicit via slot 2)
           Poke: reconstructed as cube_pos + cube_to_goal_pos
                 (the env's "goal_pos" key actually stores the PEG position)

Removed as redundant (27 dims across the 5 tasks):
  tcp_to_obj_pos, obj_to_goal_pos, tcp_to_cubeA_pos, tcp_to_cubeB_pos,
  cubeA_to_cubeB_pos, tcp_to_peg_pos, peg_to_cube_pos, cube_to_goal_pos,
  peghead_to_cube_pos, is_grasped (inferable from gripper qpos).
All ABSOLUTE poses (tcp / object / goal) are KEPT - the scene is localized
exclusively by these, and they are untouched.
"""
import numpy as np
import torch

UNIFIED_OBS_DIM = 42
AGENT_DIM = 18

# Per-task slot binding. Key names must match _get_obs_extra in the task sources.
SLOT_LAYOUT = {
    "PushCube-v1":  {"slot1": "obj_pose",   "slot2": None,         "goal": "goal_pos"},
    "PullCube-v1":  {"slot1": "obj_pose",   "slot2": None,         "goal": "goal_pos"},
    "PickCube-v1":  {"slot1": "obj_pose",   "slot2": None,         "goal": "goal_pos"},
    "StackCube-v1": {"slot1": "cubeA_pose", "slot2": "cubeB_pose", "goal": None},
    "PokeCube-v1":  {"slot1": "cube_pose",  "slot2": "peg_pose",   "goal": "poke_reconstruct"},
}

# Every extra key the five tasks can emit. Anything else is treated as agent
# proprioception (the 18-dim Panda state block).
EXTRA_KEYS = {
    "tcp_pose", "goal_pos", "obj_pose", "is_grasped",
    "tcp_to_obj_pos", "obj_to_goal_pos",
    "cubeA_pose", "cubeB_pose", "tcp_to_cubeA_pos", "tcp_to_cubeB_pos", "cubeA_to_cubeB_pos",
    "cube_pose", "peg_pose", "tcp_to_peg_pos", "peg_to_cube_pos",
    "cube_to_goal_pos", "peghead_to_cube_pos",
}


def _flatten_obs(obs):
    """Flatten a (possibly nested) ManiSkill obs dict into {key: 1-D float32 array}."""
    out = {}

    def _walk(key, o):
        if isinstance(o, dict):
            for k, v in o.items():
                _walk(k, v)
        elif isinstance(o, torch.Tensor):
            out[key] = o.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            arr = np.asarray(o, dtype=np.float32).reshape(-1)
            if arr.shape[0] > 0:
                out[key] = arr

    if isinstance(obs, dict):
        for k, v in obs.items():
            _walk(k, v)
    else:  # already a flat tensor / array
        out["agent"] = np.asarray(obs, dtype=np.float32).reshape(-1)
    return out


def _take(flat, key, dim):
    assert key in flat, f"obs missing key '{key}', available: {sorted(flat.keys())}"
    v = flat[key]
    assert v.shape[0] >= dim, f"key '{key}' has dim {v.shape[0]} < expected {dim}"
    return v[:dim].astype(np.float32)


def build_unified_obs(obs, cfg):
    """Map a raw ManiSkill state obs (dict, batched num_envs=1) to the 42-dim vector."""
    flat = _flatten_obs(obs)
    u = np.zeros(UNIFIED_OBS_DIM, dtype=np.float32)

    # [0:18] agent proprioception = everything that is not a task extra key
    agent_parts = [v for k, v in flat.items() if k not in EXTRA_KEYS]
    assert agent_parts, f"no agent keys found in obs: {sorted(flat.keys())}"
    agent_vec = np.concatenate(agent_parts).astype(np.float32)
    assert agent_vec.shape[0] == AGENT_DIM, (
        f"agent proprioception dim {agent_vec.shape[0]} != {AGENT_DIM}; "
        f"non-extra keys: {[k for k in flat if k not in EXTRA_KEYS]}")
    u[0:18] = agent_vec

    # [18:25] tcp pose (all tasks)
    u[18:25] = _take(flat, "tcp_pose", 7)

    # [25:32] slot 1: primary manipulated object
    u[25:32] = _take(flat, cfg["slot1"], 7)

    # [32:39] slot 2: second object / tool (zero-padded for single-object tasks)
    if cfg["slot2"] is not None:
        u[32:39] = _take(flat, cfg["slot2"], 7)

    # [39:42] goal position
    if cfg["goal"] == "goal_pos":
        u[39:42] = _take(flat, "goal_pos", 3)
    elif cfg["goal"] == "poke_reconstruct":
        # PokeCube never exposes goal_region xyz directly; recover it from the
        # (otherwise discarded) relative vector: goal = cube + (goal - cube).
        cube_xyz = _take(flat, "cube_pose", 7)[:3]
        u[39:42] = cube_xyz + _take(flat, "cube_to_goal_pos", 3)
    # StackCube: goal slot stays zero (target = on top of cubeB, already in slot 2)
    return u


def slot_mask(cfg):
    """Boolean mask of the 42 dims carrying real data (True=real, False=padded zero).
    Reserved for masked normalization; current trainer does not normalize obs."""
    m = np.ones(UNIFIED_OBS_DIM, dtype=bool)
    if cfg["slot2"] is None:
        m[32:39] = False
    if cfg["goal"] is None:
        m[39:42] = False
    return m


# ============================================================
# Batched version for vectorized training (num_envs > 1)
# ============================================================

def _leaves_torch(obs):
    """Like _flatten_obs but keeps tensors batched: {key: (N, d) tensor}."""
    out = {}

    def _walk(key, o):
        if isinstance(o, dict):
            for k, v in o.items():
                _walk(k, v)
        elif isinstance(o, torch.Tensor):
            if o.numel() > 0:
                out[key] = o.float()
        else:
            t = torch.as_tensor(o, dtype=torch.float32)
            if t.numel() > 0:
                out[key] = t

    if isinstance(obs, dict):
        for k, v in obs.items():
            _walk(k, v)
    else:
        out["agent"] = torch.as_tensor(obs, dtype=torch.float32)
    return out


def build_unified_obs_batch(obs, cfg):
    """Map a BATCHED state_dict obs (nested dict of (N, d) tensors) to (N, 42)."""
    flat = _leaves_torch(obs)
    n_envs = next(iter(flat.values())).shape[0]
    u = torch.zeros((n_envs, UNIFIED_OBS_DIM), dtype=torch.float32)

    agent_parts = [v.reshape(n_envs, -1) for k, v in flat.items() if k not in EXTRA_KEYS]
    assert agent_parts, f"no agent keys found in batched obs: {sorted(flat.keys())}"
    agent_vec = torch.cat(agent_parts, dim=1)
    assert agent_vec.shape[1] == AGENT_DIM, (
        f"agent proprioception dim {agent_vec.shape[1]} != {AGENT_DIM}; "
        f"non-extra keys: {[k for k in flat if k not in EXTRA_KEYS]}")
    u[:, 0:18] = agent_vec

    u[:, 18:25] = flat["tcp_pose"].reshape(n_envs, -1)[:, :7]
    u[:, 25:32] = flat[cfg["slot1"]].reshape(n_envs, -1)[:, :7]
    if cfg["slot2"] is not None:
        u[:, 32:39] = flat[cfg["slot2"]].reshape(n_envs, -1)[:, :7]
    if cfg["goal"] == "goal_pos":
        u[:, 39:42] = flat["goal_pos"].reshape(n_envs, -1)[:, :3]
    elif cfg["goal"] == "poke_reconstruct":
        cube_xyz = flat["cube_pose"].reshape(n_envs, -1)[:, :3]
        u[:, 39:42] = cube_xyz + flat["cube_to_goal_pos"].reshape(n_envs, -1)[:, :3]
    return u
