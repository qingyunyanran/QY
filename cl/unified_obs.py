# -*- coding: utf-8 -*-
"""
Unified 42-dim observation interface for the ManiSkill3 CL benchmark.

WHY
---
The tabletop tasks ship with different state-obs dims
(Push/Pull 35, Pick 42, LiftPegUpright 32; PushT uses panda_stick with a
14-dim proprioception block). The difference
comes ONLY from the number of scene objects and convenience relative vectors
(tcp_to_obj, obj_to_goal, ...). The relative vectors are linear differences
of absolute poses (zero new information); PushCube and PullCube contain ZERO
relative vectors and both train to 100% SR. Instead of adding a learned
mapping layer (which cost ~30 percentage points of single-task SR), we align
observations BY PHYSICAL MEANING into fixed semantic slots.

SLOT LAYOUT (42 dims, identical physical meaning in every task)
--------------------------------------------------------------
  [ 0:18] agent proprioception, laid out by physical meaning:
             [0:9] qpos = 7 arm joints + 2 gripper joints
             [9:18] qvel = same ordering
          Panda tasks: 9-dim qpos/qvel; PushT uses panda_stick (no gripper):
          7-dim qpos/qvel, gripper slots [7:9]/[16:18] zero-padded so the arm
          joints keep identical positions across tasks.
  [18:25] tcp_pose          (7: xyz + quaternion)
  [25:32] object SLOT 1     (7): PRIMARY manipulated object
           Push/Pull/Pick: the cube ; LiftPegUpright: the peg
  [32:39] object SLOT 2     (7): second object / tool
           reserved slot -> zeros for all 4 tasks (single-object tasks)
  [39:42] goal_pos          (3: xyz of goal region)
           Push/Pull/Pick: goal_pos key ;
           LiftPegUpright: zeros (success = peg upright, implicit in slot 1
           orientation, no goal site in the scene)

Removed as redundant:
  tcp_to_obj_pos, is_grasped (inferable from gripper qpos).
  Both are linear differences / functions of the absolute poses (zero new
  information); PushCube/PullCube contain ZERO relative vectors and train
  to 100% SR.
All ABSOLUTE poses (tcp / object / goal) are KEPT - the scene is localized
exclusively by these, and they are untouched.
"""
import numpy as np
import torch

UNIFIED_OBS_DIM = 42
AGENT_DIM = 18

# Per-task slot binding. Key names must match _get_obs_extra in the task sources.
SLOT_LAYOUT = {
    "PushCube-v1":        {"slot1": "obj_pose",  "slot2": None, "goal": "goal_pos"},
    "PullCube-v1":        {"slot1": "obj_pose",  "slot2": None, "goal": "goal_pos"},
    "PickCube-v1":        {"slot1": "obj_pose",  "slot2": None, "goal": "goal_pos"},
    "LiftPegUpright-v1":  {"slot1": "obj_pose",  "slot2": None, "goal": None},
    # PushT uses the panda_stick embodiment (no gripper): qpos/qvel are 7-dim
    # instead of 9; the gripper slots [7:9]/[16:18] are zero-padded so the
    # arm joints keep the same positions as in every other task.
    "PushT-v1":           {"slot1": "obj_pose",  "slot2": None, "goal": "goal_pos"},
    # Two-object / tool tasks (probe 2026-09-09):
    # slot1 = PRIMARY target object (the thing that must end up right),
    # slot2 = tool / second object (held peg, base cube, or L-tool).
    # PokeCube: poke the free cube (slot1) with the held peg (slot2) to goal.
    "PokeCube-v1":        {"slot1": "cube_pose", "slot2": "peg_pose",  "goal": "goal_pos"},
    # StackCube: stack cubeA (slot1, carried) onto cubeB (slot2, base).
    # No goal site: success is encoded in the slot1/slot2 relative pose.
    "StackCube-v1":       {"slot1": "cubeA_pose","slot2": "cubeB_pose","goal": None},
    # PullCubeTool: hook the cube (slot1) with the held L-shaped tool
    # (slot2). No goal_pos key in obs (implicit target), horizon=100.
    "PullCubeTool-v1":    {"slot1": "cube_pose", "slot2": "tool_pose", "goal": None},
}

# Every extra key the adapted tasks can emit. Anything else is treated as
# agent proprioception (the 18-dim Panda state block).
EXTRA_KEYS = {
    "tcp_pose", "goal_pos", "obj_pose", "is_grasped",
    "tcp_to_obj_pos", "obj_to_goal_pos",
    # two-object / tool tasks
    "cube_pose", "peg_pose", "cubeA_pose", "cubeB_pose", "tool_pose",
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

    # [0:18] agent proprioception, laid out by physical meaning:
    #   [0:9]  qpos (7 arm joints + 2 gripper joints; gripper slots zero-padded)
    #   [9:18] qvel (same ordering)
    # Panda tasks have 9-dim qpos/qvel; panda_stick tasks (PushT) have 7-dim
    # qpos/qvel and no gripper -> arm joints stay in [0:7]/[9:16], gripper
    # slots stay zero.
    assert "qpos" in flat and "qvel" in flat, (
        f"obs missing 'qpos'/'qvel', available: {sorted(flat.keys())}")
    qpos = flat["qpos"]
    qvel = flat["qvel"]
    n_arm = min(qpos.shape[0], 7)
    assert n_arm == min(qvel.shape[0], 7), "qpos/qvel arm-dim mismatch"
    u[0:n_arm] = qpos[:n_arm]
    if qpos.shape[0] >= 9:  # full Panda (7 arm + 2 gripper)
        u[7:9] = qpos[7:9]
    u[9:9 + n_arm] = qvel[:n_arm]
    if qvel.shape[0] >= 9:
        u[16:18] = qvel[7:9]

    # [18:25] tcp pose (all tasks)
    u[18:25] = _take(flat, "tcp_pose", 7)

    # [25:32] slot 1: primary manipulated object
    u[25:32] = _take(flat, cfg["slot1"], 7)

    # [32:39] slot 2: second object / tool (zero-padded for single-object tasks)
    if cfg["slot2"] is not None:
        u[32:39] = _take(flat, cfg["slot2"], 7)

    # [39:42] goal position: cfg["goal"] names the extra key holding the goal
    # xyz; None means the task has no goal site (LiftPegUpright: success is the
    # peg's own upright orientation, already encoded in slot 1) -> zeros.
    if cfg["goal"] is not None:
        u[39:42] = _take(flat, cfg["goal"], 3)
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

    # agent block laid out by physical meaning (see build_unified_obs comment):
    # [0:9] qpos (7 arm + 2 gripper; gripper zero-padded for panda_stick),
    # [9:18] qvel same ordering.
    assert "qpos" in flat and "qvel" in flat, (
        f"batched obs missing 'qpos'/'qvel', available: {sorted(flat.keys())}")
    qpos = flat["qpos"].reshape(n_envs, -1)
    qvel = flat["qvel"].reshape(n_envs, -1)
    n_arm = min(qpos.shape[1], 7)
    assert n_arm == min(qvel.shape[1], 7), "qpos/qvel arm-dim mismatch"
    u[:, 0:n_arm] = qpos[:, :n_arm]
    if qpos.shape[1] >= 9:  # full Panda (7 arm + 2 gripper)
        u[:, 7:9] = qpos[:, 7:9]
    u[:, 9:9 + n_arm] = qvel[:, :n_arm]
    if qvel.shape[1] >= 9:
        u[:, 16:18] = qvel[:, 7:9]

    u[:, 18:25] = flat["tcp_pose"].reshape(n_envs, -1)[:, :7]
    u[:, 25:32] = flat[cfg["slot1"]].reshape(n_envs, -1)[:, :7]
    if cfg["slot2"] is not None:
        u[:, 32:39] = flat[cfg["slot2"]].reshape(n_envs, -1)[:, :7]
    if cfg["goal"] is not None:
        u[:, 39:42] = flat[cfg["goal"]].reshape(n_envs, -1)[:, :3]
    return u
