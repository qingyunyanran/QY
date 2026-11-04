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
  [39:48] task TAIL         (9): resolved by _tail_plan, in this priority:
           -- slot3 (7) : a THIRD 7-dim pose, for three-object / goal-pose
              tasks (StackPyramid: cubeC_pose; PlugCharger: goal_pose).
              Mutually exclusive with the goal/rel forms below.
           -- goal xyz (3) then REL vectors (up to 6)
              Push/Pull/Pick/PokeCube: [39:42]=goal_pos xyz
              LiftPegUpright: zeros (success = peg upright, implicit in slot 1
              orientation, no goal site in the scene)
              StackCube: goal is None, so [39:48] holds the three native REL
              vectors tcp_to_cubeA / tcp_to_cubeB / cubeA_to_cubeB. This makes
              the unified vector BIT-IDENTICAL to the native 48-dim state obs,
              so official 48-dim checkpoints load without surgery.
           -- `aux` scalars fill whatever is left (PegInsertionSide:
              peg_half_size(3) + box_hole_radius(1) at [39:43]; both are
              randomised per episode, so they carry real information).

Removed as redundant:
  tcp_to_obj_pos, is_grasped (inferable from gripper qpos).
  Both are linear differences / functions of the absolute poses (zero new
  information); PushCube/PullCube contain ZERO relative vectors and train
  to 100% SR.
  EXCEPTION: the 9 relative dims of StackCube are KEPT in [39:48]. Dropping
  them (42-dim build) made the stacking task untrainable from scratch and
  broke the official checkpoint (SR 97.6% -> 0%), because they are the
  direct terms of the success test and the network had to compute them.
All ABSOLUTE poses (tcp / object / goal) are KEPT - the scene is localized
exclusively by these, and they are untouched.
"""
import numpy as np
import torch

UNIFIED_OBS_DIM = 48
AGENT_DIM = 18
TAIL_START = 39

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
    # No goal site, but the native 48-dim obs carries three REL vectors
    # (cubeA-tcp, cubeB-tcp, cubeB-cubeA) that are the direct terms of the
    # stacking success test. They are reproduced exactly in [39:48] below, so
    # the unified vector is bit-identical to the native state obs.
    "StackCube-v1":       {"slot1": "cubeA_pose","slot2": "cubeB_pose","goal": None,
                           "rel": (("cubeA_pose", "tcp_pose", 3),
                                   ("cubeB_pose", "tcp_pose", 3),
                                   ("cubeB_pose", "cubeA_pose", 3))},
    # PullCubeTool: hook the cube (slot1) with the held L-shaped tool
    # (slot2). No goal_pos key in obs (implicit target), horizon=100.
    "PullCubeTool-v1":    {"slot1": "cube_pose", "slot2": "tool_pose", "goal": None},
    # PlaceSphere: pick the sphere (slot1) and drop it into the shallow bin.
    # bin_pos is a 3-dim POINT, so it takes the goal block at [39:42] exactly
    # like PushCube. tcp_to_obj_pos / is_grasped stay dropped (both are
    # functions of the poses we keep, see the "removed as redundant" note).
    "PlaceSphere-v1":     {"slot1": "obj_pose", "slot2": None, "goal": "bin_pos"},
    # RollBall: push the ball (slot1) to the goal point; horizon 80. The ball
    # keeps rolling after the push, so its VELOCITY is real information (the
    # policy has to brake it) -- it goes in the free 6 dims after the goal.
    "RollBall-v1":        {"slot1": "ball_pose", "slot2": None, "goal": "goal_pos",
                           "aux": (("ball_vel", 3),)},
    # PickSingleYCB: pick ONE randomly sampled YCB object (slot1) and carry it to
    # a goal POINT. Same shape as PickCube, but the object geometry is resampled
    # every episode, so the policy has to generalise over the grasp instead of
    # memorising one box. Horizontal 2.5 cm success threshold.
    "PickSingleYCB-v1":   {"slot1": "obj_pose", "slot2": None, "goal": "goal_pos"},
    # ---- COMPLEX-SKILL SEQUENCE (probe 2026-11-03) ------------------------
    # Three-object / goal-pose tasks need one more 7-dim slot, so [39:48] is
    # split into slot3 [39:46] + a 2-dim leftover. slot3 and goal/rel are
    # MUTUALLY EXCLUSIVE (both compete for the tail): a task that has a goal
    # POSE uses slot3, a task with a goal POINT uses goal xyz at [39:42].
    #
    # StackPyramid: stack cubeA/cubeB/cubeC into a pyramid; horizon 250. The
    # three tcp_to_* / cube*_to_* relative vectors are dropped (differences of
    # the poses we keep). The goal is a stack configuration, not a site.
    "StackPyramid-v1":    {"slot1": "cubeA_pose", "slot2": "cubeB_pose",
                           "slot3": "cubeC_pose", "goal": None},
    # PegInsertionSide: insert the held long peg into the box's SIDE hole;
    # horizon 100. peg_half_size and box_hole_radius vary per episode (the
    # peg length is randomised), so they are NOT dropped -- they go in the
    # `aux` block right after the two pose slots.
    "PegInsertionSide-v1": {"slot1": "peg_pose", "slot2": "box_hole_pose",
                            "goal": None,
                            "aux": (("peg_half_size", 3), ("box_hole_radius", 1))},
    # PlugCharger: plug the charger into the receptacle; horizon 200. Success
    # is the charger matching goal_pose, which is emitted directly, so it is
    # an ABSOLUTE 7-dim pose -> slot3 (goal_pose[0:3] would lose the rotation
    # the insertion has to match).
    "PlugCharger-v1":     {"slot1": "charger_pose", "slot2": "receptacle_pose",
                           "slot3": "goal_pose", "goal": None},
}

# Every extra key the adapted tasks can emit. Anything else is treated as
# agent proprioception (the 18-dim Panda state block).
EXTRA_KEYS = {
    "tcp_pose", "goal_pos", "obj_pose", "is_grasped",
    "tcp_to_obj_pos", "obj_to_goal_pos",
    # two-object / tool tasks
    "cube_pose", "peg_pose", "cubeA_pose", "cubeB_pose", "tool_pose",
    # complex-skill sequence
    "cubeC_pose", "box_hole_pose", "box_hole_radius", "peg_half_size",
    "charger_pose", "receptacle_pose", "goal_pose",
    # PlaceSphere (goal POINT, like goal_pos but named after the bin)
    "bin_pos",
    # RollBall
    "ball_pose", "ball_vel",
}


def _tail_plan(cfg):
    """Resolve which key lands in which dim of the 9-dim tail [39:48].

    Returns a list of (offset, kind, payload, dim) with kind in
      "abs"  -- payload is a key name, copied verbatim (7-dim pose / scalars)
      "goal" -- payload is a key name, first 3 dims (goal site xyz)
      "rel"  -- payload is (key_a, key_b), filled as a - b
    slot3 (7) and goal/rel compete for the same 9 dims and are therefore
    mutually exclusive; `aux` scalars always follow whatever came first and
    fill the leftover space.
    """
    plan = []
    off = TAIL_START
    if cfg.get("slot3") is not None:
        assert cfg["goal"] is None and not cfg.get("rel"), (
            "slot3 and goal/rel both claim the 9-dim tail -- a task cannot "
            "have a goal POSE and a goal POINT at the same time")
        plan.append((off, "abs", cfg["slot3"], 7))
        off += 7
    else:
        if cfg["goal"] is not None:
            plan.append((off, "goal", cfg["goal"], 3))
            off += 3
        for key_a, key_b, dim in cfg.get("rel", ()):
            plan.append((off, "rel", (key_a, key_b), dim))
            off += dim
    for key, dim in cfg.get("aux", ()):
        plan.append((off, "abs", key, dim))
        off += dim
    assert off <= UNIFIED_OBS_DIM, (
        f"tail overflow: task needs dim {off} > {UNIFIED_OBS_DIM}")
    return plan


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

    # [39:48] task tail (9 dims): one of
    #   - slot3: a third 7-dim pose (complex tasks) + aux scalars,
    #   - goal site xyz (3) then REL vectors, or
    #   - REL vectors only (StackCube reproduces the native obs[39:48]).
    # For StackCube this is bit-identical to the native 48-dim state obs
    # (tcp_to_cubeA, tcp_to_cubeB, cubeA_to_cubeB), so the official 48-dim
    # checkpoint loads without surgery.
    for off_t, kind, payload, dim in _tail_plan(cfg):
        if kind == "rel":
            key_a, key_b = payload
            u[off_t:off_t + dim] = _take(flat, key_a, dim) - _take(flat, key_b, dim)
        else:  # "abs" (pose / scalars) or "goal" (xyz of a site)
            u[off_t:off_t + dim] = _take(flat, payload, dim)
    return u


def slot_mask(cfg):
    """Boolean mask of the unified dims carrying real data (True=real, False=padded zero).
    Reserved for masked normalization; current trainer does not normalize obs."""
    m = np.ones(UNIFIED_OBS_DIM, dtype=bool)
    if cfg["slot2"] is None:
        m[32:39] = False
    m[TAIL_START:] = False
    for off, _kind, _payload, dim in _tail_plan(cfg):
        m[off:off + dim] = True
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


def build_unified_obs_batch(obs, cfg, device=None):
    """Map a BATCHED state_dict obs (nested dict of (N, d) tensors) to (N, 42).

    `device` defaults to the device of the incoming tensors, so with a GPU sim
    backend the whole mapping stays on the GPU (no host round-trip per step).
    """
    flat = _leaves_torch(obs)
    ref = next(iter(flat.values()))
    n_envs = ref.shape[0] if ref.dim() > 0 else 1
    if device is None:
        device = ref.device
    u = torch.zeros((n_envs, UNIFIED_OBS_DIM), dtype=torch.float32, device=device)

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
    # [39:48] task tail, resolved by _tail_plan (slot3 / goal / rel / aux)
    for off_t, kind, payload, dim in _tail_plan(cfg):
        if kind == "rel":
            key_a, key_b = payload
            u[:, off_t:off_t + dim] = (flat[key_a].reshape(n_envs, -1)[:, :dim]
                                       - flat[key_b].reshape(n_envs, -1)[:, :dim])
        else:  # "abs" (pose / scalars) or "goal" (xyz of a site)
            u[:, off_t:off_t + dim] = flat[payload].reshape(n_envs, -1)[:, :dim]
    return u
