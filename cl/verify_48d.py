# -*- coding: utf-8 -*-
"""Bit-level verification of the 48-dim unified interface for the 5-task
COMPLEX sequence (StackCube / StackPyramid / PegInsertionSide / PlugCharger /
PokeCube).

What is checked, per task:
  1. build_unified_obs == a vector assembled from an INDEPENDENTLY HARD-CODED
     offset table (a literal copy of the contract, so a wrong slot in
     unified_obs.SLOT_LAYOUT / _tail_plan is caught instead of silently
     agreeing with itself). Compared with array_equal -> bit-identical.
  2. dims nobody claims are exactly 0.0 (no stale data in the padding).
  3. the batched path (build_unified_obs_batch, used by GpuVecEnv) gives the
     same rows as the scalar path.
  4. StackCube only: the vector equals the NATIVE 48-dim state obs, so the
     official 48-dim checkpoint still loads without surgery.
  5. a short random rollout keeps every value finite and the success flag is
     reachable through the info dict.

Run: python verify_48d.py
"""
import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs  # noqa: F401

from unified_obs import (
    UNIFIED_OBS_DIM, SLOT_LAYOUT, build_unified_obs, build_unified_obs_batch,
    slot_mask,
)
from ppo_er_unified import TASK_INFO

# Independent copy of the slot contract: (offset, key-or-'rel:a-b', dim).
EXPECTED = {
    "StackCube-v1": [
        (0, "qpos", 9), (9, "qvel", 9), (18, "tcp_pose", 7),
        (25, "cubeA_pose", 7), (32, "cubeB_pose", 7),
        (39, "rel:cubeA_pose-tcp_pose", 3),
        (42, "rel:cubeB_pose-tcp_pose", 3),
        (45, "rel:cubeB_pose-cubeA_pose", 3),
    ],
    "StackPyramid-v1": [
        (0, "qpos", 9), (9, "qvel", 9), (18, "tcp_pose", 7),
        (25, "cubeA_pose", 7), (32, "cubeB_pose", 7), (39, "cubeC_pose", 7),
    ],
    "PegInsertionSide-v1": [
        (0, "qpos", 9), (9, "qvel", 9), (18, "tcp_pose", 7),
        (25, "peg_pose", 7), (32, "box_hole_pose", 7),
        (39, "peg_half_size", 3), (42, "box_hole_radius", 1),
    ],
    "PlugCharger-v1": [
        (0, "qpos", 9), (9, "qvel", 9), (18, "tcp_pose", 7),
        (25, "charger_pose", 7), (32, "receptacle_pose", 7),
        (39, "goal_pose", 7),
    ],
    "PokeCube-v1": [
        (0, "qpos", 9), (9, "qvel", 9), (18, "tcp_pose", 7),
        (25, "cube_pose", 7), (32, "peg_pose", 7), (39, "goal_pos", 3),
    ],
    # Replacement tasks (reward audit 2026-11-03).
    "PullCubeTool-v1": [
        (0, "qpos", 9), (9, "qvel", 9), (18, "tcp_pose", 7),
        (25, "cube_pose", 7), (32, "tool_pose", 7),
    ],
    "PlaceSphere-v1": [
        (0, "qpos", 9), (9, "qvel", 9), (18, "tcp_pose", 7),
        (25, "obj_pose", 7), (39, "bin_pos", 3),
    ],
    "RollBall-v1": [
        (0, "qpos", 9), (9, "qvel", 9), (18, "tcp_pose", 7),
        (25, "ball_pose", 7), (39, "goal_pos", 3), (42, "ball_vel", 3),
    ],
    "PickSingleYCB-v1": [
        (0, "qpos", 9), (9, "qvel", 9), (18, "tcp_pose", 7),
        (25, "obj_pose", 7), (39, "goal_pos", 3),
    ],
}


def flatten(o, key, out):
    if isinstance(o, dict):
        for k, v in o.items():
            flatten(v, k, out)
    else:
        out[key] = torch.as_tensor(o).float().reshape(-1).cpu().numpy()


def make(task, n=1):
    return gym.make(task, num_envs=n, obs_mode="state_dict",
                    control_mode=TASK_INFO[task]["control_mode"],
                    sim_backend="cpu", render_backend="none")


def expected_vector(flat, table):
    v = np.zeros(UNIFIED_OBS_DIM, dtype=np.float32)
    claimed = np.zeros(UNIFIED_OBS_DIM, dtype=bool)
    for off, key, dim in table:
        if key.startswith("rel:"):
            a, b = key[4:].split("-")
            v[off:off + dim] = flat[a][:dim] - flat[b][:dim]
        else:
            v[off:off + dim] = flat[key][:dim]
        claimed[off:off + dim] = True
    return v, claimed


def check(task):
    env = make(task)
    obs, _ = env.reset(seed=7)
    flat = {}
    flatten(obs, "", flat)

    u = build_unified_obs(obs, SLOT_LAYOUT[task])
    assert u.shape == (UNIFIED_OBS_DIM,), f"{task}: dim {u.shape} != 48"

    exp, claimed = expected_vector(flat, EXPECTED[task])
    if not np.array_equal(u, exp):
        bad = np.abs(u - exp) > 0
        raise AssertionError(
            f"{task}: SLOT MISMATCH at dims {np.nonzero(bad)[0].tolist()} "
            f"max|diff|={np.abs(u - exp).max():.3e}")

    assert np.all(u[~claimed] == 0.0), (
        f"{task}: padding dims not zero: {u[~claimed]}")

    # batched path must agree with the scalar path. num_envs>1 is illegal with
    # sim_backend="cpu" (ManiSkill raises), so the batch is faked by stacking
    # the single-env state_dict twice -- that is exactly the input shape
    # GpuVecEnv hands to build_unified_obs_batch.
    def _stack(o):
        if isinstance(o, dict):
            return {k: _stack(v) for k, v in o.items()}
        t = torch.as_tensor(o).float().reshape(1, -1)
        return t.repeat(2, 1)

    ub = build_unified_obs_batch(_stack(obs), SLOT_LAYOUT[task]).cpu().numpy()
    assert ub.shape == (2, UNIFIED_OBS_DIM), f"{task}: batch dim {ub.shape}"
    assert np.array_equal(ub[0], u) and np.array_equal(ub[1], u), (
        f"{task}: batch rows != scalar, max|diff|="
        f"{np.abs(ub[0] - u).max():.3e}")

    # short random rollout: finite everywhere, success flag reachable
    torch.manual_seed(0)
    succeeded_flag = False
    for _ in range(8):
        a = torch.rand((1, 8), dtype=torch.float32) * 2 - 1
        obs, rew, term, trunc, info = env.step(a.numpy())
        flat = {}
        flatten(obs, "", flat)
        uu = build_unified_obs(obs, SLOT_LAYOUT[task])
        assert np.all(np.isfinite(uu)), f"{task}: non-finite obs after step"
        if isinstance(info, dict) and info.get("success") is not None:
            succeeded_flag = True
    assert succeeded_flag, f"{task}: no 'success' key in info"

    # StackCube: must equal the NATIVE 48-dim state obs bit-for-bit
    if task == "StackCube-v1":
        native = np.concatenate([
            flat["qpos"], flat["qvel"], flat["tcp_pose"],
            flat["cubeA_pose"], flat["cubeB_pose"],
            flat["tcp_to_cubeA_pos"], flat["tcp_to_cubeB_pos"],
            flat["cubeA_to_cubeB_pos"],
        ]).astype(np.float32)
        assert native.shape[0] == UNIFIED_OBS_DIM, native.shape
        assert np.array_equal(uu, native), (
            f"StackCube: unified != native 48-dim, max|diff|="
            f"{np.abs(uu - native).max():.3e}")

    # mask sanity: the mask marks exactly the dims the table claims are real.
    # (Not [0:39]: a task with slot2=None -- PlaceSphere -- has no third pose,
    # so [32:39] is padding, and advertising it as real would feed the policy
    # 7 dims of zeros it cannot distinguish from a genuine object at origin.)
    m = slot_mask(SLOT_LAYOUT[task])
    assert np.array_equal(m, claimed), (
        f"{task}: slot_mask disagrees with the offset table at dims "
        f"{np.nonzero(m != claimed)[0].tolist()}")

    env.close()
    print(f"[OK] {task:20s} dim=48  slots_ok=True  batch_ok=True  "
          f"padding_clean=True  tail_mask={int(m[39:].sum())}/9")


if __name__ == "__main__":
    for t in EXPECTED:
        check(t)
    print("\nAll 48-dim slot bindings verified (bit-identical).")
