# -*- coding: utf-8 -*-
"""Bit-level check of the unified 48-dim vector for LiftPegUpright-v1.

LiftPegUpright's native state obs is 32 dims: agent 18 (qpos 9 + qvel 9)
+ tcp_pose 7 + obj_pose 7. The unified layout puts those blocks in
[0:18] / [18:25] / [25:32] and zero-pads [32:48], so the check is element-wise
equality per block plus "no native key is dropped".
"""
import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa: F401
from unified_obs import build_unified_obs, SLOT_LAYOUT, UNIFIED_OBS_DIM, _flatten_obs

task = "LiftPegUpright-v1"
env = gym.make(task, obs_mode="state_dict", control_mode="pd_joint_delta_pos",
               render_mode=None, render_backend="none")
cfg = SLOT_LAYOUT[task]


def agent_vec(obs):
    d = obs["agent"]
    return np.concatenate([np.asarray(d["qpos"], dtype=np.float32).reshape(-1),
                           np.asarray(d["qvel"], dtype=np.float32).reshape(-1)])


obs, _ = env.reset(seed=0)
worst = {}
for step in range(200):
    a = env.action_space.sample() * 0.5
    obs, *_ = env.step(np.asarray(a, dtype=np.float32))
    u = build_unified_obs(obs, cfg)
    ref = {
        "agent[0:18]": agent_vec(obs),
        "tcp_pose[18:25]": np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32).reshape(-1),
        "obj_pose[25:32]": np.asarray(obs["extra"]["obj_pose"], dtype=np.float32).reshape(-1),
    }
    sl = {"agent[0:18]": (0, 18), "tcp_pose[18:25]": (18, 25), "obj_pose[25:32]": (25, 32)}
    for k, v in ref.items():
        a_, b_ = sl[k]
        d = float(np.abs(u[a_:b_] - v[: b_ - a_]).max())
        worst[k] = max(worst.get(k, 0.0), d)
    tail = float(np.abs(u[32:UNIFIED_OBS_DIM]).max())
    worst["tail_zero[32:48]"] = max(worst.get("tail_zero[32:48]", 0.0), tail)

flat = _flatten_obs(obs)
print(f"task={task}  unified_dim={UNIFIED_OBS_DIM}  flattened native keys={len(flat)}")
for k, v in worst.items():
    flag = "OK" if v < 1e-6 else "MISMATCH"
    print(f"  {flag:8s} max|diff| {k:20s} = {v:.3e}")

covered = {"tcp_pose", cfg["slot1"], cfg.get("slot2"), cfg.get("goal")}
covered = {c for c in covered if c}
print("  native keys:")
for k, v in sorted(flat.items()):
    print(f"    {k:16s} dim={v.shape[0]:2d} -> {'USED' if k in covered else 'DROPPED'}")
env.close()
