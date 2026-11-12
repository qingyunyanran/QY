"""Rule-based grasp + learned Cartesian insertion for PlugCharger.

The approach stage no longer comes from BC.  Measured over 543 expert episodes,
the tcp pose relative to the charger at the instant the fingers close is:

    translation [-18.3, +0.2, +5.6] mm   (std 6.8-11.3 mm)
    rotation    0.0019 rad from the mean (p90 0.0040)

i.e. the grasp orientation is essentially the charger orientation.  The approach
is rebuilt in closed loop as T_grasp = T_charger @ REL, using a four-stage
schedule so that each IK query only has to change one thing at a time:

    stage 0  move to the pre-grasp point, keeping the current orientation
    stage 1  rotate in place onto the grasp orientation
    stage 2  feed straight in to the grasp pose
    stage 3  close the fingers
    stage 4  insert with the learned pi_cart displacement policy + exact IK

mplib is used only for per-step analytic IK (no trajectory planning).
"""
import os
import sys

sys.path.insert(0, "/root/cl")

import numpy as np
import torch
import torch.nn as nn
import transforms3d as t3d

import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver)

from unified_obs import SLOT_LAYOUT, build_unified_obs_batch

TASK = "PlugCharger-v1"
N = int(os.environ.get("PROBE_N", "20"))
SEED = int(os.environ.get("PROBE_SEED", "80000"))
HOR = int(os.environ.get("PROBE_HOR", "400"))
GAINC = float(os.environ.get("PROBE_GAIN", "2.5"))
# The joint controller only realises ~40% of a commanded delta, so the IK step
# is scaled by 1/0.396 to actually land on the requested pose.
IGAIN = float(os.environ.get("PROBE_IGAIN", "1.0"))
DBG = bool(os.environ.get("PROBE_DBG"))
REL = np.array([-0.01828, 0.00016, 0.00557])
# Gripper orientation relative to the charger at finger closure, calibrated over
# 586 demo grasps (angular spread med 0.53 deg, p90 0.92 deg).  Omitting it
# left the target 180 deg away from every real grasp.
REL_QUAT = np.array([-0.000205, 0.991622, 0.000165, 0.129171])
MAXROT = 0.05
layout = SLOT_LAYOUT[TASK]

env = gym.make(TASK, num_envs=1, obs_mode="state_dict",
               control_mode="pd_joint_delta_pos", sim_backend="cpu",
               render_backend="none", max_episode_steps=HOR)
u = env.unwrapped


def _R(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _M(p, q):
    M = np.eye(4)
    M[:3, :3] = _R(q)
    M[:3, 3] = p
    return M


def _n(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def _ang(q1, q2):
    return 2.0 * np.arccos(min(1.0, abs(float(np.dot(_n(q1), _n(q2))))))


def _slerp(q1, q2, t):
    """Shortest-arc spherical interpolation of unit quaternions (w,x,y,z)."""
    q1, q2 = _n(q1), _n(q2)
    d = float(np.dot(q1, q2))
    if d < 0.0:
        q2, d = -q2, -d
    if d > 0.9995:
        q = q1 + t * (q2 - q1)
    else:
        th = np.arccos(min(1.0, d))
        q = (np.sin((1.0 - t) * th) * q1 + np.sin(t * th) * q2) / np.sin(th)
    return _n(q)


ck = torch.load(os.environ.get("CART_CKPT", "/root/cl/ckpts/plg_cart2.pt"),
                map_location="cpu")
pi = nn.Sequential(nn.Linear(66, 256), nn.Tanh(), nn.Linear(256, 256),
                   nn.Tanh(), nn.Linear(256, 4))
pi.load_state_dict(ck["pi"])
pi.eval()

ok, ikfail, nsteps = 0, 0, 0
stage_hist = np.zeros(5, dtype=int)
for k in range(N):
    env.reset(seed=SEED + k)
    planner = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False, base_pose=u.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False,
        joint_vel_limits=0.5, joint_acc_limits=0.5)
    _rp = np.asarray(u.agent.robot.pose.raw_pose.cpu()).reshape(-1)
    B = np.linalg.inv(_M(_rp[:3], _rp[3:7]))
    succ = False
    stage, hold = 0, 0
    last_a = np.zeros(7)
    # Freeze the pre-grasp side once per episode: recomputing the sign from the
    # live tcp makes it flip as the hand approaches, which teleports the target
    # 2*0.10 m and stalls the whole approach (measured: |pre| 0.17 -> 0.13 ->
    # 0.18 with 49% IK failures).
    _tp0 = np.asarray(u.agent.tcp.pose.raw_pose.cpu()).reshape(-1)[:3].copy()
    for t in range(HOR):
        nsteps += 1
        dbg_dp = 0.0
        o = np.asarray(build_unified_obs_batch(u.get_obs(), layout).cpu())
        o = o.reshape(-1).astype(np.float32)
        x = torch.as_tensor(o)[None]
        qnow = np.asarray(u.agent.robot.get_qpos().cpu()).reshape(-1)
        _tp = np.asarray(u.agent.tcp.pose.raw_pose.cpu()).reshape(-1)
        tcp, tcpq = _tp[:3], _tp[3:7]

        p_ch, q_ch = o[25:28], _n(o[28:32])
        Rch = _R(q_ch)
        p_grasp = p_ch + Rch @ REL
        q_grasp = _n(t3d.quaternions.qmult(q_ch, REL_QUAT))
        # Approach from straight above.  Offsetting along the socket axis put
        # the pre-grasp point behind the charger or below the table, so the
        # hand jammed there (measured: |pre| frozen at 0.10 m for 400 steps).
        pre = p_grasp + np.array([0.0, 0.0, 0.08])

        # ---------------- stage machine ----------------
        if stage == 0:
            tgt, step, qtar, grip = pre, 0.008, tcpq, 1.0
            if np.linalg.norm(tcp - pre) < 0.015:
                stage = 1
        elif stage == 1:
            # Rotate and translate together towards the grasp pose.  Keeping the
            # target at the live tcp (rotate-in-place) created a positive
            # feedback loop: every wrist move displaced the tool a little, the
            # target followed it, and 62 steps accumulated 25 cm of drift
            # (measured z 0.036 -> 0.427).  A fixed p_grasp pulls it back.
            tgt, step, qtar, grip = p_grasp, 0.005, q_grasp, 1.0
            if (np.linalg.norm(tcp - p_grasp) < 0.008
                    and _ang(tcpq, q_grasp) < 0.10):
                stage, hold = 3, 0
        elif stage == 2:
            tgt, step, qtar, grip = p_grasp, 0.004, q_grasp, 1.0
            if np.linalg.norm(tcp - p_grasp) < 0.005:
                stage, hold = 3, 0
        elif stage == 3:
            tgt, step, qtar, grip = tcp, 0.0, tcpq, -1.0
            hold += 1
            if hold >= 8:
                stage = 4
        else:
            # Geometric insertion: drive the held charger straight at the
            # receptacle with the wrist locked.  pi_cart was measured pushing
            # horizontally near z=0.022 while the socket sits at z=0.100, so it
            # could never line up; amplifying it (x2.5) inflated |dp| to
            # 12-24 mm per step and knocked the plug out of the jaws.
            p_rc = o[32:35]
            d = p_rc - p_ch
            nd = float(np.linalg.norm(d))
            step_t = float(os.environ.get("PROBE_INS_STEP", "0.004"))
            tgt = p_ch + (d * (step_t / nd) if nd > step_t else d)
            dbg_dp = nd
            step = 0.0                 # target already carries the increment
            qtar = tcpq
            # Keep the jaws shut through insertion: letting pi drive the gripper
            # relaxed it to -0.14 and the charger slipped.
            grip = -1.0
        stage_hist[stage] += 1


        # ---------------- bounded increments ----------------
        if step > 0.0:
            d = tgt - tcp
            nd = float(np.linalg.norm(d))
            if nd > step:
                d = d * (step / nd)
            tgt = tcp + d
        if _ang(tcpq, qtar) > MAXROT:
            qtar = _slerp(tcpq, qtar, MAXROT / _ang(tcpq, qtar))

        # Shrink the increment until IK accepts it.  Blindly coasting on the
        # previous action made the arm wander (measured: |tcp-grasp| growing
        # from 0.10 m to 0.45 m at z=0.29 m), so a failed query now simply
        # tries a smaller move and otherwise holds still.
        a_arm = np.zeros(7)
        for shrink in (1.0, 0.4):
            _t = tcp + (tgt - tcp) * shrink
            _a = _ang(tcpq, qtar)
            _q = qtar if _a * shrink <= MAXROT else _slerp(
                tcpq, qtar, MAXROT * shrink / _a)
            # mplib was initialised with set_base_pose(robot.pose), so its FK
            # already lives in the world frame and IK wants world poses.
            # Feeding base-frame poses added the -0.615 m x offset, which made
            # the arm reach for an unreachable point and park at z=0.292 with
            # |tcp-grasp| = 0.29 m (ik_fail was still 0 - IK "succeeded").
            _p = (B @ np.r_[_t, 1.0])[:3]
            _qb = t3d.quaternions.mat2quat(B[:3, :3] @ _R(_q))
            st, sol = planner.planner.IK(np.r_[_p, _qb], qnow,
                                         n_init_qpos=4, threshold=1e-3)
            if st == "Success" and len(sol):
                # Pick the branch closest to the current joints.  Taking sol[0]
                # handed back far elbow-flip solutions that satisfy the pose but
                # teleport the tool 0.28 m in 40 steps (measured z 0.042->0.324).
                cand = min((np.asarray(s)[:7] for s in sol),
                           key=lambda s: float(np.linalg.norm(s - qnow[:7])))
                _ig = float(os.environ.get("PROBE_S4", "1.0")) if stage == 4 else IGAIN
                a_arm = np.clip((cand - qnow[:7]) / 0.1 * _ig, -1.0, 1.0)
                break
        else:
            ikfail += 1

        obs, rew, term, trunc, info = env.step(
            np.r_[a_arm, grip].astype(np.float32))
        if DBG and t % 40 == 0:
            print("   t%3d st%d tcp %s |pre| %.3f |grasp| %.3f ang %.2f g%+.2f"
                  % (t, stage, np.round(tcp, 3),
                     float(np.linalg.norm(tcp - pre)),
                     float(np.linalg.norm(tcp - p_grasp)),
                     _ang(tcpq, q_grasp), grip), flush=True)
            print("        |dp| %.4f |a| %.3f |a0| %.3f" % (dbg_dp,
                  float(np.linalg.norm(a_arm)), float(np.abs(a_arm).max())), flush=True)
        s = info.get("success") if isinstance(info, dict) else None
        if s is not None and bool(np.asarray(s).reshape(-1)[0]):
            succ = True
            break
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            break
    ok += int(succ)
    if DBG:
        print("ep%03d succ=%d stage_end=%d t=%d" % (k, int(succ), stage, t), flush=True)
print("SR %.0f%% (%d/%d)  ik_fail=%d  steps=%d  stage_hist=%s"
      % (100.0 * ok / N, ok, N, ikfail, nsteps, stage_hist.tolist()))
