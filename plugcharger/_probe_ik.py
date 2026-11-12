"""Verify whether mplib's IK solution actually reaches the requested pose in the
env: solve for (p_grasp, q_grasp), push the joints back into the simulator and
read u.agent.tcp.  If they disagree, the two models differ; if they agree, the
problem is downstream in the action path."""
import os
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import transforms3d as t3d

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from unified_obs import SLOT_LAYOUT, build_unified_obs_batch

TASK = "PlugCharger-v1"
REL = np.array([-0.01828, 0.00016, 0.00557])
layout = SLOT_LAYOUT[TASK]


def _R(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _M(p, q):
    m = np.eye(4)
    m[:3, :3] = _R(q)
    m[:3, 3] = p
    return m


env = gym.make(TASK, num_envs=1, obs_mode="state_dict",
               control_mode="pd_joint_delta_pos", sim_backend="cpu",
               render_backend="none", max_episode_steps=400)
env.reset(seed=80000)
u = env.unwrapped
planner = PandaArmMotionPlanningSolver(
    env, debug=False, vis=False, base_pose=u.agent.robot.pose,
    visualize_target_grasp_pose=False, print_env_info=False,
    joint_vel_limits=0.5, joint_acc_limits=0.5)

rp = np.asarray(u.agent.robot.pose.raw_pose.cpu()).reshape(-1)
B = np.linalg.inv(_M(rp[:3], rp[3:7]))

o = np.asarray(build_unified_obs_batch(u.get_obs(), layout).cpu())
o = o.reshape(-1).astype(np.float32)
tp = np.asarray(u.agent.tcp.pose.raw_pose.cpu()).reshape(-1)
print("tcp world ", np.round(tp[:3], 4), "q", np.round(tp[3:7], 3))
print("obs[18:25] tcp   ", np.round(o[18:25], 4))
print("obs[25:32] charg ", np.round(o[25:32], 4))
print("obs[32:39] recept", np.round(o[32:39], 4))
print("obs[39:46] goal  ", np.round(o[39:46], 4))

p_ch, q_ch = o[25:28], o[28:32]
Rch = _R(q_ch)
p_grasp = p_ch + Rch @ REL
q_grasp = t3d.quaternions.mat2quat(Rch)
print("p_grasp world", np.round(p_grasp, 4),
      " base", np.round((B @ np.r_[p_grasp, 1.0])[:3], 4))

qnow = np.asarray(u.agent.robot.get_qpos().cpu()).reshape(-1).astype(np.float64)
p_b = (B @ np.r_[p_grasp, 1.0])[:3]
q_b = t3d.quaternions.mat2quat(B[:3, :3] @ _R(q_grasp))
st, sol = planner.planner.IK(np.r_[p_b, q_b], qnow, n_init_qpos=12,
                             threshold=8e-3)
print("IK status", st, "n_sol", len(sol))
if len(sol):
    cand = min((np.asarray(s)[:7] for s in sol),
               key=lambda s: float(np.linalg.norm(s - qnow[:7])))
    print("picked joints", np.round(cand, 4),
          "delta", round(float(np.linalg.norm(cand - qnow[:7])), 4))
    u.agent.robot.set_qpos(torch.as_tensor(np.r_[cand, qnow[7:9]],
                                           dtype=torch.float32)[None])
    u.scene.step()
    tp2 = np.asarray(u.agent.tcp.pose.raw_pose.cpu()).reshape(-1)
    print("tcp after set_qpos", np.round(tp2[:3], 4), "q", np.round(tp2[3:7], 3))
    print("|tcp - p_grasp| =", round(float(np.linalg.norm(tp2[:3] - p_grasp)), 5))
    print("rot err =", round(2 * np.arccos(min(1.0, abs(float(
        np.dot(tp2[3:7] / np.linalg.norm(tp2[3:7]),
               q_grasp / np.linalg.norm(q_grasp)))))), 4), "rad")
env.close()
