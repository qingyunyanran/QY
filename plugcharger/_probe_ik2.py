"""Calibrate one small IK step: ask for a 2 cm downward move, push the returned
joints into the simulator and compare where the tool really ends up.  Tests the
base-frame and world-frame conventions side by side."""
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import transforms3d as t3d

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

TASK = "PlugCharger-v1"


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

q0 = np.asarray(u.agent.robot.get_qpos().cpu()).reshape(-1).astype(np.float64)
tp0 = np.asarray(u.agent.tcp.pose.raw_pose.cpu()).reshape(-1)
print("tcp world", np.round(tp0[:3], 4))
print("tcp base ", np.round((B @ np.r_[tp0[:3], 1.0])[:3], 4))

for name in ("base", "world"):
    for dz in (-0.02, -0.05):
        p_w = tp0[:3].copy()
        p_w[2] += dz
        p_b = (B @ np.r_[p_w, 1.0])[:3]
        goal = p_b if name == "base" else p_w
        st, sol = planner.planner.IK(np.r_[goal, tp0[3:7]], q0,
                                     n_init_qpos=12, threshold=8e-3)
        if st != "Success" or not len(sol):
            print(f"[{name}] dz={dz} -> IK FAILED ({st})")
            continue
        cand = min((np.asarray(s)[:7] for s in sol),
                   key=lambda s: float(np.linalg.norm(s - q0[:7])))
        env.reset(seed=80000)
        u.agent.robot.set_qpos(torch.as_tensor(np.r_[cand, q0[7:9]],
                                              dtype=torch.float32)[None])
        u.scene.step()
        t1 = np.asarray(u.agent.tcp.pose.raw_pose.cpu()).reshape(-1)
        print(f"[{name}] dz={dz} IK ok  goal_world={np.round(p_w, 4)}")
        print(f"        got world={np.round(t1[:3], 4)} "
              f"err={np.linalg.norm(t1[:3] - p_w):.4f} "
              f"djoint={np.linalg.norm(cand - q0[:7]):.3f}")
env.close()
