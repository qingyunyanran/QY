"""Compare mplib's FK frame against env's u.agent.tcp, both in the robot base
frame, to quantify the rigid offset between the two end-effector conventions."""
import os
import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import transforms3d as t3d

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)

TASK = "PlugCharger-v1"


def _M(p, q):
    m = np.eye(4)
    m[:3, :3] = t3d.quaternions.quat2mat(q)
    m[:3, 3] = p
    return m


env = gym.make(TASK, num_envs=1, obs_mode="state_dict",
               control_mode="pd_joint_delta_pos", sim_backend="cpu",
               render_backend="none", max_episode_steps=400)
env.reset(seed=0)
u = env.unwrapped

planner = PandaArmMotionPlanningSolver(
    env, debug=False, vis=False, base_pose=u.agent.robot.pose,
    visualize_target_grasp_pose=False, print_env_info=False,
    joint_vel_limits=0.5, joint_acc_limits=0.5)

rp = np.asarray(u.agent.robot.pose.raw_pose.cpu()).reshape(-1)
B = np.linalg.inv(_M(rp[:3], rp[3:7]))
print("base pose", rp)

for step in range(3):
    q = np.asarray(u.agent.robot.get_qpos().cpu()).reshape(-1).astype(np.float64)
    tp = np.asarray(u.agent.tcp.pose.raw_pose.cpu()).reshape(-1)
    T_env_b = B @ _M(tp[:3], tp[3:7])

    pl = planner.planner
    if step == 0:
        print("env tcp name:", getattr(u.agent.tcp, "name", "?"))
        print("mplib user_link_names:", pl.user_link_names)
        print("mplib move_group_link_id:", pl.move_group_link_id)
        print("pl.p fk attrs:", [m for m in dir(pl.p)
                                 if "fk" in m.lower() or "link" in m.lower()])
    ok, fk = pl.p.FK(q, pl.move_group_link_id)
    fk = np.asarray(fk).reshape(-1)
    T_fk_b = _M(fk[:3], fk[3:7])

    off = np.linalg.inv(T_fk_b) @ T_env_b
    print(f"--- step {step}  qpos_dim={q.shape[0]} fk_ok={ok}")
    print("  env tcp (base) p", np.round(T_env_b[:3, 3], 4),
          "q", np.round(t3d.quaternions.mat2quat(T_env_b[:3, :3]), 4))
    print("  mplib fk (base) p", np.round(T_fk_b[:3, 3], 4),
          "q", np.round(t3d.quaternions.mat2quat(T_fk_b[:3, :3]), 4))
    print("  dp = |env - fk|", round(float(np.linalg.norm(
        T_env_b[:3, 3] - T_fk_b[:3, 3])), 5))
    print("  offset in FK frame  t", np.round(off[:3, 3], 4),
          "  rot", np.round(t3d.quaternions.mat2quat(off[:3, :3]), 4))

    # nudge one joint so the comparison is not a single-config coincidence
    a = np.zeros(7)
    a[0] = 0.2 * (step + 1)
    env.step(np.r_[a, 1.0].astype(np.float32))

env.close()
