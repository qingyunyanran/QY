"""
评估 PullCube-v1 官方 PPO checkpoint（自动测试全部3个控制器）
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PATH"] = (
    r"E:\My_programs\anaconda\envs\diffcl10\Library\bin"
    + ";" + r"E:\My_programs\anaconda\envs\diffcl10\Library\lib"
    + ";" + os.environ.get("PATH", "")
)

import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs

parser = argparse.ArgumentParser()
parser.add_argument("--num-episodes", type=int, default=50)
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

TASK = "PullCube-v1"
CONTROL_MODES = ["pd_ee_delta_pos", "pd_ee_delta_pose", "pd_joint_delta_pos"]

ckpt_dir = os.path.join(os.path.expanduser("~"), ".maniskill", "demos", TASK, "rl")

class ActorMean(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, act_dim),
        )
    def forward(self, obs):
        return self.net(obs)

def evaluate(control_mode, num_episodes):
    ckpt_path = os.path.join(ckpt_dir, f"ppo_{control_mode}_ckpt.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [跳过] {ckpt_path} 不存在")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    obs_dim = ckpt["actor_mean.0.weight"].shape[1]
    act_dim = ckpt["actor_mean.6.weight"].shape[0]

    actor = ActorMean(obs_dim, act_dim)
    actor_state = {}
    for k, v in ckpt.items():
        if k.startswith("actor_mean."):
            actor_state[k.replace("actor_mean.", "net.")] = v
    actor.load_state_dict(actor_state)
    actor.eval()

    render_mode = "human" if args.render else None
    env = gym.make(TASK, control_mode=control_mode, obs_mode="state", render_mode=render_mode)

    successes = []
    total_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            with torch.no_grad():
                if isinstance(obs, torch.Tensor):
                    obs_t = obs.clone().detach().float().unsqueeze(0)
                else:
                    obs_t = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0)
                action = actor(obs_t).squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)

        raw_success = info.get("success", False)
        if isinstance(raw_success, torch.Tensor):
            is_success = bool(raw_success.item())
        elif isinstance(raw_success, np.ndarray):
            is_success = bool(raw_success.item())
        else:
            is_success = bool(raw_success)

        successes.append(is_success)
        total_rewards.append(ep_reward)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    Ep {ep+1:3d}/{num_episodes}: success={is_success}, "
                  f"reward={ep_reward:.2f}, sr={np.mean(successes)*100:.1f}%")

    env.close()
    sr = np.mean(successes) * 100
    avg_r = np.mean(total_rewards)
    return sr, avg_r, sum(successes), len(successes)

# ─── 主流程 ───────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"  {TASK} 官方 PPO Checkpoint 评估")
print(f"  Episodes per mode: {args.num_episodes}")
print(f"{'='*60}\n")

results = {}
for ctrl in CONTROL_MODES:
    print(f"\n▶ 控制器: {ctrl}")
    print(f"  obs/action dim: 从checkpoint推断")
    result = evaluate(ctrl, args.num_episodes)
    if result:
        sr, avg_r, n_ok, n_total = result
        results[ctrl] = (sr, avg_r, n_ok, n_total)
        print(f"  → 成功率: {sr:.1f}% ({n_ok}/{n_total}), 平均奖励: {avg_r:.2f}")
    print()

print(f"\n{'='*60}")
print(f"  {TASK} 汇总")
print(f"{'='*60}")
for ctrl, (sr, avg_r, n_ok, n_total) in results.items():
    marker = " ★" if sr == max(r[0] for r in results.values()) else ""
    print(f"  {ctrl:25s}  SR={sr:5.1f}%  avg_R={avg_r:6.2f}{marker}")
print()
