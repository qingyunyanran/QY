"""
评估 PokeCube-v1 官方 PPO checkpoint
用法: python eval_poke_ckpt.py [--num-episodes 50] [--control-mode pd_ee_delta_pos]
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
parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pos",
                    choices=["pd_ee_delta_pos", "pd_ee_delta_pose", "pd_joint_delta_pos"])
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

# ─── 1. 加载 checkpoint ───────────────────────────────────────────
ckpt_dir = os.path.join(os.path.expanduser("~"), ".maniskill", "demos", "PokeCube-v1", "rl")
ckpt_path = os.path.join(ckpt_dir, f"ppo_{args.control_mode}_ckpt.pt")
print(f"加载 checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

# ─── 2. 推断维度 ──────────────────────────────────────────────────
obs_dim = ckpt["actor_mean.0.weight"].shape[1]
act_dim = ckpt["actor_mean.6.weight"].shape[0]
print(f"obs_dim={obs_dim}, action_dim={act_dim}")

# ─── 3. 构建 actor ────────────────────────────────────────────────
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

actor = ActorMean(obs_dim, act_dim)
actor_state = {}
for k, v in ckpt.items():
    if k.startswith("actor_mean."):
        actor_state[k.replace("actor_mean.", "net.")] = v
actor.load_state_dict(actor_state)
actor.eval()
print("Actor 网络加载完成")

# ─── 4. 创建环境 ──────────────────────────────────────────────────
render_mode = "human" if args.render else None
env = gym.make(
    "PokeCube-v1",
    control_mode=args.control_mode,
    obs_mode="state",
    render_mode=render_mode,
)
print(f"环境: PokeCube-v1, 控制: {args.control_mode}")

# ─── 5. 评估 ──────────────────────────────────────────────────────
successes = []
total_rewards = []

for ep in range(args.num_episodes):
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

    # 处理 success：可能是 tensor / numpy / bool
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
        print(f"  Episode {ep+1:3d}/{args.num_episodes}: "
              f"success={is_success}, reward={ep_reward:.2f}, "
              f"running_sr={np.mean(successes)*100:.1f}%")

env.close()

# ─── 6. 汇总 ──────────────────────────────────────────────────────
sr = np.mean(successes) * 100
avg_r = np.mean(total_rewards)
print(f"\n{'='*50}")
print(f"评估完成: {args.num_episodes} 个 episode")
print(f"成功率: {sr:.1f}% ({sum(successes)}/{len(successes)})")
print(f"平均奖励: {avg_r:.2f}")
print(f"控制模式: {args.control_mode}")
print(f"{'='*50}")
