"""Test the REAL official PokeCube checkpoint."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_conda = r"E:\My_programs\anaconda\envs\diffcl10\Library\bin"
if _conda not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class OfficialAgent(nn.Module):
    def __init__(self, obs_dim=54, act_dim=8):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

# Test BOTH checkpoints
checkpoints = {
    "ckpt_991 (user training)": r"E:\munichi\rl_baselines\PokeCube-v1\ppo\ckpt_991.pt",
    "official pd_joint_delta_pos": os.path.expanduser(
        r"~\.maniskill\demos\PokeCube-v1\rl\ppo_pd_joint_delta_pos_ckpt.pt"),
}

for name, ckpt_path in checkpoints.items():
    print("=" * 60)
    print(f"Testing: {name}")
    print(f"Path: {ckpt_path}")
    
    if not os.path.isfile(ckpt_path):
        print("  FILE NOT FOUND!")
        continue
    
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  Keys: {list(raw.keys())}")
    print(f"  actor_logstd: {raw['actor_logstd'].flatten().tolist()}")
    print(f"  actor_mean.0.weight shape: {raw['actor_mean.0.weight'].shape}")
    
    # Check if obs_dim matches 54
    obs_dim = raw['actor_mean.0.weight'].shape[1]
    act_dim = raw['actor_mean.6.weight'].shape[0]
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    
    agent = OfficialAgent(obs_dim, act_dim)
    agent.load_state_dict(raw)
    agent.eval()
    
    env = gym.make("PokeCube-v1", obs_mode="state", control_mode="pd_joint_delta_pos",
                   max_episode_steps=200)
    
    successes = []
    rewards = []
    for ep in range(20):
        obs, info = env.reset(seed=ep*10)
        total_r = 0
        steps = 0
        while True:
            with torch.no_grad():
                action = agent.actor_mean(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward.item()
            steps += 1
            term = terminated.item() if hasattr(terminated, 'item') else terminated
            trunc = truncated.item() if hasattr(truncated, 'item') else truncated
            if term or trunc:
                s = info.get("success", None)
                if hasattr(s, 'item'): s = s.item()
                if hasattr(s, '__len__'):
                    s = s[0].item() if hasattr(s[0], 'item') else s[0]
                successes.append(bool(s))
                break
        rewards.append(total_r)
    
    print(f"\n  Results (20 episodes, 200 steps):")
    print(f"    SR = {np.mean(successes)*100:.1f}%")
    print(f"    avg_R = {np.mean(rewards):.2f}")
    env.close()

# Also test the SB3 model with VecNormalize
print("\n" + "=" * 60)
print("Testing SB3 PokeCube model (with VecNormalize)...")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle

sb3_path = r"E:\munichi\rl_baselines\PokeCube-v1\ppo_sb3\ppo_pokecube_final.zip"
vecnorm_path = r"E:\munichi\rl_baselines\PokeCube-v1\ppo_sb3\vec_normalize_pokecube.pkl"

if os.path.isfile(sb3_path):
    print(f"  SB3 model: {sb3_path}")
    print(f"  VecNormalize: {vecnorm_path}")
    
    # Load VecNormalize stats
    with open(vecnorm_path, 'rb') as f:
        vecnorm_stats = pickle.load(f)
    print(f"  VecNormalize obs_rms mean[:5]: {vecnorm_stats.obs_rms.mean[:5]}")
    print(f"  VecNormalize obs_rms var[:5]: {vecnorm_stats.obs_rms.var[:5]}")
    print(f"  VecNormalize ret_rms mean: {vecnorm_stats.ret_rms.mean}")
    print(f"  VecNormalize training: {vecnorm_stats.training}")
    print(f"  VecNormalize norm_obs: {vecnorm_stats.norm_obs}")
    print(f"  VecNormalize norm_reward: {vecnorm_stats.norm_reward}")
else:
    print("  SB3 model not found!")

print("\nDONE")
