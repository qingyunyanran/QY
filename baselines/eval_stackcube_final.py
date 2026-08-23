"""Evaluate StackCube checkpoint success rate."""
import os, sys
sys.stdout.reconfigure(line_buffering=True)

_lib_bin = r"E:\My_programs\anaconda\envs\diffcl10\Library\bin"
_lib_lib = r"E:\My_programs\anaconda\envs\diffcl10\Library\lib"
os.environ["PATH"] = _lib_bin + ";" + _lib_lib + ";" + os.environ.get("PATH", "")

import gymnasium as gym
import mani_skill.envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import numpy as np

print("Loading checkpoint: 1000000 steps", flush=True)
model = PPO.load(r"E:\munichi\models\ppo_stackcube_v1_ppo_stackcube_v1_final", device="cpu")

NUM_EVAL_ENVS = 4
MAX_EPISODE_STEPS = 300
NUM_EPISODES = 20

def make_env(seed=0):
    def _init():
        env = gym.make(
            'StackCube-v1',
            num_envs=1,
            control_mode='pd_joint_delta_pos',
            reward_mode='dense',
            obs_mode='state',
            max_episode_steps=MAX_EPISODE_STEPS,
        )
        return SqueezeObsWrapper(env)
    return _init

class SqueezeObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        orig_obs = env.observation_space
        if isinstance(orig_obs, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=orig_obs.low[0], high=orig_obs.high[0], dtype=orig_obs.dtype
            )
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if hasattr(obs, 'cpu'): obs = obs.cpu().numpy()
        return obs[0], info
    def step(self, action):
        action_tensor = torch.tensor(action[np.newaxis, ...], dtype=torch.float32)
        obs, reward, terminated, truncated, info = self.env.step(action_tensor)
        if hasattr(obs, 'cpu'): obs = obs.cpu().numpy()
        if hasattr(reward, 'cpu'): reward = reward.cpu().numpy()
        if hasattr(terminated, 'cpu'): terminated = terminated.cpu().numpy()
        if hasattr(truncated, 'cpu'): truncated = truncated.cpu().numpy()
        return obs[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), info

print("Creating eval envs...", flush=True)
envs = [make_env(seed=i+100) for i in range(NUM_EVAL_ENVS)]
eval_env = DummyVecEnv(envs)

# Load VecNormalize stats
import glob
vec_norm_files = glob.glob(r"E:\munichi\models\vec_normalize_stackcube*")
if vec_norm_files:
    latest = sorted(vec_norm_files)[-1]
    print(f"Loading VecNormalize stats from {latest}", flush=True)
    eval_env = VecNormalize.load(latest, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
else:
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env.training = False

successes = 0
total_episodes = 0
episode_rewards = []

print(f"\nEvaluating {NUM_EPISODES} episodes...", flush=True)
obs = eval_env.reset()

# Track per-env episode info
episode_counts = [0] * NUM_EVAL_ENVS
episode_rewards_current = [0.0] * NUM_EVAL_ENVS
episode_success = [False] * NUM_EVAL_ENVS

while total_episodes < NUM_EPISODES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    
    for i in range(NUM_EVAL_ENVS):
        episode_rewards_current[i] += float(reward[i]) if hasattr(reward, '__len__') else float(reward)
        
        if done[i]:
            total_episodes += 1
            # Check if success from info
            info_item = info[i] if isinstance(info, list) else info
            success = False
            if isinstance(info_item, dict):
                success = info_item.get('success', False)
                if hasattr(success, '__len__'):
                    success = bool(success[0]) if len(success) > 0 else False
                success = bool(success)
            
            if success:
                successes += 1
            
            ep_rew = episode_rewards_current[i]
            episode_rewards.append(ep_rew)
            episode_counts[i] += 1
            episode_rewards_current[i] = 0.0
            
            if total_episodes % 5 == 0 or total_episodes == NUM_EPISODES:
                print(f"  Episode {total_episodes}/{NUM_EPISODES}: success={success}, reward={ep_rew:.3f}, running_success_rate={successes}/{total_episodes}={successes/total_episodes:.1%}", flush=True)
            
            if total_episodes >= NUM_EPISODES:
                break

print(f"\n=== EVALUATION RESULTS ===", flush=True)
print(f"Checkpoint: final", flush=True)
print(f"Episodes: {total_episodes}", flush=True)
print(f"Successes: {successes}/{total_episodes} = {successes/total_episodes:.1%}", flush=True)
print(f"Avg reward: {np.mean(episode_rewards):.3f} +/- {np.std(episode_rewards):.3f}", flush=True)

eval_env.close()
