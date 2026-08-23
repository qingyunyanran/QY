"""
Diagnostic: Can PickCube learn from scratch with 54-dim zero-padded obs?
If yes -> zero-padding is fine, transfer strategy is the issue (v8)
If no  -> zero-padding fundamentally breaks PickCube (need new approach)
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_conda = r"E:\My_programs\anaconda\envs\diffcl10\Library\bin"
if _conda not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda + os.pathsep + os.environ.get("PATH", "")

import gymnasium as gym
import mani_skill.envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy

class FixedDummyVecEnv(DummyVecEnv):
    def step_wait(self):
        results = super().step_wait()
        obs, rewards, dones, infos = results
        for info in infos:
            if "terminal_observation" in info:
                tobs = info["terminal_observation"]
                if isinstance(tobs, torch.Tensor):
                    tobs = tobs.cpu().numpy()
                info["terminal_observation"] = np.asarray(tobs, dtype=np.float32).flatten()
        return obs, rewards, dones, infos

def _patched_obs_to_tensor(self, observation):
    if not isinstance(observation, np.ndarray):
        observation = np.array(observation)
    if observation.ndim == 1:
        observation = observation.reshape(1, -1)
    return torch.as_tensor(observation, dtype=torch.float32).to(self.device), True

BasePolicy.obs_to_tensor = _patched_obs_to_tensor

# ═══════════════════════════════════════════════════════════════════
UNIFIED_OBS_DIM = 54
CONTROL_MODE = "pd_joint_delta_pos"

class UnifiedObsWrapper(gym.Wrapper):
    def __init__(self, env, raw_obs_dim):
        super().__init__(env)
        self.raw_obs_dim = raw_obs_dim
        self.pad_dim = UNIFIED_OBS_DIM - raw_obs_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(UNIFIED_OBS_DIM,), dtype=np.float32
        )
    def _pad(self, obs):
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        obs = np.array(obs, dtype=np.float32).flatten()
        if self.pad_dim > 0:
            obs = np.concatenate([obs, np.zeros(self.pad_dim, dtype=np.float32)])
        return obs
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pad(obs), info
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._pad(obs)
        if isinstance(reward, torch.Tensor): reward = reward.item()
        if isinstance(terminated, torch.Tensor): terminated = terminated.item()
        if isinstance(truncated, torch.Tensor): truncated = truncated.item()
        return obs, float(reward), bool(terminated), bool(truncated), info

class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "success" in info:
            s = info["success"]
            if hasattr(s, "item"): s = s.item()
            if hasattr(s, "__len__") and not isinstance(s, (str, bytes)):
                s = s[0]
                if hasattr(s, "item"): s = s.item()
            info["is_success"] = bool(s)
        return obs, reward, terminated, truncated, info

def make_env(task_id, idx=0):
    def _init():
        env = gym.make(task_id, obs_mode="state", control_mode=CONTROL_MODE)
        env = UnifiedObsWrapper(env, 42)  # PickCube raw=42, pad to 54
        env = SuccessInfoWrapper(env)
        env = Monitor(env)
        return env
    return _init

def make_eval_env():
    env = gym.make("PickCube-v1", obs_mode="state", control_mode=CONTROL_MODE)
    env = UnifiedObsWrapper(env, 42)
    env = SuccessInfoWrapper(env)
    return env

@torch.no_grad()
def eval_pickcube(model, n_episodes=30):
    env = make_eval_env()
    obs, _ = env.reset()
    successes = []
    total_rewards = []
    ep_reward = 0.0
    while len(successes) < n_episodes:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = model.predict(obs_tensor, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        if terminated or truncated:
            total_rewards.append(ep_reward)
            s = info.get("is_success", False)
            if not s and "success" in info:
                sv = info["success"]
                if hasattr(sv, "item"): sv = sv.item()
                if hasattr(sv, "__len__"):
                    sv = sv[0]
                    if hasattr(sv, "item"): sv = sv.item()
                s = bool(sv)
            successes.append(s)
            ep_reward = 0.0
            obs, _ = env.reset()
    env.close()
    return float(np.mean(successes)), float(np.mean(total_rewards))

if __name__ == "__main__":
    print("=" * 70)
    print("  Diagnostic: PickCube from scratch with 54-dim zero-padded obs")
    print("  Goal: verify zero-padding doesn't break PickCube learning")
    print("=" * 70)

    N_ENVS = 4
    TOTAL_STEPS = 1_000_000
    EVAL_FREQ = 100_000

    train_envs = FixedDummyVecEnv([make_env("PickCube-v1", i) for i in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        train_envs,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
            activation_fn=nn.Tanh,
        ),
        learning_rate=3e-4,
        ent_coef=0.0,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.8,
        verbose=1,
    )

    total_done = 0
    print(f"\n  Training {TOTAL_STEPS} steps, eval every {EVAL_FREQ}...")
    while total_done < TOTAL_STEPS:
        steps = min(EVAL_FREQ, TOTAL_STEPS - total_done)
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        total_done += steps

        sr, r = eval_pickcube(model)
        pct = total_done / TOTAL_STEPS * 100
        print(f"  [{total_done:>7d}/{TOTAL_STEPS}] ({pct:.0f}%) PickCube SR={sr*100:.1f}% avg_R={r:.2f}")

        if sr > 0.5:
            print(f"\n  SUCCESS: PickCube SR={sr*100:.1f}% at {total_done} steps!")
            print(f"  Zero-padding is FINE. Transfer strategy is the bottleneck.")
            break

    print(f"\n  Final: PickCube SR={sr*100:.1f}% avg_R={r:.2f}")
    train_envs.close()
    print("\nDONE")
