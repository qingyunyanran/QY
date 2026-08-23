"""
PushCube-v1 PPO 训练 v2 - state_dict + flatten wrapper 匹配 CL 实验环境
"""
import os
import sys
import gymnasium as gym
from gymnasium import spaces
import mani_skill.envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import numpy as np
import pickle

# 导入与 CL 实验完全相同的 flatten_obs，保证顺序一致
sys.path.insert(0, os.path.dirname(__file__))
from train_bc_experts import flatten_obs

device = "cpu"
print(f"使用设备: {device}")

os.makedirs('./models', exist_ok=True)

NUM_ENVS = 4
TOTAL_STEPS = 5_000_000
SAVE_DIR = "E:\\munich\\rl_baselines\\push_cube_v2"
os.makedirs(SAVE_DIR, exist_ok=True)


class FlattenStateDictWrapper(gym.ObservationWrapper):
    """将 state_dict 嵌套 dict 展平为 1D Box，使用与 CL 实验相同的 flatten_obs"""
    def __init__(self, env):
        super().__init__(env)
        dummy_obs, _ = env.reset()
        flat = flatten_obs(dummy_obs)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, flat.shape, dtype=np.float32
        )
        print(f"[Wrapper] Obs flattened to dim: {flat.shape[0]}")

    def observation(self, obs):
        return flatten_obs(obs)


def make_env(seed=0):
    def _init():
        env = gym.make(
            'PushCube-v1',
            num_envs=1,
            control_mode='pd_ee_delta_pose',
            obs_mode='state_dict',
            reward_mode='dense',
            render_mode='default',
        )
        env = FlattenStateDictWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


# 创建向量化环境
env = DummyVecEnv([make_env(i) for i in range(NUM_ENVS)])

# VecNormalize
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 创建 PPO 模型
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device=device,
)

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=SAVE_DIR,
    name_prefix='ppo_pushcube',
)

print(f"\n开始训练 PushCube-v1...")
print(f"总步数: {TOTAL_STEPS:,}")
print(f"保存目录: {SAVE_DIR}\n")

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=checkpoint_callback,
)

# 保存最终模型和 VecNormalize
final_model_path = os.path.join(SAVE_DIR, 'push_cube_final.zip')
model.save(final_model_path)
env.save(os.path.join(SAVE_DIR, 'vec_normalize.pkl'))

print(f"\n✓ 训练完成！")
print(f"模型: {final_model_path}")
print(f"VecNormalize: {os.path.join(SAVE_DIR, 'vec_normalize.pkl')}")

# 验证：手动归一化，不依赖 VecNormalize wrapper
print("\n验证中 (50 episodes)...")
eval_env = FlattenStateDictWrapper(gym.make(
    'PushCube-v1', num_envs=1,
    control_mode='pd_ee_delta_pose',
    obs_mode='state_dict',
    reward_mode='dense',
))

# 加载归一化统计量
vn_path = os.path.join(SAVE_DIR, 'vec_normalize.pkl')
with open(vn_path, 'rb') as f:
    vn_data = pickle.load(f)

# VecNormalize 保存的是完整对象，提取 RMS 统计量
if hasattr(vn_data, 'obs_rms'):
    obs_rms = vn_data.obs_rms
    clip_obs = vn_data.clip_obs
else:
    # 兼容 dict 格式
    obs_rms = vn_data.get('obs_rms', vn_data)
    clip_obs = vn_data.get('clip_obs', 10.0) if isinstance(vn_data, dict) else 10.0

print(f"Loaded VecNormalize: obs_dim={obs_rms.mean.shape[0]}, clip_obs={clip_obs}")

obs, _ = eval_env.reset()
successes = 0
n_episodes = 50

for ep in range(n_episodes):
    done = False
    while not done:
        # 手动归一化
        flat = obs.copy()
        norm_obs = np.clip(
            (flat - obs_rms.mean) / np.sqrt(obs_rms.var + obs_rms.epsilon),
            -clip_obs, clip_obs
        )
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        if done and info.get('success', False):
            successes += 1
    obs, _ = eval_env.reset()
    if (ep + 1) % 10 == 0:
        print(f"  Ep {ep+1}/{n_episodes}, 成功率: {successes}/{ep+1}")

print(f"\n验证结果: {successes}/{n_episodes} = {successes/n_episodes*100:.1f}%")
