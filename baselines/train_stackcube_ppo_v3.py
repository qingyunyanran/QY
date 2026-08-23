"""
StackCube PPO 训练 - 使用与 PushCube 成功的相同配置
obs_mode='state_dict' + control_mode='pd_ee_delta_pose'
"""
import os
import sys
import gymnasium as gym
import torch
import numpy as np
import pickle
import mani_skill.envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback


class FlattenStateDictWrapper(gym.ObservationWrapper):
    """将 state_dict 观测展平为一维向量"""
    def __init__(self, env):
        super().__init__(env)
        # 先做一次 reset 获取实际观测结构
        obs, _ = env.reset()
        flat_obs = self._flatten(obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32
        )
    
    def _flatten(self, obs):
        """递归展平观测"""
        flat_obs = []
        if isinstance(obs, dict):
            for key in sorted(obs.keys()):
                flat_obs.append(self._flatten(obs[key]))
        elif isinstance(obs, torch.Tensor):
            flat_obs.append(obs.cpu().numpy().flatten())
        elif isinstance(obs, np.ndarray):
            flat_obs.append(obs.flatten())
        else:
            flat_obs.append(np.array([obs], dtype=np.float32))
        
        if len(flat_obs) == 1:
            return flat_obs[0]
        return np.concatenate(flat_obs).astype(np.float32)
    
    def observation(self, obs):
        return self._flatten(obs)


def make_env(env_id, seed=0):
    """创建环境（使用 PushCube 成功的配置）"""
    def _init():
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="state_dict",           # 使用 state_dict
            control_mode="pd_ee_delta_pose",  # 使用 pd_ee_delta_pose
            reward_mode="dense",
            render_mode="default",
            max_episode_steps=300,            # 增加最大步数
        )
        env = FlattenStateDictWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


def train_ppo_stackcube(total_timesteps=5_000_000, num_envs=8):
    """训练 StackCube PPO"""
    env_id = "StackCube-v1"
    save_dir = f"runs/{env_id}_ppo_v3"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"训练 {env_id} (PushCube 成功配置)")
    print(f"{'='*60}")
    print(f"num_envs={num_envs}, total_timesteps={total_timesteps:,}")
    print(f"save_dir={save_dir}")
    print(f"obs_mode=state_dict, control_mode=pd_ee_delta_pose")
    
    # 创建向量化环境
    envs = DummyVecEnv([make_env(env_id, seed=i) for i in range(num_envs)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 创建评估环境
    eval_envs = DummyVecEnv([make_env(env_id, seed=1000+i) for i in range(4)])
    eval_envs = VecNormalize(eval_envs, norm_obs=True, norm_reward=False, training=False, clip_obs=10.)
    
    # 创建模型（使用 PushCube 成功的超参数）
    model = PPO(
        'MlpPolicy',
        envs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=save_dir,
    )
    
    # 评估回调
    eval_freq = 5000
    eval_callback = EvalCallback(
        eval_envs,
        best_model_save_path=os.path.join(save_dir, 'best_model'),
        log_path=os.path.join(save_dir, 'eval_logs'),
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        warn=False,
    )
    
    print(f"\n开始训练...")
    print(f"  obs_space: {envs.observation_space}")
    print(f"  act_space: {envs.action_space}")
    print(f"  eval_freq: {eval_freq} steps")
    
    # 训练
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    
    # 保存
    final_path = os.path.join(save_dir, "final_model.zip")
    model.save(final_path)
    
    vn_path = os.path.join(save_dir, "vec_normalize.pkl")
    with open(vn_path, 'wb') as f:
        pickle.dump(envs, f)
    
    print(f"\n✓ 训练完成！")
    print(f"  最终模型: {final_path}")
    print(f"  VecNormalize: {vn_path}")
    
    envs.close()
    eval_envs.close()
    
    return model, save_dir


def evaluate(save_dir, num_episodes=50):
    """评估模型"""
    env_id = "StackCube-v1"
    
    print(f"\n{'='*60}")
    print(f"评估 {env_id} 成功率")
    print(f"{'='*60}")
    
    eval_envs = DummyVecEnv([make_env(env_id, seed=1000)])
    vec_norm_path = os.path.join(save_dir, "vec_normalize.pkl")
    eval_envs = VecNormalize.load(vec_norm_path, eval_envs)
    eval_envs.training = False
    eval_envs.norm_reward = False
    
    model = PPO.load(os.path.join(save_dir, "final_model.zip"), device="cpu")
    
    success_count = 0
    
    for ep in range(num_episodes):
        obs = eval_envs.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_envs.step(action)
            
            if dones[0]:
                info = infos[0] if isinstance(infos, list) else infos
                if info.get("success", False):
                    success_count += 1
                break
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes}, Success: {success_count}/{ep+1}")
    
    success_rate = success_count / num_episodes * 100
    print(f"\n最终成功率: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"{'='*60}")
    eval_envs.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate("runs/StackCube-v1_ppo_v3")
    else:
        model, save_dir = train_ppo_stackcube(args.total_timesteps, args.num_envs)
        evaluate(save_dir)
