"""
ManiSkill 环境适配层
替换ContinualWorld中的metaworld_compat.py
提供与MetaWorld类似的接口，封装ManiSkill的5个任务
"""
import gymnasium as gym
import mani_skill.envs
import numpy as np
from typing import List, Any, Dict


# 支持的任务配置
TASK_CONFIGS = {
    "PickCube-v1": {
        "control_mode": "pd_ee_delta_pose",
        "max_episode_steps": 300,
        "obs_mode": "state",
        "reward_mode": "dense",
    },
    "PushCube-v1": {
        "control_mode": "pd_joint_delta_pos",
        "max_episode_steps": 300,
        "obs_mode": "state",
        "reward_mode": "dense",
    },
    "StackCube-v1": {
        "control_mode": "pd_joint_delta_pos",
        "max_episode_steps": 400,
        "obs_mode": "state",
        "reward_mode": "dense",
    },
    "LiftCube-v1": {
        "control_mode": "pd_joint_delta_pos",
        "max_episode_steps": 300,
        "obs_mode": "state",
        "reward_mode": "dense",
    },
    "PegInsertionSide-v1": {
        "control_mode": "pd_joint_delta_pos",
        "max_episode_steps": 400,
        "obs_mode": "state",
        "reward_mode": "dense",
    },
}

# 默认任务序列（类似CW10）
TASK_SEQS = {
    "MS5": ["PushCube-v1", "PickCube-v1", "LiftCube-v1", "StackCube-v1", "PegInsertionSide-v1"],
    "MS3": ["PushCube-v1", "PickCube-v1", "LiftCube-v1"],
    "MS2": ["PushCube-v1", "PickCube-v1"],
}


class ManiSkillTask:
    """模拟MetaWorld的Task对象，用于环境初始化"""
    def __init__(self, env_name, task_id=0):
        self.env_name = env_name
        self.task_id = task_id


class ManiSkillEnvWrapper(gym.Wrapper):
    """
    包装ManiSkill环境，提供与MetaWorld类似的接口
    主要适配：
    - 观测空间（统一为向量）
    - 成功信号（info['success']）
    - 任务设置接口（set_task）
    """
    def __init__(self, task_name, **kwargs):
        config = TASK_CONFIGS[task_name].copy()
        config.update(kwargs)
        
        env = gym.make(
            task_name,
            num_envs=1,
            obs_mode=config.get("obs_mode", "state"),
            control_mode=config.get("control_mode", "pd_joint_delta_pos"),
            max_episode_steps=config.get("max_episode_steps", 300),
            reward_mode=config.get("reward_mode", "dense"),
        )
        super().__init__(env)
        
        self.task_name = task_name
        self._current_task = ManiSkillTask(task_name)
        
        # ManiSkill的观测是(1, dim)的tensor，需要转成numpy向量
        # 我们在step和reset中处理
        
        # 记录原始观测维度
        obs, _ = env.reset()
        self._original_obs_dim = obs.shape[1] if len(obs.shape) > 1 else obs.shape[0]
        self._has_batch_dim = len(obs.shape) > 1 and obs.shape[0] == 1
        
        # 重定义观测空间（去掉batch维）
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self._original_obs_dim,),
            dtype=np.float32
        )
        
        # 动作空间也需要适配（ManiSkill的动作空间可能是gym.spaces.Box）
        self.action_space = env.action_space
        if hasattr(self.action_space, 'shape') and len(self.action_space.shape) > 1:
            self.action_space = gym.spaces.Box(
                low=env.action_space.low.flatten(),
                high=env.action_space.high.flatten(),
            )
    
    def _process_obs(self, obs):
        """将ManiSkill的观测转成numpy向量"""
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if self._has_batch_dim:
            obs = obs[0]  # 去掉batch维
        return obs.astype(np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._process_obs(obs)
        # ManiSkill的info可能也有嵌套结构，简化一下
        success = info.get("success", False)
        if isinstance(success, (list, np.ndarray, torch.Tensor)):
            success = bool(success[0]) if len(success) > 0 else False
        simple_info = {"success": success}
        return obs, simple_info
    
    def step(self, action):
        # ManiSkill的动作可能需要batch维
        if self._has_batch_dim and len(action.shape) == 1:
            action = action[np.newaxis, :]
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        obs = self._process_obs(obs)
        
        # 处理reward
        if isinstance(reward, (list, np.ndarray, torch.Tensor)):
            reward = float(reward[0]) if len(reward) > 0 else float(reward)
        
        # 处理terminated/truncated
        if isinstance(terminated, (list, np.ndarray, torch.Tensor)):
            terminated = bool(terminated[0])
        if isinstance(truncated, (list, np.ndarray, torch.Tensor)):
            truncated = bool(truncated[0])
        
        # 处理info中的success
        success = info.get("success", False)
        if isinstance(success, (list, np.ndarray, torch.Tensor)):
            success = bool(success[0]) if len(success) > 0 else False
        
        simple_info = {"success": success}
        
        return obs, reward, terminated, truncated, simple_info
    
    def set_task(self, task):
        """设置任务（ManiSkill单任务环境不需要切换，保留接口兼容性）"""
        self._current_task = task
    
    def get_task(self):
        return self._current_task


class MT50Simulator:
    """模拟MetaWorld的MT50对象，提供train_classes和train_tasks"""
    def __init__(self):
        self.train_classes = {}
        self.train_tasks = []
        
        for task_name in TASK_CONFIGS.keys():
            # 创建一个工厂函数（闭包）
            def make_env_factory(name=task_name):
                def _factory():
                    return ManiSkillEnvWrapper(name)
                return _factory
            
            self.train_classes[task_name] = make_env_factory()
            self.train_tasks.append(ManiSkillTask(task_name))
    
    def __getitem__(self, idx):
        return self.train_tasks[idx]


# 全局单例
MS_ENVS = MT50Simulator()


def get_task_name(name_or_number):
    """获取任务名称"""
    try:
        index = int(name_or_number)
        return list(TASK_CONFIGS.keys())[index]
    except (ValueError, IndexError):
        return name_or_number


def set_simple_goal(env, name):
    """设置目标（ManiSkill单任务不需要，保留接口）"""
    pass


def get_subtasks(name):
    """获取子任务列表（ManiSkill单任务只有一个子任务）"""
    return [task for task in MS_ENVS.train_tasks if task.env_name == name]


def get_single_env(task_name, **kwargs):
    """
    创建单个ManiSkill环境
    参数与ContinualWorld的get_single_env类似
    """
    task_name = get_task_name(task_name)
    env = ManiSkillEnvWrapper(task_name)
    return env


# 导入torch（延迟导入，避免循环依赖）
import torch
