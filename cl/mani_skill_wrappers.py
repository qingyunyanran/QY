"""
ManiSkill 版环境 Wrappers
基于ContinualWorld的wrappers.py，适配ManiSkill环境
"""
import random
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class SuccessCounter(gym.Wrapper):
    """统计成功率的Wrapper"""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.successes = []
        self.current_success = False

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("success", False):
            self.current_success = True
        if terminated or truncated:
            self.successes.append(self.current_success)
        return obs, reward, terminated, truncated, info

    def pop_successes(self) -> List[bool]:
        res = self.successes
        self.successes = []
        return res

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self.current_success = False
        return self.env.reset(**kwargs)


class OneHotAdder(gym.Wrapper):
    """给观测添加one-hot任务编码"""

    def __init__(
        self, env: gym.Env, one_hot_idx: int, one_hot_len: int, orig_one_hot_dim: int = 0
    ) -> None:
        super().__init__(env)
        assert 0 <= one_hot_idx < one_hot_len
        self.one_hot_idx = one_hot_idx
        self.one_hot_len = one_hot_len
        self.to_append = np.zeros(one_hot_len)
        self.to_append[one_hot_idx] = 1.0

        orig_obs_low = self.env.observation_space.low
        orig_obs_high = self.env.observation_space.high
        if orig_one_hot_dim > 0:
            orig_obs_low = orig_obs_low[:-orig_one_hot_dim]
            orig_obs_high = orig_obs_high[:-orig_one_hot_dim]
        
        new_low = np.concatenate([orig_obs_low, np.zeros(one_hot_len)])
        new_high = np.concatenate([orig_obs_high, np.ones(one_hot_len)])
        self.observation_space = Box(low=new_low, high=new_high, dtype=np.float32)
        
        self.orig_one_hot_dim = orig_one_hot_dim

    def _append_one_hot(self, obs: np.ndarray) -> np.ndarray:
        if self.orig_one_hot_dim > 0:
            obs = obs[: -self.orig_one_hot_dim]
        return np.concatenate([obs, self.to_append]).astype(np.float32)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._append_one_hot(obs), reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        return self._append_one_hot(obs), info


class RandomizationWrapper(gym.Wrapper):
    """随机化初始状态（简化版，ManiSkill本身已有随机化）"""

    ALLOWED_KINDS = [
        "deterministic",
        "random_init_all",
    ]

    def __init__(self, env: gym.Env, kind: str = "random_init_all") -> None:
        assert kind in RandomizationWrapper.ALLOWED_KINDS
        super().__init__(env)
        self.kind = kind
        # ManiSkill默认就是随机初始化的，所以这里主要是接口兼容

    def set_task(self, task):
        """ManiSkill单任务不需要切换，保留接口"""
        pass

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
