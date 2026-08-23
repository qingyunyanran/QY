"""
env_wrapper.py - ManiSkill Environment Wrapper & Expert Data Collection

Contains:
  - ManiSkillWrapper: wraps a single ManiSkill env with obs padding + task_id one-hot
  - PolicyRunner: loads pre-trained policies, collects expert trajectories, saves to .npz

Run:
  cd E:\\munichi\\continual_learning_v2
  $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" env_wrapper.py
"""

import os
import sys
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # Register ManiSkill environments
from tqdm import tqdm

from config import (
    TASK_CONFIGS, TASK_SEQUENCE, MAX_OBS_DIM, TASK_ID_DIM, OBS_DIM, ACTION_DIM,
    ENV_CONFIG, EXPERT_CONFIG, EXPERT_DATA_DIR, NUM_TASKS, USE_SCRIPTED_EXPERT,
)
from scripted_experts import create_scripted_expert

# ============================================================
# ManiSkill Environment Wrapper
# ============================================================

class ManiSkillWrapper:
    """Wraps a single ManiSkill3 environment for CPU-only continual learning.

    Observation processing:
      1. Extract raw obs from torch.Tensor / dict → numpy 1-D array
      2. Squeeze batch dim if present
      3. Zero-pad to MAX_OBS_DIM (48)
      4. Concatenate task_id one-hot (4) → final dim = 52

    Action processing:
      1. numpy (7,) → torch (1, 7) on the correct device
    """

    def __init__(self, env_id, task_id, control_mode=None, sim_backend=None,
                 render_mode=None, max_episode_steps=None):
        """
        Args:
            env_id: ManiSkill environment id, e.g. "PushCube-v1"
            task_id: integer task index for one-hot encoding
            control_mode: e.g. "pd_ee_delta_pose"
            sim_backend: "cpu" or "gpu"
            render_mode: "cameras" for headless, "human" for viewer
            max_episode_steps: override max episode length
        """
        self.task_id = task_id
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps or EXPERT_CONFIG["max_steps"]

        # Build environment kwargs
        control_mode = control_mode or ENV_CONFIG["control_mode"]
        sim_backend = sim_backend or ENV_CONFIG["sim_backend"]
        render_mode = render_mode or ENV_CONFIG["render_mode"]

        self.env = gym.make(
            env_id,
            num_envs=1,                    # CPU: single env
            obs_mode=ENV_CONFIG.get("obs_mode", "state_dict"),
            control_mode=control_mode,
            sim_backend=sim_backend,
            render_mode=render_mode,
            max_episode_steps=self.max_episode_steps,
        )

        # Pre-compute task_id one-hot (numpy)
        self._task_onehot = np.zeros(TASK_ID_DIM, dtype=np.float32)
        self._task_onehot[task_id] = 1.0
        self._last_raw_obs = None  # stores raw obs (numpy) before padding for expert policies
        self._last_raw_obs_dict = None  # stores raw obs dict for scripted experts

        # Detect raw observation dimension
        obs, _ = self.env.reset()
        raw = self._extract_raw_obs(obs)
        self.raw_obs_dim = len(raw)
        print(f"[ManiSkillWrapper] {env_id} | raw_obs_dim={self.raw_obs_dim}, "
              f"padded_obs_dim={OBS_DIM}, action_dim={ACTION_DIM}")

    # ----------------------------------------------------------
    # Observation processing
    # ----------------------------------------------------------
    def _extract_raw_obs(self, obs):
        """Convert env obs to 1-D numpy float32 array."""
        if isinstance(obs, dict):
            parts = []
            for k in sorted(obs.keys()):
                v = obs[k]
                if isinstance(v, dict):
                    # Recursively handle nested dicts (state_dict mode)
                    for sub_k in sorted(v.keys()):
                        sub_v = v[sub_k]
                        if isinstance(sub_v, torch.Tensor):
                            sub_v = sub_v.cpu().numpy()
                        parts.append(np.asarray(sub_v, dtype=np.float32).flatten())
                else:
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().numpy()
                    parts.append(np.asarray(v, dtype=np.float32).flatten())
            return np.concatenate(parts)
        else:
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
            obs = np.asarray(obs, dtype=np.float32)
            return obs.flatten()

    def _pad_obs(self, raw_obs):
        """Zero-pad raw observation to MAX_OBS_DIM."""
        padded = np.zeros(MAX_OBS_DIM, dtype=np.float32)
        n = min(len(raw_obs), MAX_OBS_DIM)
        padded[:n] = raw_obs[:n]
        return padded

    def process_obs(self, obs):
        """Full obs pipeline: extract → pad → concat task_id → (52,).
        Also stores raw obs in self._last_raw_obs for expert policies.
        Stores raw obs dict in self._last_raw_obs_dict for scripted experts."""
        raw = self._extract_raw_obs(obs)
        self._last_raw_obs = raw.copy()
        # Store raw obs dict for scripted expert policies
        if isinstance(obs, dict):
            self._last_raw_obs_dict = obs
        else:
            self._last_raw_obs_dict = None
        padded = self._pad_obs(raw)
        return np.concatenate([padded, self._task_onehot]).astype(np.float32)

    # ----------------------------------------------------------
    # Action processing
    # ----------------------------------------------------------
    @staticmethod
    def process_action(action):
        """Convert numpy action to torch tensor (1, action_dim)."""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = np.asarray(action, dtype=np.float32).flatten()
        return torch.from_numpy(action).unsqueeze(0)  # (1, 7)

    # ----------------------------------------------------------
    # Gym API wrappers
    # ----------------------------------------------------------
    def reset(self, seed=None):
        """Reset env, return (processed_obs, raw_obs, raw_dict, info)."""
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        processed = self.process_obs(obs)
        raw = self._last_raw_obs.copy() if self._last_raw_obs is not None else processed
        raw_dict = self._last_raw_obs_dict
        return processed, raw, raw_dict, info

    def step(self, action):
        """Step env with numpy action, return (processed_obs, raw_obs, raw_dict, reward, terminated, truncated, info)."""
        torch_action = self.process_action(action)
        obs, reward, terminated, truncated, info = self.env.step(torch_action)
        processed = self.process_obs(obs)
        raw = self._last_raw_obs.copy() if self._last_raw_obs is not None else processed
        raw_dict = self._last_raw_obs_dict
        return processed, raw, raw_dict, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)


# ============================================================
# Policy Runner - Expert Data Collection
# ============================================================

class PolicyRunner:
    """Loads pre-trained policies and collects expert trajectories.

    Supports:
      - Scripted expert policies (default, no model file needed)
      - SB3 PPO models (PushCube, PickCube)
      - torch.load() models with .predict() or direct forward (StackCube, PegInsertionSide)
    """

    def __init__(self, task_name, expert_model_path=None):
        """
        Args:
            task_name: e.g. "PushCube-v1"
            expert_model_path: path to the pre-trained model file (not needed for scripted)
        """
        self.task_name = task_name
        self.task_config = TASK_CONFIGS[task_name]
        self.task_id = self.task_config["task_id"]
        self.expert_type = self.task_config["expert_type"]
        self.expert_model_path = expert_model_path or self.task_config["expert_model_path"]
        self.model = None
        self.scripted_policy = None

    def load_policy(self):
        """Load the pre-trained policy based on expert_type."""
        if self.expert_type == "scripted":
            self._load_scripted_expert()
        elif self.expert_type == "sb3_ppo":
            self._load_sb3_ppo()
        elif self.expert_type in ("torch_bc", "torch_mp"):
            self._load_torch_model()
        else:
            raise ValueError(f"Unknown expert_type: {self.expert_type}")
        print(f"[PolicyRunner] Loaded {self.expert_type} expert for {self.task_name}")

    def _load_scripted_expert(self):
        """Instantiate scripted expert policy."""
        self.scripted_policy = create_scripted_expert(self.task_name)
        self.scripted_policy.reset()
        print(f"[PolicyRunner] Scripted expert policy created for {self.task_name}")

    def _load_sb3_ppo(self):
        """Load SB3 PPO model."""
        from stable_baselines3 import PPO
        self.model = PPO.load(self.expert_model_path)
        print(f"[PolicyRunner] SB3 PPO loaded from {self.expert_model_path}")

    def _load_torch_model(self):
        """Load torch model (BC or MP). Expects a model with .predict(obs) -> action."""
        self.model = torch.load(self.expert_model_path, map_location="cpu")
        if hasattr(self.model, "eval"):
            self.model.eval()
        print(f"[PolicyRunner] Torch model loaded from {self.expert_model_path}")

    def get_action(self, obs_processed, obs_raw=None, obs_dict=None):
        """Get action from expert policy.

        Args:
            obs_processed: numpy array (52,) - padded + task_id (for BC/torch models)
            obs_raw: numpy array (raw_obs_dim,) - unpadded obs (for SB3 PPO models)
            obs_dict: raw obs dict from ManiSkill env (for scripted experts)

        Returns:
            action: numpy array (7,)
        """
        if self.expert_type == "scripted":
            # Scripted expert needs raw obs dict with 'extra' fields
            if obs_dict is None:
                raise ValueError("Scripted expert requires raw obs_dict (with 'extra' key)")
            action = self.scripted_policy.get_action(obs_dict)
            return np.asarray(action, dtype=np.float32).flatten()[:ACTION_DIM]
        elif self.expert_type == "sb3_ppo":
            # SB3 expects raw obs (the same dim it was trained on)
            obs_for_sb3 = obs_raw if obs_raw is not None else obs_processed
            action, _ = self.model.predict(obs_for_sb3, deterministic=True)
            return action.astype(np.float32)
        else:
            # Torch model: expects (1, obs_dim) tensor
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs_processed).float().unsqueeze(0)
                if hasattr(self.model, "predict"):
                    action = self.model.predict(obs_tensor)
                    if isinstance(action, tuple):
                        action = action[0]
                else:
                    # Direct forward pass
                    action = self.model(obs_tensor)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                return action.flatten().astype(np.float32)

    def collect_episodes(self, env_wrapper, n_episodes=None, max_steps=None, seed=None):
        """Collect expert trajectories.

        Args:
            env_wrapper: ManiSkillWrapper instance
            n_episodes: number of episodes to collect
            max_steps: max steps per episode
            seed: random seed

        Returns:
            episodes: list of dicts, each with keys:
                'observations': (T, 52)
                'actions': (T, 7)
                'rewards': (T,)
                'success': bool
                'return': float
        """
        n_episodes = n_episodes or EXPERT_CONFIG["n_episodes"]
        max_steps = max_steps or EXPERT_CONFIG["max_steps"]
        seed = seed or EXPERT_CONFIG["seed"]

        episodes = []
        success_count = 0

        pbar = tqdm(range(n_episodes), desc=f"Collecting {self.task_name}", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for ep in pbar:
            obs, raw_obs, raw_dict, info = env_wrapper.reset(seed=seed + ep)
            
            # Reset scripted policy at episode start
            if self.expert_type == "scripted" and self.scripted_policy is not None:
                self.scripted_policy.reset()
            
            ep_obs, ep_actions, ep_rewards = [], [], []
            ep_success = False

            for step in range(max_steps):
                action = self.get_action(obs, raw_obs, obs_dict=raw_dict)
                next_obs, next_raw, next_raw_dict, reward, terminated, truncated, info = env_wrapper.step(action)

                ep_obs.append(obs)
                ep_actions.append(action)
                ep_rewards.append(reward)

                # Check success from info
                if isinstance(info, dict) and info.get("success", False):
                    ep_success = True
                elif isinstance(info, dict) and "final_info" in info:
                    # ManiSkill may pack info differently
                    final_info = info["final_info"]
                    if isinstance(final_info, dict):
                        ep_success = ep_success or final_info.get("success", False)
                    elif isinstance(final_info, np.ndarray):
                        ep_success = ep_success or bool(final_info[0].get("success", False)
                                                        if len(final_info) > 0 else False)

                obs = next_obs
                raw_obs = next_raw
                raw_dict = next_raw_dict
                if terminated or truncated:
                    break

            ep_return = sum(ep_rewards)
            episodes.append({
                "observations": np.array(ep_obs, dtype=np.float32),    # (T, 52)
                "actions": np.array(ep_actions, dtype=np.float32),     # (T, 7)
                "rewards": np.array(ep_rewards, dtype=np.float32),     # (T,)
                "success": ep_success,
                "return": ep_return,
            })

            if ep_success:
                success_count += 1

            pbar.set_postfix(sr=f"{success_count}/{ep+1}", steps=len(ep_obs))

        pbar.close()

        sr = success_count / n_episodes * 100
        print(f"[PolicyRunner] {self.task_name}: {success_count}/{n_episodes} "
              f"successful ({sr:.1f}%)")

        return episodes

    @staticmethod
    def save_expert_data(episodes, save_path):
        """Save collected episodes to .npz file.

        Format (object arrays for variable-length episodes):
            obs_list[i]: (T_i, 52) observations
            actions_list[i]: (T_i, 7) actions
            rewards_list[i]: (T_i,) rewards
            successes[i]: bool
            returns[i]: float
            n_episodes: int
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        n = len(episodes)
        obs_list = np.empty(n, dtype=object)
        actions_list = np.empty(n, dtype=object)
        rewards_list = np.empty(n, dtype=object)
        successes = np.zeros(n, dtype=bool)
        returns = np.zeros(n, dtype=np.float32)

        for i, ep in enumerate(episodes):
            obs_list[i] = ep["observations"]
            actions_list[i] = ep["actions"]
            rewards_list[i] = ep["rewards"]
            successes[i] = ep["success"]
            returns[i] = ep["return"]

        np.savez_compressed(
            save_path,
            obs_list=obs_list,
            actions_list=actions_list,
            rewards_list=rewards_list,
            successes=successes,
            returns=returns,
            n_episodes=n,
        )
        print(f"[PolicyRunner] Saved {n} episodes to {save_path}")

    @staticmethod
    def load_expert_data(load_path):
        """Load expert data from .npz file.

        Returns:
            dict with keys: obs_list, actions_list, rewards_list, successes, returns, n_episodes
        """
        data = np.load(load_path, allow_pickle=True)
        return {
            "obs_list": data["obs_list"],
            "actions_list": data["actions_list"],
            "rewards_list": data["rewards_list"],
            "successes": data["successes"],
            "returns": data["returns"],
            "n_episodes": int(data["n_episodes"]),
        }

    @staticmethod
    def flatten_episodes(expert_data):
        """Flatten episode data into transition arrays for BC training.

        Returns:
            all_obs: (N, 52)
            all_actions: (N, 7)
            all_rewards: (N,)
        """
        all_obs = np.concatenate([expert_data["obs_list"][i]
                                  for i in range(expert_data["n_episodes"])], axis=0)
        all_actions = np.concatenate([expert_data["actions_list"][i]
                                      for i in range(expert_data["n_episodes"])], axis=0)
        all_rewards = np.concatenate([expert_data["rewards_list"][i]
                                      for i in range(expert_data["n_episodes"])], axis=0)
        return all_obs, all_actions, all_rewards


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing ManiSkillWrapper + Scripted Expert Policy")
    print("=" * 60)

    # Test 1: ManiSkillWrapper basic
    try:
        wrapper = ManiSkillWrapper("PushCube-v1", task_id=0)
        obs, raw_obs, raw_dict, info = wrapper.reset(seed=42)
        print(f"Obs shape: {obs.shape}")
        print(f"Raw obs shape: {raw_obs.shape}")
        print(f"Raw dict type: {type(raw_dict)}")
        if raw_dict is not None:
            print(f"Raw dict keys: {list(raw_dict.keys())[:5]}...")
        print(f"Obs dtype: {obs.dtype}")
        print(f"Obs[:10]: {obs[:10]}")
        print(f"Task one-hot part: {obs[MAX_OBS_DIM:]}")

        action = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        next_obs, next_raw, next_raw_dict, reward, term, trunc, info = wrapper.step(action)
        print(f"Next obs shape: {next_obs.shape}")
        print(f"Reward: {reward}, terminated: {term}, truncated: {trunc}")

        wrapper.close()
        print("\n[OK] ManiSkillWrapper test passed!")
    except Exception as e:
        print(f"[ERROR] ManiSkillWrapper test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Scripted expert policies
    print("\n" + "=" * 60)
    print("Testing Scripted Expert Policies")
    print("=" * 60)
    try:
        from scripted_experts import create_scripted_expert
        
        for task_name in ["PushCube-v1", "PickCube-v1", "StackCube-v1", "PegInsertionSide-v1"]:
            policy = create_scripted_expert(task_name)
            policy.reset()
            
            # Simulate obs dict (as if from ManiSkill env)
            obs = {
                'extra': {
                    'tcp_pose': np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
                    'cube_pos': np.array([0.1, 0.0, 0.02]),
                    'obj_pos': np.array([0.1, 0.0, 0.02]),
                    'cubeA_pose': np.array([0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]),
                    'cubeB_pose': np.array([0.1, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]),
                    'target_pos': np.array([0.2, 0.0, 0.02]),
                    'goal_pos': np.array([0.15, 0.0, 0.08]),
                    'peg_pose': np.array([0.1, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]),
                    'box_hole_pose': np.array([0.2, 0.1, 0.1, 1.0, 0.0, 0.0, 0.0]),
                    'peg_half_size': np.array([0.03, 0.015, 0.015]),
                }
            }
            
            action = policy.get_action(obs)
            print(f"  {task_name}: action={action[:3]}..., phase={policy.phase}")
            
            # Run a few more steps
            for i in range(3):
                action = policy.get_action(obs)
            print(f"    after 4 steps: phase={policy.phase}, action={action[:3]}...")
        
        print("\n[OK] All scripted expert policies working!")
    except Exception as e:
        print(f"[ERROR] Scripted expert test failed: {e}")
        import traceback
        traceback.print_exc()
