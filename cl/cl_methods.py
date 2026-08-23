"""
cl_methods.py - Continual Learning Methods (SeqFT, ER, DISTR)

Contains:
  - BaseCLMethod: abstract base class for CL methods
  - SeqFT: Sequential fine-tuning (no replay, naive baseline)
  - ER: Experience Replay (store old task transitions, mix during training)
  - DISTR: Diffusion-based replay (train diffusion model per task, generate replay)

Each method:
  1. train_task(task_id, expert_data, env_wrapper): train BC on current task
  2. get_replay_data(): return (obs, actions) from old tasks for replay
  3. after_task(task_id, expert_data): post-training hook (save models, train diffusion, etc.)

Reference: ContinualWorld evaluation protocol.

Run:
  cd E:\\munichi\\continual_learning_v2
  $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" -u main.py --method all
"""

import os
import numpy as np
import torch

from config import (
    OBS_DIM, ACTION_DIM, BC_CONFIG, ER_CONFIG, DISTR_CONFIG,
    MODEL_DIR, NUM_TASKS, TASK_SEQUENCE,
)
from bc_agent import BCAgent
from diffusion_model import ConditionalDDPM


def _flatten_expert_data(expert_data):
    """Flatten episode data into transition arrays.

    Args:
        expert_data: dict with obs_list, actions_list, rewards_list, n_episodes

    Returns:
        all_obs: (N, obs_dim)
        all_actions: (N, action_dim)
        all_rewards: (N,)
    """
    n = expert_data["n_episodes"]
    all_obs = np.concatenate([expert_data["obs_list"][i] for i in range(n)], axis=0)
    all_actions = np.concatenate([expert_data["actions_list"][i] for i in range(n)], axis=0)
    all_rewards = np.concatenate([expert_data["rewards_list"][i] for i in range(n)], axis=0)
    return all_obs, all_actions, all_rewards


# ============================================================
# Base CL Method
# ============================================================

class BaseCLMethod:
    """Abstract base class for continual learning methods.

    All methods use BC as the base learner. The difference is how they
    handle catastrophic forgetting when learning new tasks.
    """

    def __init__(self, method_name="base", device="cpu"):
        self.method_name = method_name
        self.device = device
        self.agent = BCAgent(device=device)
        self.n_tasks_completed = 0

    def train_task(self, task_id, expert_data, env_wrapper, epochs=None, verbose=True):
        """Train BC on the current task.

        Args:
            task_id: integer task index
            expert_data: dict with obs_list, actions_list, etc.
            env_wrapper: ManiSkillWrapper for the current task
            epochs: BC training epochs
            verbose: print progress

        Returns:
            dict with training metrics
        """
        # Flatten expert episodes to transitions
        all_obs, all_actions, _ = _flatten_expert_data(expert_data)

        # Get replay data (subclasses override this)
        replay_obs, replay_actions = self.get_replay_data()

        if replay_obs is not None and len(replay_obs) > 0:
            metrics = self.agent.train_with_replay(
                new_obs=all_obs, new_actions=all_actions,
                replay_obs=replay_obs, replay_actions=replay_actions,
                replay_ratio=self._get_replay_ratio(),
                epochs=epochs, verbose=verbose,
            )
        else:
            metrics = self.agent.train(
                expert_obs=all_obs, expert_actions=all_actions,
                epochs=epochs, verbose=verbose,
            )

        self.n_tasks_completed += 1
        return metrics

    def _get_replay_ratio(self):
        """Override in subclasses to set replay ratio."""
        return 0.0

    def get_replay_data(self):
        """Return (replay_obs, replay_actions) from old tasks.

        Returns:
            (obs_array, actions_array) or (None, None) if no replay
        """
        return None, None

    def after_task(self, task_id, expert_data):
        """Post-training hook: called after training on task_id.

        Override in subclasses to store data, train diffusion, etc.
        """
        pass

    def save_model(self, path=None):
        """Save current BC model."""
        if path is None:
            path = os.path.join(MODEL_DIR, f"bc_{self.method_name}_task_{self.n_tasks_completed - 1}.pt")
        self.agent.save(path)
        return path

    def load_model(self, path):
        """Load BC model."""
        self.agent.load(path)

    def evaluate(self, env_wrapper, n_episodes=20, max_steps=200, seed=42, verbose=False):
        """Evaluate current model on a task."""
        return self.agent.evaluate(env_wrapper, n_episodes, max_steps, seed, verbose)


# ============================================================
# SeqFT: Sequential Fine-Tuning (Baseline)
# ============================================================

class SeqFT(BaseCLMethod):
    """Sequential fine-tuning without any replay.

    Simply trains BC on each task's expert data sequentially.
    This is the naive baseline that suffers from catastrophic forgetting.
    """

    def __init__(self, device="cpu"):
        super().__init__(method_name="seqft", device=device)

    def _get_replay_ratio(self):
        return 0.0  # No replay


# ============================================================
# ER: Experience Replay
# ============================================================

class ER(BaseCLMethod):
    """Experience Replay: store old task transitions in a buffer,
    mix with new task data during training.
    """

    def __init__(self, replay_ratio=None, max_buffer_size=None, device="cpu"):
        super().__init__(method_name="er", device=device)
        self.replay_ratio = replay_ratio or ER_CONFIG["replay_ratio"]
        self.max_buffer_size = max_buffer_size or ER_CONFIG["max_buffer_size"]

        # Replay buffer: stores flattened transitions from all past tasks
        self.replay_obs = []      # list of numpy arrays
        self.replay_actions = []  # list of numpy arrays

    def _get_replay_ratio(self):
        return self.replay_ratio

    def get_replay_data(self):
        """Return all stored replay transitions."""
        if len(self.replay_obs) == 0:
            return None, None
        all_obs = np.concatenate(self.replay_obs, axis=0)
        all_actions = np.concatenate(self.replay_actions, axis=0)
        return all_obs, all_actions

    def after_task(self, task_id, expert_data):
        """Store current task's expert data in replay buffer."""
        all_obs, all_actions, _ = _flatten_expert_data(expert_data)

        self.replay_obs.append(all_obs)
        self.replay_actions.append(all_actions)

        # Trim buffer if too large
        total = sum(len(o) for o in self.replay_obs)
        if total > self.max_buffer_size:
            self._trim_buffer()

        total_after = sum(len(o) for o in self.replay_obs)
        print(f"  [ER] Replay buffer: {total_after} transitions from "
              f"{len(self.replay_obs)} tasks")

    def _trim_buffer(self):
        """Trim replay buffer to max size by uniform subsampling."""
        all_obs = np.concatenate(self.replay_obs, axis=0)
        all_actions = np.concatenate(self.replay_actions, axis=0)
        n = len(all_obs)
        if n > self.max_buffer_size:
            indices = np.random.choice(n, self.max_buffer_size, replace=False)
            all_obs = all_obs[indices]
            all_actions = all_actions[indices]
        self.replay_obs = [all_obs]
        self.replay_actions = [all_actions]


# ============================================================
# DISTR: Diffusion-based Strategy for Incremental Task Replay
# ============================================================

class DISTR(BaseCLMethod):
    """Diffusion-based replay: after each task, train a conditional DDPM
    on skilled expert trajectories. During subsequent tasks, use the
    diffusion model to generate synthetic replay data.
    """

    def __init__(self, device="cpu"):
        super().__init__(method_name="distr", device=device)
        self.replay_ratio = DISTR_CONFIG["replay_ratio"]
        self.skilled_ratio = DISTR_CONFIG["skilled_ratio"]
        self.n_generated = DISTR_CONFIG["n_generated_samples"]

        # Store diffusion models per completed task
        self.diffusion_models = {}  # task_id -> ConditionalDDPM

        # Store skilled (obs, action) pairs per task for diffusion training
        self.skilled_data = {}  # task_id -> (obs, actions)

        # Store all expert data for trajectory selection
        self.all_expert_data = {}  # task_id -> expert_data dict

    def _get_replay_ratio(self):
        return self.replay_ratio

    def _select_skilled_trajectories(self, expert_data):
        """Select top-k trajectories by return.

        Args:
            expert_data: dict with obs_list, actions_list, returns, etc.

        Returns:
            (skilled_obs, skilled_actions): flattened transitions from top trajectories
        """
        n_episodes = expert_data["n_episodes"]
        returns = expert_data["returns"]
        k = max(1, int(n_episodes * self.skilled_ratio))

        # Get indices of top-k episodes by return
        top_indices = np.argsort(returns)[::-1][:k]

        skilled_obs = np.concatenate(
            [expert_data["obs_list"][i] for i in top_indices], axis=0
        )
        skilled_actions = np.concatenate(
            [expert_data["actions_list"][i] for i in top_indices], axis=0
        )

        print(f"  [DISTR] Selected {k}/{n_episodes} skilled trajectories, "
              f"{len(skilled_obs)} transitions")

        return skilled_obs, skilled_actions

    def get_replay_data(self):
        """Generate replay data from all trained diffusion models."""
        if len(self.diffusion_models) == 0:
            return None, None

        all_obs = []
        all_actions = []

        for task_id, ddpm in self.diffusion_models.items():
            # Sample observations from skilled data
            skilled_obs, _ = self.skilled_data[task_id]
            n_skill = len(skilled_obs)

            # Determine how many samples to generate
            n_gen = min(self.n_generated, n_skill * 2)

            # Randomly select observations to condition on
            indices = np.random.choice(n_skill, n_gen, replace=(n_gen > n_skill))
            sampled_obs = skilled_obs[indices]

            # Generate actions using diffusion model
            generated_actions = ddpm.sample(sampled_obs, n_samples=1)

        if len(all_obs) == 0:
            return None, None

        replay_obs = np.concatenate(all_obs, axis=0)
        replay_actions = np.concatenate(all_actions, axis=0)

        print(f"  [DISTR] Generated {len(replay_obs)} replay transitions from "
              f"{len(self.diffusion_models)} diffusion models")

        return replay_obs, replay_actions

    def after_task(self, task_id, expert_data):
        """After training on task_id:
        1. Select skilled trajectories
        2. Train diffusion model on skilled data
        3. Store for future replay
        """
        # Store expert data
        self.all_expert_data[task_id] = expert_data

        # Select skilled trajectories
        skilled_obs, skilled_actions = self._select_skilled_trajectories(expert_data)
        self.skilled_data[task_id] = (skilled_obs, skilled_actions)

        # Train diffusion model
        print(f"  [DISTR] Training diffusion model for task {task_id}...")
        ddpm = ConditionalDDPM(device=self.device)
        metrics = ddpm.train(
            skilled_obs, skilled_actions,
            epochs=DISTR_CONFIG["diffusion_epochs"],
            batch_size=DISTR_CONFIG["diffusion_batch_size"],
            verbose=True,
        )
        print(f"  [DISTR] Diffusion model for task {task_id}: "
              f"final_loss={metrics['final_loss']:.6f}")

        # Save diffusion model
        model_path = os.path.join(MODEL_DIR, f"diffusion_distr_task_{task_id}.pt")
        ddpm.save(model_path)

        self.diffusion_models[task_id] = ddpm


# ============================================================
# Factory function
# ============================================================

def create_cl_method(method_name, device="cpu"):
    """Create a CL method by name.

    Args:
        method_name: "seqft", "er", or "distr"
        device: torch device

    Returns:
        BaseCLMethod subclass instance
    """
    methods = {
        "seqft": SeqFT,
        "er": ER,
        "distr": DISTR,
    }
    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(methods.keys())}")
    return methods[method_name](device=device)
