"""
bc_agent.py - Behavior Cloning Agent for Continual Learning

Contains:
  - BCNetwork: MLP policy network (obs → action mean + learned std)
  - BCAgent: training, evaluation, save/load

The BC agent is the base learner for all CL methods (SeqFT, ER, DISTR).
Input: OBS_DIM (52) = MAX_OBS_DIM (48) + TASK_ID_DIM (4)
Output: ACTION_DIM (7) with Tanh squashing + Gaussian noise during training.

Run:
  cd E:\\munichi\\continual_learning_v2
  $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" -u main.py --method all
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import OBS_DIM, ACTION_DIM, BC_CONFIG, DEVICE


# ============================================================
# BC Network
# ============================================================

class BCNetwork(nn.Module):
    """MLP policy: obs → action mean, with Tanh output squashing.

    Architecture:
        input(OBS_DIM) → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(ACTION_DIM) → Tanh

    During training, Gaussian noise is added to the output for exploration/regularization.
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or BC_CONFIG["hidden_dim"]

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Learnable log std for action noise (used during training)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        """Return action mean (before noise)."""
        return self.net(obs)

    def forward_with_noise(self, obs, noise_std=None):
        """Return action with optional Gaussian noise (for training)."""
        mean = self.forward(obs)
        if noise_std is not None and noise_std > 0:
            std = torch.exp(self.log_std) * noise_std
            noise = torch.randn_like(mean) * std
            return torch.clamp(mean + noise, -1.0, 1.0)
        return mean


# ============================================================
# BC Agent
# ============================================================

class BCAgent:
    """Behavior Cloning agent for continual learning.

    Usage:
        agent = BCAgent()
        agent.train(expert_obs, expert_actions, epochs=50)
        success_rate = agent.evaluate(env_wrapper, n_episodes=20)
        agent.save("path/to/model.pt")
        agent.load("path/to/model.pt")
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=None,
                 lr=None, device=None):
        self.device = device or DEVICE
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Build network
        self.model = BCNetwork(obs_dim, action_dim, hidden_dim).to(self.device)

        # Optimizer setup
        lr = lr or BC_CONFIG["lr"]
        weight_decay = BC_CONFIG["weight_decay"]
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

        # Training state
        self.current_epoch = 0
        self.current_noise_std = BC_CONFIG["noise_std"]

    def train(self, expert_obs, expert_actions, epochs=None, batch_size=None,
              verbose=True):
        """Train BC policy on expert demonstrations.

        Args:
            expert_obs: numpy array (N, obs_dim) - expert observations
            expert_actions: numpy array (N, action_dim) - expert actions
            epochs: number of training epochs
            batch_size: mini-batch size
            verbose: print training progress

        Returns:
            dict with training metrics (losses per epoch)
        """
        epochs = epochs or BC_CONFIG["epochs"]
        batch_size = batch_size or BC_CONFIG["batch_size"]
        noise_std = BC_CONFIG["noise_std"]
        noise_decay = BC_CONFIG["noise_decay"]
        grad_clip = BC_CONFIG["grad_clip"]

        # Convert to tensors
        obs_tensor = torch.from_numpy(expert_obs).float().to(self.device)
        act_tensor = torch.from_numpy(expert_actions).float().to(self.device)

        # DataLoader
        dataset = TensorDataset(obs_tensor, act_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)

        epoch_losses = []

        pbar = tqdm(range(epochs), desc="BC Training", disable=not verbose,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for epoch in pbar:
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch_obs, batch_act in loader:
                self.optimizer.zero_grad()

                # Forward with noise
                pred_act = self.model.forward_with_noise(batch_obs, noise_std)
                loss = self.criterion(pred_act, batch_act)

                # Backward
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

            # Decay noise
            noise_std *= noise_decay
            pbar.set_postfix(loss=f"{avg_loss:.4f}", noise=f"{noise_std:.4f}")

        pbar.close()

        self.current_epoch += epochs
        self.current_noise_std = noise_std

        return {"epoch_losses": epoch_losses, "final_loss": epoch_losses[-1]}

    def train_with_replay(self, new_obs, new_actions,
                          replay_obs=None, replay_actions=None,
                          replay_ratio=None, epochs=None, batch_size=None,
                          verbose=True):
        """Train BC with mixed new-task and replay data.

        Args:
            new_obs, new_actions: current task expert data
            replay_obs, replay_actions: replay data from previous tasks (optional)
            replay_ratio: fraction of replay data in each batch (0 to 1)
            epochs, batch_size: training hyperparameters
            verbose: print progress

        Returns:
            dict with training metrics
        """
        epochs = epochs or BC_CONFIG["epochs"]
        batch_size = batch_size or BC_CONFIG["batch_size"]
        replay_ratio = replay_ratio if replay_ratio is not None else 0.5
        noise_std = BC_CONFIG["noise_std"]
        noise_decay = BC_CONFIG["noise_decay"]
        grad_clip = BC_CONFIG["grad_clip"]

        # New task data
        new_obs_t = torch.from_numpy(new_obs).float().to(self.device)
        new_act_t = torch.from_numpy(new_actions).float().to(self.device)

        has_replay = (replay_obs is not None and len(replay_obs) > 0)
        if has_replay:
            rep_obs_t = torch.from_numpy(replay_obs).float().to(self.device)
            rep_act_t = torch.from_numpy(replay_actions).float().to(self.device)

        # Compute batch sizes
        replay_batch = int(batch_size * replay_ratio) if has_replay else 0
        new_batch = batch_size - replay_batch

        n_new = len(new_obs_t)
        n_rep = len(rep_obs_t) if has_replay else 0

        epoch_losses = []

        pbar = tqdm(range(epochs), desc="BC+Replay", disable=not verbose,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for epoch in pbar:
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            # Number of batches determined by new data size
            n_iterations = max(1, n_new // new_batch)

            # Shuffle indices
            new_indices = torch.randperm(n_new)
            if has_replay:
                rep_indices = torch.randperm(n_rep)

            for it in range(n_iterations):
                self.optimizer.zero_grad()

                # Sample new data
                new_idx = new_indices[it * new_batch:(it + 1) * new_batch]
                if len(new_idx) == 0:
                    new_idx = new_indices[torch.randint(0, n_new, (new_batch,))]
                batch_obs = new_obs_t[new_idx]
                batch_act = new_act_t[new_idx]

                # Sample replay data
                if has_replay and replay_batch > 0:
                    rep_start = (it * replay_batch) % n_rep
                    rep_end = min(rep_start + replay_batch, n_rep)
                    if rep_end <= rep_start:
                        rep_idx = torch.randint(0, n_rep, (replay_batch,))
                    else:
                        rep_idx = rep_indices[rep_start:rep_end]
                        if len(rep_idx) < replay_batch:
                            extra = torch.randint(0, n_rep,
                                                  (replay_batch - len(rep_idx),))
                            rep_idx = torch.cat([rep_idx, extra])
                    rep_o = rep_obs_t[rep_idx]
                    rep_a = rep_act_t[rep_idx]
                    batch_obs = torch.cat([batch_obs, rep_o], dim=0)
                    batch_act = torch.cat([batch_act, rep_a], dim=0)

                # Forward
                pred_act = self.model.forward_with_noise(batch_obs, noise_std)
                loss = self.criterion(pred_act, batch_act)

                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            noise_std *= noise_decay
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        pbar.close()

        self.current_epoch += epochs
        return {"epoch_losses": epoch_losses, "final_loss": epoch_losses[-1]}

    def predict(self, obs, deterministic=True):
        """Get action from policy.

        Args:
            obs: numpy array (obs_dim,) or (batch, obs_dim)
            deterministic: if True, no noise

        Returns:
            action: numpy array (action_dim,) or (batch, action_dim)
        """
        self.model.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(self.device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False

            if deterministic:
                action = self.model(obs_t)
            else:
                action = self.model.forward_with_noise(obs_t, self.current_noise_std)

            action = action.cpu().numpy()
            if squeeze:
                action = action.squeeze(0)
            return action

    def evaluate(self, env_wrapper, n_episodes=20, max_steps=200, seed=42,
                 verbose=True):
        """Evaluate policy in environment.

        Args:
            env_wrapper: ManiSkillWrapper instance
            n_episodes: number of evaluation episodes
            max_steps: max steps per episode
            seed: random seed
            verbose: print progress

        Returns:
            dict: success_rate, avg_return, n_success, n_episodes
        """
        self.model.eval()
        successes = 0
        total_return = 0.0

        pbar = tqdm(range(n_episodes), desc="Evaluating", disable=not verbose,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for ep in pbar:
            obs, raw_obs, raw_dict, info = env_wrapper.reset(seed=seed + ep)
            ep_return = 0.0
            ep_success = False

            for step in range(max_steps):
                action = self.predict(obs, deterministic=True)
                obs, raw_obs, raw_dict, reward, terminated, truncated, info = env_wrapper.step(action)
                ep_return += reward

                # Check success
                if isinstance(info, dict) and info.get("success", False):
                    ep_success = True
                elif isinstance(info, dict) and "final_info" in info:
                    fi = info["final_info"]
                    if isinstance(fi, dict):
                        ep_success = ep_success or fi.get("success", False)
                    elif isinstance(fi, np.ndarray) and len(fi) > 0:
                        ep_success = ep_success or fi[0].get("success", False)

                if terminated or truncated:
                    break

            if ep_success:
                successes += 1
            total_return += ep_return
            pbar.set_postfix(sr=f"{successes}/{ep+1}", ret=f"{ep_return:.1f}")

        pbar.close()

        success_rate = successes / n_episodes
        avg_return = total_return / n_episodes

        if verbose:
            print(f"  [Eval] success_rate={success_rate:.2%} | "
                  f"avg_return={avg_return:.2f} | {successes}/{n_episodes}")

        return {
            "success_rate": success_rate,
            "avg_return": avg_return,
            "n_success": successes,
            "n_episodes": n_episodes,
        }

    def save(self, path):
        """Save model state dict."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "current_noise_std": self.current_noise_std,
        }, path)
        print(f"[BCAgent] Saved model to {path}")

    def load(self, path):
        """Load model state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.current_noise_std = checkpoint.get("current_noise_std", BC_CONFIG["noise_std"])
        self.model.eval()
        print(f"[BCAgent] Loaded model from {path}")

    def get_params(self):
        """Return a copy of model parameters (for EWC or other regularization)."""
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def set_params(self, params):
        """Set model parameters from a dict."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(params[name])
