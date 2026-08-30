# -*- coding: utf-8 -*-
"""
ManiSkill3 SINGLE-TASK PPO with the unified 42-dim observation interface.

This is the OFFICIAL ManiSkill PPO baseline (examples/baselines/ppo/ppo.py)
with ONLY TWO categories of change:

  1. Observation interface: obs_mode="state_dict" + conversion of the nested
     state_dict obs into the fixed 42-dim semantic-slot vector
     (unified_obs.build_unified_obs_batch). Same keys, same absolute poses,
     slots aligned across the five tasks. Nothing else about the observation
     is touched.

  2. CPU / hardware adaptation:
     - sim_backend="cpu" (this machine has no CUDA). ManiSkill forbids
       num_envs>1 with a CPU sim backend inside one process, so N independent
       num_envs=1 environments are stacked in MultiSingleEnvVector, which
       presents the EXACT same interface as ManiSkillVectorEnv: batched torch
       tensors, SAME_STEP auto-reset, terminal obs exposed via
       infos["final_observation"], episode metrics via
       infos["final_info"]["episode"], record_metrics accumulators and
       ignore_terminations semantics.
     - num_envs 512->16, num_steps 50->512 (batch 8192, same order as the
       official 25600); no video/W&B/tensorboard; ckpts in
       ckpts/ppo_official_unified/.

Every ALGORITHMIC hyperparameter is the official default:
  lr=3e-4 constant, gamma=0.8, gae_lambda=0.9, epochs=4, target_kl=0.1,
  minibatches=32, clip=0.2, ent_coef=0.0, vf_coef=0.5, no value clipping,
  3-layer 256 Tanh orthogonal-init actor-critic, final_values bootstrap for
  every in-rollout episode end, partial reset on, eval reconfigure per reset.

Usage:
    cd E:\\munichi\\maniskill-cl\\cl
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    python ppo_official_unified.py --env-id PushCube-v1 --total-timesteps 2000000
"""
import os
import json
import random
import time
import argparse
from collections import defaultdict

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_conda = r"E:\My_programs\anaconda\envs\diffcl10\Library\bin"
if _conda not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda + os.pathsep + os.environ.get("PATH", "")

import mani_skill.envs
from mani_skill.utils import gym_utils

from unified_obs import UNIFIED_OBS_DIM, SLOT_LAYOUT, build_unified_obs_batch

CONTROL_MODE = "pd_joint_delta_pos"

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts", "ppo_official_unified")
os.makedirs(CKPT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="PushCube-v1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-eval-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=512)
    p.add_argument("--eval-freq", type=int, default=25, help="in iterations")
    return p.parse_args()


# ============================================================
# CPU vector env: N independent num_envs=1 ManiSkill envs stacked,
# exposing the SAME interface as mani_skill.vector.wrappers.gymnasium.ManiSkillVectorEnv
# ============================================================

class MultiSingleEnvVector:
    """Stack of N single (num_envs=1) CPU ManiSkill environments.

    Mirrors ManiSkillVectorEnv behaviour exactly:
      * reset/step return batched torch tensors;
      * observations are the unified 42-dim vectors (conversion done here);
      * SAME_STEP auto-reset: when an env ends, the returned obs is the fresh
        post-reset obs, and the terminal obs is exposed in
        infos["final_observation"];
      * record_metrics: per-env success_once / fail_once / return /
        episode_len accumulators, snapshotted into infos["final_info"]
        ["episode"] at episode end (before the accumulators are cleared);
      * ignore_terminations: terminations are masked out (used by eval envs so
        every episode runs the full horizon and only truncation resets).
    """

    def __init__(self, envs, cfg, ignore_terminations=False, record_metrics=True):
        self.envs = envs
        self._env = envs[0]  # for gym_utils.find_max_episode_steps_value
        self.cfg = cfg
        self.num_envs = len(envs)
        self.ignore_terminations = ignore_terminations
        self.record_metrics = record_metrics
        self.spec = envs[0].spec

        # gymnasium>=1.0 wrappers do not proxy non-standard attributes through
        # __getattr__; use get_wrapper_attr to walk the wrapper chain (same as
        # ManiSkillVectorEnv does).
        self.single_action_space = envs[0].get_wrapper_attr("single_action_space")
        self.action_space = self.single_action_space
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(UNIFIED_OBS_DIM,), dtype=np.float32)
        self.observation_space = self.single_observation_space
        self.max_episode_steps = gym_utils.find_max_episode_steps_value(envs[0])

        self.returns = torch.zeros(self.num_envs, dtype=torch.float32)
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.fail_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.elapsed = torch.zeros(self.num_envs, dtype=torch.int32)

    def _convert(self, raw_obs):
        # raw obs of one num_envs=1 env -> (1, 42)
        return build_unified_obs_batch(raw_obs, self.cfg)

    def reset(self, *, seed=None, options=None):
        obs_list = []
        for i, env in enumerate(self.envs):
            s = None if seed is None else int(seed) + i
            obs, _ = env.reset(seed=s)
            obs_list.append(self._convert(obs))
        self.returns.zero_()
        self.success_once.zero_()
        self.fail_once.zero_()
        self.elapsed.zero_()
        return torch.cat(obs_list, dim=0), {}

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32)
        raw_obs = []
        rews = torch.zeros(self.num_envs, dtype=torch.float32)
        terms = torch.zeros(self.num_envs, dtype=torch.bool)
        truncs = torch.zeros(self.num_envs, dtype=torch.bool)
        final_obs = torch.zeros((self.num_envs, UNIFIED_OBS_DIM), dtype=torch.float32)

        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(action[i:i + 1])
            rews[i] = rew.view(-1)[0]
            terms[i] = term.view(-1)[0]
            truncs[i] = trunc.view(-1)[0]
            # metrics accumulation (same order as ManiSkillVectorEnv.step)
            self.returns[i] += rews[i]
            self.elapsed[i] += 1
            if "success" in info:
                self.success_once[i] = self.success_once[i] or bool(info["success"].view(-1)[0])
            if "fail" in info:
                self.fail_once[i] = self.fail_once[i] or bool(info["fail"].view(-1)[0])
            raw_obs.append(obs)

        if self.ignore_terminations:
            terms[:] = False
        dones = torch.logical_or(terms, truncs)

        infos = {}
        if dones.any():
            # snapshot terminal metrics/obs BEFORE auto-resetting (same point as
            # ManiSkillVectorEnv's torch_clone_dict of infos)
            ep_info = {
                "success_once": self.success_once.clone(),
                "fail_once": self.fail_once.clone(),
                "return": self.returns.clone(),
                "episode_len": self.elapsed.clone(),
                "reward": (self.returns / self.elapsed.clamp(min=1)).clone(),
            }
            for i in range(self.num_envs):
                if dones[i]:
                    final_obs[i] = self._convert(raw_obs[i])[0]
            infos["final_observation"] = final_obs
            infos["final_info"] = {"episode": ep_info}
            infos["_final_info"] = dones.clone()
            infos["_final_observation"] = dones.clone()

            # SAME_STEP auto-reset: done envs reset now; the obs returned for
            # them is the fresh post-reset obs
            for i in range(self.num_envs):
                if dones[i]:
                    obs_new, _ = self.envs[i].reset()
                    raw_obs[i] = obs_new
                    self.returns[i] = 0.0
                    self.success_once[i] = False
                    self.fail_once[i] = False
                    self.elapsed[i] = 0

        obs_out = torch.cat([self._convert(o) for o in raw_obs], dim=0)
        return obs_out, rews, terms, truncs, infos

    def close(self):
        for env in self.envs:
            env.close()


# ============================================================
# Official PPO Agent (verbatim from examples/baselines/ppo/ppo.py)
# ============================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)),
                       std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = parse_args()

    # ---- official hyperparameters (defaults from ppo.py) ----
    learning_rate = 3e-4
    anneal_lr = False
    gamma = 0.8
    gae_lambda = 0.9
    num_minibatches = 32
    update_epochs = 4
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = False
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = 0.1
    reward_scale = 1.0
    partial_reset = True
    eval_partial_reset = False
    eval_reconfiguration_freq = 1

    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = args.total_timesteps // batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print("=" * 70)
    print(f"OFFICIAL PPO + unified 42-dim obs | task={args.env_id}")
    print(f"steps={args.total_timesteps:,} envs={args.num_envs} "
          f"num_steps={args.num_steps} batch={batch_size} gamma={gamma} "
          f"lr={learning_rate} ent={ent_coef}")
    print("=" * 70)

    # ---- env setup (CPU adaptation: stacked num_envs=1 envs) ----
    cfg = SLOT_LAYOUT[args.env_id]

    def _make_vec(num_envs, ignore_terminations, reconfiguration_freq):
        envs = []
        for _ in range(num_envs):
            envs.append(gym.make(
                args.env_id, num_envs=1, obs_mode="state_dict",
                control_mode=CONTROL_MODE, sim_backend="cpu",
                reconfiguration_freq=reconfiguration_freq))
        return MultiSingleEnvVector(envs, cfg,
                                    ignore_terminations=ignore_terminations,
                                    record_metrics=True)

    envs = _make_vec(args.num_envs,
                     ignore_terminations=not partial_reset,
                     reconfiguration_freq=None)
    eval_envs = _make_vec(args.num_eval_envs,
                          ignore_terminations=not eval_partial_reset,
                          reconfiguration_freq=eval_reconfiguration_freq)

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    print(f"task horizon: {max_episode_steps} steps/episode")

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    obs_store = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions_store = torch.zeros((args.num_steps, args.num_envs) + act_shape).to(device)
    logprobs_store = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_store = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_store = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_store = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    action_space_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_space_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    best_sr = 0.0
    eval_history = []
    for iteration in range(1, num_iterations + 1):
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()

        # ---------------- evaluation (official schedule) ----------------
        if (iteration - 1) % args.eval_freq == 0 or iteration == num_iterations:
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            # run one full horizon (+small margin) so every eval env truncates
            # exactly once -> num_eval_envs complete episodes each eval
            for _ in range(max_episode_steps + 10):
                with torch.no_grad():
                    eval_obs, eval_rew, _, _, eval_infos = eval_envs.step(
                        agent.get_action(eval_obs, deterministic=True))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += int(mask.sum())
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            sr = np.nan
            if "success_once" in eval_metrics:
                sr = float(torch.stack(eval_metrics["success_once"]).float().mean())
                best_sr = max(best_sr, sr)
            ret = float(torch.stack(eval_metrics["return"]).float().mean()) if "return" in eval_metrics else np.nan
            print(f"  -- Eval @ step {global_step:,}: SR={sr*100:.1f}% "
                  f"avg_R={ret:.2f} episodes={num_episodes} best_SR={best_sr*100:.1f}%",
                  flush=True)
            eval_history.append({"step": global_step, "success_rate": sr,
                                 "avg_return": ret, "episodes": num_episodes})
            ckpt_path = os.path.join(CKPT_DIR, f"{args.env_id}_seed{args.seed}_latest.pt")
            torch.save(agent.state_dict(), ckpt_path)

        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * learning_rate

        # ---------------- rollout ----------------
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_store[step] = next_obs
            dones_store[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_store[step] = value.flatten()
            actions_store[step] = action
            logprobs_store[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards_store[step] = reward.view(-1) * reward_scale
            if "final_info" in infos:
                done_mask = infos["_final_info"]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = \
                        agent.get_value(infos["final_observation"][done_mask]).view(-1)

        # ---------------- GAE (verbatim official logic) ----------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_store).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones_store[t + 1]
                    nextvalues = values_store[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards_store[t] + gamma * real_next_values - values_store[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
            returns = advantages + values_store

        b_obs = obs_store.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_store.reshape(-1)
        b_actions = actions_store.reshape((-1,) + act_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_store.reshape(-1)

        # ---------------- PPO update (verbatim official logic) ----------------
        agent.train()
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                if target_kl is not None and approx_kl > target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -clip_coef, clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
            if target_kl is not None and approx_kl > target_kl:
                break

        if iteration % 5 == 0 or iteration == 1:
            sps = int(global_step / (time.time() - start_time))
            print(f"  [{iteration}/{num_iterations}] step={global_step:,} "
                  f"sps={sps} v_loss={v_loss.item():.3f} kl={approx_kl.item():.4f} "
                  f"clipfrac={np.mean(clipfracs):.3f} elapsed={(time.time()-start_time)/60:.1f}min",
                  flush=True)

    final_path = os.path.join(CKPT_DIR, f"{args.env_id}_seed{args.seed}_final.pt")
    torch.save(agent.state_dict(), final_path)
    result_path = os.path.join(CKPT_DIR, f"{args.env_id}_seed{args.seed}_results.json")
    with open(result_path, "w") as f:
        json.dump({"task": args.env_id, "seed": args.seed,
                   "total_timesteps": args.total_timesteps,
                   "best_success_rate": best_sr,
                   "eval_history": eval_history}, f, indent=2)
    print(f"\nTraining done. best_SR={best_sr*100:.1f}%")
    print(f"Final checkpoint: {final_path}")
    print(f"Results: {result_path}")
    envs.close()
    eval_envs.close()
