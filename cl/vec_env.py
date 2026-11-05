# -*- coding: utf-8 -*-
"""
Shared vectorised-environment wrappers for the ManiSkill3 CL benchmark.

WHY THIS FILE EXISTS
--------------------
The original port runs N independent `num_envs=1` CPU environments in a Python
`for` loop: ManiSkill forbids `num_envs>1` with `sim_backend="cpu"` inside one
process, so it tops out around 16-32 envs. Two of the four simple tasks still
reached 100% that way, but StackCube did not: PPO collapsed to a fixed action
pattern (SR 0%, entropy -9.94 -> -16.95) with only 8 envs, and the failure
report's first root cause is exactly "official uses 512 parallel envs, this used
8" (see StackCube_PPO_失败分析报告.md).

On a CUDA machine ManiSkill can run ONE env object holding N parallel sims with
`sim_backend="gpu"`. Physics runs on the GPU, the state observation comes back
as GPU tensors, and a step costs a handful of kernel launches instead of N
Python-level `Sim.step()` calls. That is the official PPO setup: 512 envs x 50
steps per rollout = batch 25,600.

    GpuVecEnv   one batched env, everything on the GPU  (official setup)
    CPU path    unchanged, still lives in the main scripts (MultiEnvWrapper)

INTERFACE (same as MultiEnvWrapper, so the trainer only needs a tensor-aware
`_as_tensor` helper instead of special-casing the backend):

    obs, info = envs.reset()
    obs, rew, done, trunc, infos, final_obs = envs.step(actions)

      obs        (N, 42) next observation; finished envs are ALREADY auto-reset
      rew        (N,)    float32
      done       (N,)    bool, `terminated | truncated`
      trunc      (N,)    bool
      infos      dict of batched tensors (`infos["success"]` -> (N,1) bool)
      final_obs  (N, 42) observation AT THE MOMENT each env finished, needed for
                 the GAE terminal bootstrap (same role as MultiEnvWrapper's
                 `final_obs`)

All tensors live on `device`; callers must NOT call `.numpy()` on them.
"""
import numpy as np
import torch
import gymnasium as gym

from unified_obs import UNIFIED_OBS_DIM, build_unified_obs_batch


def to_action_tensor(actions, device):
    """Accept a tensor or an ndarray, return a clipped (N, action_dim) float32
    tensor on `device`. Clipping matches MultiEnvWrapper ([-1, 1] action box)."""
    if torch.is_tensor(actions):
        a = actions.detach().to(device=device, dtype=torch.float32)
    else:
        a = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=device)
    return a.clamp(-1.0, 1.0)


def as_tensor(x, device):
    """Tensor-aware replacement for `torch.tensor(x)`: GPU envs already return
    tensors on the right device, CPU envs return ndarrays."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def _resolve_reconfiguration_freq(reconfiguration_freq, who):
    """Partial auto-reset and scene reconfiguration are mutually exclusive.

    `SapienEnv.reset(options=dict(env_idx=...))` raises
        RuntimeError("Cannot do a partial reset and reconfigure the environment")
    whenever `reconfiguration_freq != 0` and the reset only covers a subset --
    which is exactly what self-resetting batched envs do. The official wrapper
    guards this with `assert reconfiguration_freq == 0 or num_envs == 1`.

    Our call sites pass 1 to mimic the single-task eval protocol (re-randomise
    the scene every episode). For these tabletop tasks freq=0 still re-randomises
    object poses on every `_initialize_episode`; only the asset load/setup is
    skipped. So we clamp to 0 here instead of crashing mid-run, and say so once.
    """
    if reconfiguration_freq is None:
        return None
    if int(reconfiguration_freq) == 0:
        return 0
    if not getattr(_resolve_reconfiguration_freq, "_warned", False):
        _resolve_reconfiguration_freq._warned = True
        print(f"[vec_env] {who}: reconfiguration_freq={reconfiguration_freq} is "
              f"incompatible with partial auto-resets -> using 0 "
              f"(object poses are still randomised every episode)")
    return 0


class GpuVecEnv:
    """One ManiSkill env holding `num_envs` parallel sims on the GPU."""

    backend = "gpu"

    def __init__(self, env_id, num_envs, cfg, control_mode, device="cuda",
                 seed=0, max_episode_steps=None, reconfiguration_freq=None):
        kwargs = dict(
            num_envs=int(num_envs),
            obs_mode="state_dict",
            control_mode=control_mode,
            sim_backend="gpu",
            # 状态观测训练用不到渲染。ManiSkill 的 can_render() 只看
            # render_device is not None，在没有 Vulkan 的容器里会去建
            # RenderSystem 然后崩在 vk::createInstanceUnique，
            # 所以显式关掉（官方文档推荐的 render_backend="none"）。
            render_backend="none",
        )
        # StackCube-v1 is registered with max_episode_steps=50 like the other
        # tabletop tasks, but grasp+carry+place+release+settle does not fit in
        # 50 steps, so the trainer passes 100 for it. The easy tasks pass None
        # and keep their native horizon.
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = int(max_episode_steps)
        reconfiguration_freq = _resolve_reconfiguration_freq(
            reconfiguration_freq, "GpuVecEnv")
        if reconfiguration_freq is not None:
            kwargs["reconfiguration_freq"] = int(reconfiguration_freq)

        self.env = gym.make(env_id, **kwargs)
        self.env_id = env_id
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.obs_dim = UNIFIED_OBS_DIM
        self.action_dim = int(self.env.action_space.shape[-1])
        self.num_resets = 0

    # -- observation ------------------------------------------------------
    def _obs(self, raw_obs):
        return build_unified_obs_batch(raw_obs, self.cfg, device=self.device)

    # -- api --------------------------------------------------------------
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.num_resets += self.num_envs
        return self._obs(obs), info

    def step(self, actions):
        actions = to_action_tensor(actions, self.device)
        raw_obs, rew, terminated, truncated, info = self.env.step(actions)

        # Capture the terminal observation BEFORE resetting: the GAE bootstrap
        # must use V(s_T) of the state where the episode actually ended.
        final_obs = self._obs(raw_obs)

        rew = torch.as_tensor(rew, device=self.device, dtype=torch.float32).reshape(-1)
        terminated = torch.as_tensor(terminated, device=self.device, dtype=torch.bool).reshape(-1)
        truncated = torch.as_tensor(truncated, device=self.device, dtype=torch.bool).reshape(-1)
        done = terminated | truncated

        if bool(done.any()):
            # Partial reset: ManiSkill resets only the selected sub-envs and
            # returns the FULL batch (already-reset envs carry their new obs).
            idx = done.nonzero(as_tuple=False).reshape(-1)
            new_obs, _ = self.env.reset(options=dict(env_idx=idx))
            new_obs = self._obs(new_obs)
            if new_obs.shape[0] == self.num_envs:
                next_obs = new_obs
            else:  # defensive: some versions return only the reset subset
                next_obs = final_obs.clone()
                next_obs[idx] = new_obs
            self.num_resets += int(idx.numel())
        else:
            next_obs = final_obs

        return next_obs, rew, done, truncated, info, final_obs

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    @property
    def single_action_space(self):
        return self.env.action_space

    @property
    def single_observation_space(self):
        return self.env.observation_space


# ============================================================
# Batched evaluation (one launch for all episodes)
# ============================================================

@torch.no_grad()
def evaluate_vec(agent, env_id, cfg, control_mode, device, num_eps=20,
                 max_episode_steps=None, deterministic=True, tid=0):
    """Evaluate `agent` on `env_id` with all `num_eps` episodes in parallel.

    This is the GPU analogue of `evaluate_task` in the training scripts. The
    single-env CPU loop costs 20 x 100 sequential Sim calls per task, which
    would dominate a GPU run; here it is 100 batched steps.

    Success counts once per episode (`success` latches), exactly like the CPU
    evaluator. The scene is loaded once and object poses are re-randomised at
    every reset: reconfiguration_freq cannot be 1 here, because a batched
    self-resetting env can only do partial resets (see
    `_resolve_reconfiguration_freq`).
    """
    env = GpuVecEnv(env_id, num_envs=num_eps, cfg=cfg, control_mode=control_mode,
                    device=device, max_episode_steps=max_episode_steps)
    horizon = max_episode_steps
    if horizon is None:
        tl = getattr(env.env, "spec", None)
        horizon = getattr(tl, "max_episode_steps", None)
        if horizon is None and hasattr(env.env, "_max_episode_steps"):
            horizon = env.env._max_episode_steps
    if horizon is None:
        horizon = 100
    horizon = int(horizon)

    obs, _ = env.reset()
    success_once = torch.zeros(num_eps, dtype=torch.bool, device=env.device)
    ep_reward = torch.zeros(num_eps, dtype=torch.float32, device=env.device)
    finished = torch.zeros(num_eps, dtype=torch.bool, device=env.device)

    steps = 0
    while not bool(finished.all()) and steps < horizon:
        action = agent.get_action(obs, deterministic=deterministic, tid=tid)
        alive = (~finished).float()
        obs, rew, done, trunc, info, _ = env.step(action)

        succ = info.get("success", None) if isinstance(info, dict) else None
        if succ is not None:
            s = torch.as_tensor(succ, device=env.device).reshape(num_eps).bool()
            success_once |= (s & (~finished))
        ep_reward += rew * alive
        finished |= done
        steps += 1

    sr = float(success_once.float().mean().item() * 100.0)
    avg_r = float(ep_reward.mean().item())
    env.close()
    return sr, avg_r


# ============================================================
# Adapter for the official single-task baseline (ppo_official_unified.py)
# ============================================================

class GpuSingleVec:
    """One CUDA env holding `num_envs` parallel sims, dressed as ManiSkillVecEnv.

    `ppo_official_unified.py` is a near-verbatim copy of the official PPO
    baseline, which expects ManiSkillVectorEnv's contract rather than the
    simpler (obs, rew, done, trunc, info) tuple: it reads
    `infos["final_observation"]`, `infos["_final_info"]` and
    `infos["final_info"]["episode"]["success_once"]`.

    The CPU port reproduced that contract by looping over N single-env sims
    (MultiSingleEnvVector). This class is the GPU twin: same observable
    behaviour, one batched simulator call per step.
    """

    def __init__(self, env_id, num_envs, cfg, control_mode, device="cuda",
                 ignore_terminations=False, record_metrics=True, raw_state=False,
                 reconfiguration_freq=None, max_episode_steps=None, seed=0):
        kwargs = dict(
            num_envs=int(num_envs),
            obs_mode="state" if raw_state else "state_dict",
            control_mode=control_mode,
            sim_backend="gpu",
            # 见 GpuVecEnv：无 Vulkan 的机器上必须关掉渲染
            render_backend="none",
        )
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = int(max_episode_steps)
        reconfiguration_freq = _resolve_reconfiguration_freq(
            reconfiguration_freq, "GpuSingleVec")
        if reconfiguration_freq is not None:
            kwargs["reconfiguration_freq"] = int(reconfiguration_freq)
        self.env = gym.make(env_id, **kwargs)
        self._env = self.env      # for gym_utils.find_max_episode_steps_value
        self.env_id = env_id
        self.cfg = cfg
        self.raw_state = raw_state
        self.ignore_terminations = ignore_terminations
        self.record_metrics = record_metrics
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.spec = getattr(self.env, "spec", None)

        single_act = self.env.get_wrapper_attr("single_action_space")
        self.single_action_space = single_act
        self.action_space = single_act
        self.action_dim = int(np.prod(single_act.shape))

        if raw_state:
            obs_dim = int(np.prod(
                self.env.get_wrapper_attr("single_observation_space").shape))
        else:
            obs_dim = UNIFIED_OBS_DIM
        self.obs_dim = obs_dim
        box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,),
                             dtype=np.float32)
        self.single_observation_space = box
        self.observation_space = box

        spec = self.spec
        self.max_episode_steps = getattr(spec, "max_episode_steps", None) \
            if spec is not None else None
        if self.max_episode_steps is None:
            self.max_episode_steps = int(getattr(self.env, "_max_episode_steps", 100))
        self._alloc_metrics()

    # -- metrics ---------------------------------------------------------
    def _alloc_metrics(self):
        n, dev = self.num_envs, self.device
        self.returns = torch.zeros(n, dtype=torch.float32, device=dev)
        self.success_once = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fail_once = torch.zeros(n, dtype=torch.bool, device=dev)
        self.elapsed = torch.zeros(n, dtype=torch.int32, device=dev)

    def _convert(self, raw_obs):
        if self.raw_state:
            x = raw_obs if torch.is_tensor(raw_obs) else torch.as_tensor(
                raw_obs, device=self.device)
            return x.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)
        return build_unified_obs_batch(raw_obs, self.cfg, device=self.device)

    # -- api -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed)
        self._alloc_metrics()
        return self._convert(obs), {}

    def step(self, action):
        action = to_action_tensor(action, self.device)
        raw_obs, rew, term, trunc, info = self.env.step(action)
        rew = as_tensor(rew, self.device).reshape(-1)
        term = as_tensor(term, self.device).reshape(-1).bool()
        trunc = as_tensor(trunc, self.device).reshape(-1).bool()
        obs_now = self._convert(raw_obs)

        if self.record_metrics:
            self.returns = self.returns + rew
            self.elapsed = self.elapsed + 1
            if isinstance(info, dict) and info.get("success") is not None:
                self.success_once |= as_tensor(
                    info["success"], self.device).reshape(-1).bool()
            if isinstance(info, dict) and info.get("fail") is not None:
                self.fail_once |= as_tensor(
                    info["fail"], self.device).reshape(-1).bool()

        if self.ignore_terminations:
            term = torch.zeros_like(term)
        done = term | trunc

        infos = {}
        if bool(done.any()):
            # Snapshot the terminal obs / episode stats BEFORE the auto-reset,
            # same point as ManiSkillVectorEnv's torch_clone_dict(infos).
            infos["final_observation"] = obs_now
            infos["final_info"] = {"episode": {
                "success_once": self.success_once.clone(),
                "fail_once": self.fail_once.clone(),
                "return": self.returns.clone(),
                "episode_len": self.elapsed.clone(),
                "reward": (self.returns / self.elapsed.clamp(min=1)).clone(),
            }}
            infos["_final_info"] = done.clone()
            infos["_final_observation"] = done.clone()

            idx = done.nonzero(as_tuple=False).reshape(-1)
            new_obs, _ = self.env.reset(options=dict(env_idx=idx))
            new_obs = self._convert(new_obs)
            if new_obs.shape[0] == self.num_envs:
                obs_out = new_obs
            else:  # defensive: only the reset subset came back
                obs_out = obs_now.clone()
                obs_out[idx] = new_obs

            keep = (~done)
            self.returns = torch.where(keep, self.returns, torch.zeros_like(self.returns))
            self.success_once = torch.where(
                keep, self.success_once, torch.zeros_like(self.success_once))
            self.fail_once = torch.where(
                keep, self.fail_once, torch.zeros_like(self.fail_once))
            self.elapsed = torch.where(keep, self.elapsed, torch.zeros_like(self.elapsed))
        else:
            obs_out = obs_now

        return obs_out, rew, term, trunc, infos

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
