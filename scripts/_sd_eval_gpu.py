# -*- coding: utf-8 -*-
"""Evaluate the single-head, task-label-free student on the GPU backend.

Same batched batched protocol as vec_env.evaluate_vec -- which is the protocol
every number in docs/CL_10TASK_RESULTS.md was measured under -- plus a real
STACK-frame history buffer, because the student is trained on a history window
and must be evaluated on the same input.

The CPU path is deliberately not used: PegInsertionSide scores 15% there and
75-90% on GPU for the identical checkpoint (CPU rebuilds the scene on every
reset and PhysX CPU disagrees with PhysX GPU on a sub-millimetre insertion).
"""
import os

import numpy as np
import torch

from ppo_er_unified import ACTION_LIMIT, Agent, TASK_INFO
from unified_obs import SLOT_LAYOUT, UNIFIED_OBS_DIM
from vec_env import GpuVecEnv, to_action_tensor

CKPT = os.environ.get("SD_EVAL_CKPT", "ckpts/sd_single.pt")
ALL_TASKS = ["PegInsertionSide-v1", "StackCube-v1", "PullCube-v1",
             "PickCube-v1", "LiftPegUpright-v1", "PushCube-v1",
             "PokeCube-v1", "PullCubeTool-v1",
             "StackPyramid-v1", "PlugCharger-v1", "PlaceSphere-v1"]
_sel = [t for t in os.environ.get("SD_TASKS", "").split(",") if t]
TASKS = [t for t in ALL_TASKS if (not _sel or t in _sel)]
EPS = int(os.environ.get("SD_EVAL_EPS", "100"))
STACK = int(os.environ.get("SD_STACK", "3"))
ACT_DIM = 8
REF = os.environ.get("SD_REF", "")  # multi-head reference numbers, "91,90,..."
# StackPyramid's expert needs ~250 steps and its own task_info horizon is tight,
# so evaluate it with the 400 steps used for the documented number. The closed
# loop IK dataset (demos/sd_pyr_d.npz) records one frame per IK sub-step (~1200
# per episode), so it must be evaluated with a matching horizon; pass
# SD_PYR_HZ=1500 for that checkpoint, otherwise the rollout is truncated.
HZ = {"StackPyramid-v1": int(os.environ.get("SD_PYR_HZ", "400")),
      # The Plug expert itself needs ~164 steps and the student is slower, so a
      # 200-step cap fails every episode by timeout (SR 0.0%, steps=200).
      "PlugCharger-v1": int(os.environ.get("SD_PLUG_HZ", "400")),
      # PlaceSphere's native horizon is 50 but the official solver needs ~450
      # steps (reach, grasp, lift, carry, 1 mm pose refinement, mid-air release,
      # settle); the student is no faster, so 50 truncates every episode.
      "PlaceSphere-v1": int(os.environ.get("SD_PS_HZ", "800"))}

sd = torch.load(CKPT, map_location="cpu")
# SD_NORM checkpoints carry the standardisation stats next to the weights; they
# must be applied here exactly as in training, or the policy sees a different
# input scale than it was fitted on.
NORM_STATS = sd.get("norm") if isinstance(sd, dict) else None
if NORM_STATS is not None:
    NORM_STATS = {k: torch.as_tensor(v, dtype=torch.float32)
                  for k, v in NORM_STATS.items()}
# Dual-scale checkpoints see [raw, standardised] side by side, so the recorded
# width is twice the real per-frame width and both copies must be rebuilt here.
DUAL = int(float(NORM_STATS["dual"])) if (NORM_STATS is not None
                                          and "dual" in NORM_STATS) else 0
if isinstance(sd, dict) and "actor_mean.0.weight" not in sd:
    for k in ("state_dict", "agent", "model"):
        if isinstance(sd.get(k), dict):
            sd = sd[k]
            break
# The student is built from the checkpoint itself: PlugCharger ships 66 obs dims
# while StackPyramid ships 68 (its two derived scalars), so the width is not a
# constant any more. W_IN is the flattened frame-stack width.
W_IN = int(sd["actor_mean.0.weight"].shape[1])
HIDDEN = int(sd["actor_mean.0.weight"].shape[0])
assert W_IN % STACK == 0, f"obs width {W_IN} not divisible by STACK={STACK}"
TGT = W_IN // STACK // (2 if DUAL else 1)
agent = Agent(W_IN, ACT_DIM, n_heads=1, hidden=HIDDEN)
missing, unexpected = agent.load_state_dict(sd, strict=False)
print(f"== {CKPT}\n   obs_dim={W_IN} per_frame={TGT} stack={STACK} hidden={HIDDEN} "
      f"missing={list(missing)} unexpected={list(unexpected)}")
agent = agent.cuda().eval()


def _fit(o):
    """Zero-pad the per-frame obs to the width the checkpoint was trained on.

    The two extra StackPyramid scalars (grip_open, holding) are exactly 0 for
    every other task, so padding is semantically exact, not an approximation.
    """
    if o.shape[1] < TGT:
        return torch.cat([o, torch.zeros(o.shape[0], TGT - o.shape[1],
                                        device=o.device)], 1)
    return o


srs = []
for env_id in TASKS:
    ti = TASK_INFO[env_id]
    cfg = SLOT_LAYOUT[env_id]
    alim = (ACTION_LIMIT if ti.get("action_limit") is None
            else float(ti["action_limit"]))
    horizon = HZ.get(env_id) or ti.get("max_episode_steps") or 100
    env = GpuVecEnv(env_id, num_envs=EPS, cfg=cfg,
                    control_mode=ti["control_mode"], device="cuda",
                    max_episode_steps=horizon, action_limit=alim)
    obs, _ = env.reset()
    obs = _fit(obs)
    hist = [obs.clone() for _ in range(STACK)]
    succ_once = torch.zeros(EPS, dtype=torch.bool, device=env.device)
    finished = torch.zeros(EPS, dtype=torch.bool, device=env.device)
    steps = 0
    with torch.no_grad():
        while not bool(finished.all()) and steps < int(horizon):
            x = torch.cat(hist, dim=1)
            if NORM_STATS is not None:
                xn = ((x - NORM_STATS["x_mu"].to(x.device))
                      / NORM_STATS["x_sd"].to(x.device))
                x = torch.cat([x, xn], dim=1) if DUAL else xn
            a = agent.get_action(x, deterministic=True)
            if NORM_STATS is not None:
                a = (a * NORM_STATS["y_sd"].to(a.device)
                     + NORM_STATS["y_mu"].to(a.device))
            a = to_action_tensor(a, env.device, alim)
            obs, rew, done, trunc, info, _final = env.step(a)
            s = info.get("success", None) if isinstance(info, dict) else None
            if s is not None:
                sv = torch.as_tensor(s, device=env.device).reshape(-1).bool()
                succ_once |= (sv & (~finished))
            finished |= done
            obs = _fit(obs)
            hist = hist[1:] + [obs]
            steps += 1
    env.close()
    sr = float(succ_once.float().mean().item() * 100.0)
    srs.append(sr)
    print(f"   {env_id[:-3]:<16s} SR={sr:5.1f}%  steps={steps}", flush=True)

print(f"   mean={np.mean(srs):.1f}%  min={min(srs):.1f}%  eps/task={EPS}  "
      f"all>=80%: {all(x >= 80.0 for x in srs)}")
if REF:
    ref = [float(x) for x in REF.split(",")]
    if len(ref) == len(srs):
        print("   delta vs multi-head:", " ".join(
            f"{a-b:+.1f}" for a, b in zip(srs, ref)),
            f" (mean {np.mean(srs)-np.mean(ref):+.1f})")


