# -*- coding: utf-8 -*-
"""Train ONE policy for all tasks -- no task label, no per-task head.

Student: a plain single-head actor (Agent with n_heads=1) that maps a
3-frame stack of the unified 66-dim state (198 dims) straight to the 8-dim
joint-delta action. Task identity is NOT an input: the only way it can tell
StackCube from PickCube is what the state history looks like.

Why 3 frames: the unified state is a single snapshot, so "approaching the goal"
and "leaving the goal" are the same vector (documented gap in section 4 of
CL_10TASK_RESULTS.md). Two extra frames make the arm's direction of travel
observable, which is what the single head needs to disambiguate phases.

Loss is Gaussian NLL on the expert action (equivalent to MSE when logstd is
frozen, but lets the net widen its own std in genuinely ambiguous states, where
the multi-head expert had the answer and this student does not).

Data is re-sampled with per-task balancing: the 8 tasks contribute equal mass to
every batch, otherwise the tasks with longer episodes would dominate.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from ppo_er_unified import Agent

DATA = os.environ.get("SD_DATA", "demos/sd_8task.npz")
STACK = int(os.environ.get("SD_STACK", "3"))
EPOCHS = int(os.environ.get("SD_EPOCHS", "200"))
BS = int(os.environ.get("SD_BS", "2048"))
LR = float(os.environ.get("SD_LR", "3e-4"))
OBNOISE = float(os.environ.get("SD_OBNOISE", "0"))
OUT = os.environ.get("SD_OUT_CKPT", "ckpts/sd_single.pt")
NLL = os.environ.get("SD_NLL", "1") == "1"
# L1 (median regression) resists mean-collapse: the frame distribution is dominated
# by near-zero teacher actions, so a squared loss is minimised by predicting ~0 and
# the long tail of large corrective actions gets averaged away.
L1 = os.environ.get("SD_L1", "0") == "1"
ACT_DIM = 8

# SD_DATA may list several files comma-separated; DAgger rounds are appended to
# the BC set this way. Episode ids are shifted per file so a frame stack never
# spans two files.
_parts = [np.load(p.strip()) for p in DATA.split(",") if p.strip()]
# Older collections were captured at 66 dims, before the two derived scalars
# (grip opening / holding flag) were appended at [66:68]. Training a mixed set
# is legitimate: those two dims are zeros for every task except StackPyramid, so
# padding them with 0 means exactly "fingers open, holding nothing".
TGT_DIM = 48 + int(os.environ.get("UO_EXTRA_N", "0"))
_obs, _act, _tid, _ep, _w = [], [], [], [], []
_off = 0
for _p in _parts:
    _o = _p["obs"]
    _w.append(np.full(_o.shape[0], _o.shape[1], dtype=np.int64))
    if _o.shape[1] < TGT_DIM:
        print(f"pad {_p.filename if hasattr(_p, 'filename') else '?'}: "
              f"{_o.shape[1]} -> {TGT_DIM} zero cols")
        _o = np.pad(_o, ((0, 0), (0, TGT_DIM - _o.shape[1])))
    assert _o.shape[1] == TGT_DIM, f"obs width {_o.shape[1]} != {TGT_DIM}"
    _obs.append(_o)
    _act.append(_p["act"])
    _tid.append(_p["tid"])
    _ep.append(_p["ep"].astype(np.int64) + _off)
    _off += int(_p["ep"].max()) + 1
obs = np.concatenate(_obs)
act = np.concatenate(_act)
tid = np.concatenate(_tid)
ep = np.concatenate(_ep)

# SD_ONLY_TID=t trains the same single head on ONE task only. It is the control
# that separates "single-head BC is broken" from "one head cannot serve 8 tasks":
# if the control reaches the expert's own success rate, the pipeline is fine and
# the loss is task interference.
ONLY = os.environ.get("SD_ONLY_TID", "").strip()
if ONLY:
    keep = tid == int(ONLY)
    obs, act, ep = obs[keep], act[keep], ep[keep]
    tid = np.zeros(len(obs), dtype=np.int64)
    ep = np.unique(ep, return_inverse=True)[1].astype(np.int64)
    print(f"SD_ONLY_TID={ONLY}: single-task control, {len(obs)} frames")
print(f"loaded {DATA}: frames={len(obs)} eps={len(np.unique(ep))} obs_dim={obs.shape[1]}")
print("per-task frames:", {int(t): int((tid == t).sum()) for t in np.unique(tid)})

# ---- build the frame stack per episode (pad the first frames by repetition) --
xs = np.zeros((len(obs), obs.shape[1] * STACK), dtype=np.float32)
order = np.argsort(ep, kind="stable")
ep_sorted = ep[order]
starts = np.r_[0, np.flatnonzero(np.diff(ep_sorted)) + 1, len(ep_sorted)]
obs_o = obs[order]
for a, b in zip(starts[:-1], starts[1:]):
    seg = obs_o[a:b]
    n = len(seg)
    # Vectorised stack: pad the head with repeats of frame 0, then take the
    # sliding window [i-STACK+1 .. i] for every i at once. Identical to the
    # naive per-frame loop but ~100x faster on 700k+ frames.
    pad = np.repeat(seg[:1], STACK - 1, 0)
    win = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad, seg], 0), STACK, axis=0
    )                                  # (n, D, STACK) with time on the last axis
    xs[order[a:b]] = win.transpose(0, 2, 1).reshape(n, -1)
print(f"stacked -> {xs.shape}, {STACK} frames")

X = torch.from_numpy(xs)
# SD_DUAL=1 feeds the encoder BOTH the raw state and its standardised copy.
# The two task families need opposite scales (8-task checkpoints reach Peg 83%
# only on raw inputs, PlaceSphere needs standardised ones), and a single x_sd
# cannot serve both: with one scale Peg drops to 38-62% while PlaceSphere drops
# to 26% on the other. Giving the encoder both lets one head serve both without
# any task label; the width doubles but the policy stays single.
DUAL = os.environ.get("SD_DUAL", "0") == "1"
_X_RAW = X.clone() if DUAL else None
Y = torch.from_numpy(act.astype(np.float32))
# SD_NORM=1 standardises inputs and regression targets with training-set stats.
# qpos spans +-3 rad while the task-relevant relative vectors are ~0.01-0.1 m, so
# an unnormalised MLP lets the large proprioceptive dims dominate the gradient;
# the resulting ~0.09-rad single-step error is invisible in MSE yet accumulates
# into 18 cm of tcp overshoot in closed loop. Stats travel with the checkpoint.
NORM = os.environ.get("SD_NORM", "0") == "1"
# SD_INIT warm-starts from an existing checkpoint instead of from scratch.
# That is the only sane way to run a DAgger round: retraining the same head on
# the same expert data from random init already lands 6x below the checkpoint
# it is supposed to improve on, so a fresh run cannot be compared with it.
INIT = os.environ.get("SD_INIT", "").strip()
_INIT = torch.load(INIT, map_location="cpu") if INIT else None
# Checkpoints come in two shapes: {"state_dict": ..., "norm": ...} (written by
# this script) and the bare state_dict produced by the widening helpers
# (_ps_up56.py / _ps_dual.py, e.g. sd_single_d5_p56.pt). Treating the bare form
# as "no weights" silently started every warm start from random init, which made
# the mixed runs look like catastrophic forgetting.
if isinstance(_INIT, dict) and "state_dict" in _INIT:
    _INIT_SD = _INIT["state_dict"]
elif isinstance(_INIT, dict):
    _INIT_SD = {k: v for k, v in _INIT.items() if torch.is_tensor(v)} or None
else:
    _INIT_SD = _INIT
_INIT_STATS = _INIT.get("norm") if isinstance(_INIT, dict) else None
if NORM:
    # SD_NORM_FROM_DATA=1 re-estimates the statistics on the current mix even
    # when SD_INIT carries its own: warm starting from an 8-task checkpoint whose
    # last dims are zero-padded leaves those columns with sd=1e-3, which blows the
    # PlaceSphere-only dims up ~1000x and makes closed loop diverge (SR 0.0%).
    if _INIT_STATS is not None and os.environ.get("SD_NORM_FROM_DATA", "0") != "1":
        # Reuse the source statistics. Re-estimating them on a data mix that
        # now includes DAgger frames would rescale every input dim and throw
        # the warm start away.
        x_mu = torch.as_tensor(_INIT_STATS["x_mu"], dtype=torch.float32)
        x_sd = torch.as_tensor(_INIT_STATS["x_sd"], dtype=torch.float32)
        y_mu = torch.as_tensor(_INIT_STATS["y_mu"], dtype=torch.float32)
        y_sd = torch.as_tensor(_INIT_STATS["y_sd"], dtype=torch.float32)
        print("norm: reused from SD_INIT")
    else:
        # Per-column statistics over the frames where the column actually
        # exists. Zero-padded tail dims (48 -> 56) are excluded from their own
        # column's mean/std instead of being counted as real zeros, which would
        # halve the estimated sd and over-amplify that dim at inference time.
        _has = np.arange(TGT_DIM)[None, :] < np.concatenate(_w)[:, None]
        _has = np.tile(_has, (1, STACK))
        # Share the first TGT_DIM-UO_EXTRA_N (48) dims between task families by
        # statistics of the 8-task frames only, and let the padded tail be
        # estimated from the PlaceSphere frames. Fitting the shared dims on the
        # mixed set rescales every 8-task input (x_sd 0.108 -> 0.167) and drops
        # PegInsertionSide 83% -> 51% even though train MSE improves.
        _up = os.environ.get("SD_UP_TID", "").strip()
        if _up:
            _ps = (tid == int(_up))[:, None]
            _base = (_has & ~_ps)
            _has = np.where(_base.any(0, keepdims=True), _base, _has)
        # SD_NORM_TAIL_ONLY=1 leaves the shared 48 dims at identity scale and
        # standardises only the PlaceSphere-only tail. The 8-task models were
        # trained without normalisation (sd_single_d5: mean 92.4%, min 83); any
        # data-driven rescaling of the shared dims costs PegInsertionSide ~35 pt
        # (measured: 92.4% -> 51-62% for the same 8-task data), so the shared
        # dims must stay untouched and only the new family may be rescaled.
        _tail_only = os.environ.get("SD_NORM_TAIL_ONLY", "0") == "1"
        if _tail_only:
            _shared = np.tile(np.arange(TGT_DIM) < (TGT_DIM - int(os.environ.get("UO_EXTRA_N", "0"))), STACK)
            _has[:, _shared] = False
        _D = _has.shape[1]
        _cnt = _has.sum(0).astype(np.float64)
        _live = _cnt > 0
        _xs = np.asarray(xs, dtype=np.float64)
        _mu = np.zeros(_D, dtype=np.float64)
        _sd = np.ones(_D, dtype=np.float64)
        _mu[_live] = (_xs * _has).sum(0)[_live] / _cnt[_live]
        _sd[_live] = np.sqrt((((_xs - _mu) ** 2) * _has).sum(0)[_live] / _cnt[_live])
        x_mu = torch.as_tensor(_mu[None, :], dtype=torch.float32)
        x_sd = torch.as_tensor(_sd[None, :], dtype=torch.float32).clamp_min(1e-3)
        if _tail_only:
            # The recorded 8-task model regressed raw actions in [-1,1]; a
            # mixed-set y_sd is ~1.4x larger and costs resolution on the fine
            # insertion tasks, so keep the action scale at identity too.
            y_mu = torch.zeros(1, ACT_DIM)
            y_sd = torch.ones(1, ACT_DIM)
        else:
            y_mu = Y.mean(0, keepdim=True)
            y_sd = Y.std(0, keepdim=True).clamp_min(1e-3)
    X = (X - x_mu) / x_sd
    Y = (Y - y_mu) / y_sd
    if DUAL:
        assert NORM, "SD_DUAL needs SD_NORM=1"
        X = torch.cat([_X_RAW, X], dim=1)
        print(f"dual-scale input: {X.shape[1]} dims "
              f"({_X_RAW.shape[1]} raw + {_X_RAW.shape[1]} standardised)")
    print(f"norm on: x_mu|x|mean={float(x_mu.abs().mean()):.3f} "
          f"x_sd|mean={float(x_sd.mean()):.3f} y_sd={np.round(y_sd[0].tolist(), 4)}")
T = torch.from_numpy(tid.astype(np.int64))
n_tasks = int(tid.max()) + 1
per_task = [np.flatnonzero(tid == t) for t in range(n_tasks)]
# Keep only tasks that actually have data: a subset run (e.g. 9 of 10 tasks)
# leaves a hole at the missing task id and sampling would otherwise index an
# empty array. Sampling is per task, so this also equalises the per-task weight
# regardless of how many frames each task contributes.
TASK_IDS = [t for t in range(n_tasks) if len(per_task[t]) > 0]
per_task = [per_task[t] for t in TASK_IDS]
n_tasks = len(per_task)

dev = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN = int(os.environ.get("SD_HIDDEN", "256"))
agent = Agent(int(X.shape[1]), ACT_DIM, n_heads=1, hidden=HIDDEN).to(dev)
if _INIT_SD is not None:
    missing, unexpected = agent.load_state_dict(_INIT_SD, strict=False)
    print(f"warm start from {INIT}: missing={len(missing)} unexpected={len(unexpected)}")
logstd = agent.actor_logstd
opt = torch.optim.Adam(agent.actor_mean.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
# ---- EWC / MAS -------------------------------------------------------------
# Mixing PlaceSphere into a converged 8-task student collapses the fine tasks
# (Peg 87% -> 1-4%) even at lr 3e-5 x 40 epochs: the policy sits in a sharp
# minimum and the shared trunk drifts on the first steps. SD_EWC_FIT=1 records
# theta* plus a diagonal importance computed on the 8-task frames only;
# SD_EWC=<file> then adds lam/2 * sum I_i (theta_i - theta*_i)^2 to the loss.
EWC_FIT = os.environ.get("SD_EWC_FIT", "0") == "1"
EWC_PATH = os.environ.get("SD_EWC", "").strip()
EWC_LAM = float(os.environ.get("SD_EWC_LAMBDA", "0"))
if EWC_FIT:
    _ps_tid = int(os.environ.get("SD_UP_TID", "-1"))
    _keep = np.flatnonzero(tid != _ps_tid) if _ps_tid >= 0 else np.arange(len(tid))
    if len(_keep) > 20000:
        _keep = np.random.choice(_keep, 20000, replace=False)
    _params = dict(agent.actor_mean.named_parameters())
    _imp = {n: torch.zeros_like(p) for n, p in _params.items()}
    agent.train()
    _nb = 0
    for _s in range(0, len(_keep), 512):
        _idx = torch.from_numpy(_keep[_s:_s + 512].astype(np.int64))
        _l = ((agent.actor_mean(X[_idx].to(dev)) - Y[_idx].to(dev)) ** 2).mean()
        agent.actor_mean.zero_grad(set_to_none=True)
        _l.backward()
        for n, p in _params.items():
            if p.grad is not None:
                _imp[n] += p.grad.detach() ** 2
        _nb += 1
    # Normalise the importance so the penalty is dimensionless and lam is
    # comparable across runs; raw grad^2 values span many orders of magnitude.
    _flat = torch.cat([v.reshape(-1) for v in _imp.values()])
    _scale = float(_flat.mean().clamp_min(1e-12))
    _imp = {n: v / _scale for n, v in _imp.items()}
    _out = os.environ.get("SD_EWC_OUT", "ckpts/ewc.pt")
    torch.save({"theta": {n: p.detach().clone() for n, p in _params.items()},
                "imp": _imp}, _out)
    print(f"ewc: fitted on {len(_keep)} frames -> {_out} (scale={_scale:.3e})")
    sys.exit(0)
if EWC_PATH:
    _e = torch.load(EWC_PATH, map_location=dev)
    _theta0 = _e["theta"]
    _imp0 = _e["imp"]
    print(f"ewc: lam={EWC_LAM} from {EWC_PATH}")
per_batch = max(1, BS // n_tasks)
# SD_UP_TID / SD_UP raise one task's share of every batch. Per-task equal
# sampling caps PlaceSphere at 227/2048 frames per batch and its fit stalls at
# mse 0.042 (SR 37%) while the same data alone reaches 0.030 (SR 83%), so the
# task needs a bigger slice rather than more capacity.
_UP_TID = os.environ.get("SD_UP_TID", "").strip()
_UP = int(os.environ.get("SD_UP", "1"))
# SD_UP_W="10:5,0:4,1:4" sets several per-task shares at once; the fine tasks
# (Peg/Stack) need both more frames and a larger share of the shared trunk.
_wmap = {}
for _kv in os.environ.get("SD_UP_W", "").split(","):
    if ":" in _kv:
        _k, _v = _kv.split(":")
        _wmap[int(_k)] = int(_v)
_wts = np.array([_wmap.get(int(t), _UP if (str(t) == _UP_TID and _UP > 1) else 1)
                 for t in TASK_IDS])
_unit = max(1, BS // int(_wts.sum()))
per_task_bs = [int(_unit * w) for w in _wts]

print(f"train: {EPOCHS} epochs, batch {BS} ({per_task_bs}/task), nll={NLL}, dev={dev}")
t0 = time.time()
NB = max(1, len(X) // BS)   # real epochs: one pass over the data per epoch
print(f"train: {EPOCHS} epochs x {NB} batches of {BS}, nll={NLL}, dev={dev}")
t0 = time.time()
for ep_i in range(EPOCHS):
    agent.train()
    for _ in range(NB):
        idxs = []
        for t in range(n_tasks):
            pick = np.random.choice(per_task[t], size=per_task_bs[t], replace=True)
            idxs.append(pick)
        idx = torch.from_numpy(np.concatenate(idxs))
        xb, yb = X[idx].to(dev), Y[idx].to(dev)
        if OBNOISE > 0.0:
            # Observation noise is a cheap stand-in for DAgger. The expert data
            # only ever contains exact expert states, so the head extrapolates
            # the moment its own closed-loop error drifts a few millimetres off
            # the recorded manifold. Perturbing the input by a fraction of its
            # own per-dimension spread keeps small errors on-distribution.
            # Measured on PlaceSphere: raising capacity alone moved SR 18% -> 8%
            # while halving train MSE, i.e. the failure is drift, not fit.
            xb = xb + torch.randn_like(xb) * (xb.std(dim=0, keepdim=True) * OBNOISE)
        mu = agent.actor_mean(xb)
        if L1:
            loss = (mu - yb).abs().mean()
        elif NLL:
            std = torch.exp(logstd.expand_as(mu)).clamp(0.02, 1.0)
            loss = (0.5 * ((mu - yb) / std) ** 2 + torch.log(std)).mean()
        else:
            loss = ((mu - yb) ** 2).mean()
        if EWC_PATH and EWC_LAM > 0:
            # Quadratic anchor on the 8-task solution; importance is normalised
            # in the fit pass so lam ~ O(1) is a meaningful prior strength.
            pen = mu.new_zeros(())
            for _n, _p in agent.actor_mean.named_parameters():
                pen = pen + (_imp0[_n] * (_p - _theta0[_n]) ** 2).sum()
            loss = loss + EWC_LAM * pen
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.actor_mean.parameters(), 1.0)
        opt.step()
    sched.step()

    if (ep_i + 1) % 20 == 0 or ep_i == 0:
        with torch.no_grad():
            per_loss = []
            for t in range(n_tasks):
                sub = torch.from_numpy(per_task[t][:4096])
                m = agent.actor_mean(X[sub].to(dev))
                per_loss.append(float(((m - Y[sub].to(dev)) ** 2).mean()))
        msg = " ".join(f"{v:.4f}" for v in per_loss)
        print(f"  ep{ep_i+1:03d} loss={float(loss):.5f} mse/task=[{msg}] "
              f"std={float(torch.exp(logstd).mean()):.3f} {time.time()-t0:.0f}s")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
if NORM:
    torch.save({"state_dict": agent.state_dict(),
                "norm": {"x_mu": x_mu, "x_sd": x_sd,
                         "y_mu": y_mu, "y_sd": y_sd,
                         "dual": int(DUAL)}}, OUT)
else:
    torch.save(agent.state_dict(), OUT)
torch.save({"state_dict": agent.state_dict(),
            "meta": {"stack": STACK, "obs_dim": int(xs.shape[1]),
                     "tasks_source": DATA, "nll": NLL}}, OUT + ".meta.pt")
print(f"saved {OUT}  obs_dim={xs.shape[1]}")
