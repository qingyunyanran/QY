# -*- coding: utf-8 -*-
"""Independent re-evaluation of saved CL checkpoints (fresh process, fresh sims).

Deliberately does NOT reuse any number from the training logs: it loads the
saved .pt, rebuilds the envs and runs a longer 50-episode evaluation per task.
"""
import argparse
import torch
import numpy as np
from ppo_er_unified import Agent, evaluate_all_tasks, UNIFIED_OBS_DIM

TASKS = ["PushCube-v1", "PullCube-v1", "PickCube-v1",
         "LiftPegUpright-v1", "StackCube-v1"]
SHORT = ["PushCube", "PullCube", "PickCube", "LiftPeg", "StackCube"]

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--eps", type=int, default=50)
ap.add_argument("--action-dim", type=int, default=8)
args = ap.parse_args()

for ck in args.ckpts:
    sd = torch.load(ck, map_location="cpu")
    if isinstance(sd, dict) and "actor_mean.0.weight" not in sd:
        for k in ("agent", "model", "state_dict"):
            if isinstance(sd.get(k), dict):
                sd = sd[k]
                break
    agent = Agent(UNIFIED_OBS_DIM, args.action_dim)
    missing, unexpected = agent.load_state_dict(sd, strict=False)
    print(f"\n== {ck}\n   load: missing={list(missing)} unexpected={list(unexpected)}")
    agent = agent.cuda().eval()
    res = evaluate_all_tasks(agent, TASKS, num_eps=args.eps,
                            backend="gpu", device="cuda")
    srs = []
    for t, s in zip(SHORT, TASKS):
        r = res[s]
        srs.append(r["success_rate"])
        print(f"   {t:<10s} SR={r['success_rate']:5.1f}%  avg_R={r['avg_reward']:7.2f}")
    print(f"   mean SR = {np.mean(srs):.1f}%  (eps/task={args.eps})")
