# -*- coding: utf-8 -*-
"""Diff the 6-task checkpoint against the 8-task one, tensor by tensor.

The frozen-base append should leave every shared tensor untouched, so any
difference found here is exactly what the two new tasks changed -- that tells us
whether PickCube's drop comes from the base, its own adapter, or the shared
exploration noise (actor_logstd).
"""
import torch

CK = "ckpts/ppo_er_unified/"
A = CK + "ppo_er_unified_task0_PegInsertionSide-v1_c6v4_refit_best.pt"
B = CK + "ppo_er_unified_task7_PullCubeTool-v1_c8v2_ad.pt"

a = torch.load(A, map_location="cpu")
b = torch.load(B, map_location="cpu")
print(f"A: {A.split('/')[-1]}  {len(a)} keys")
print(f"B: {B.split('/')[-1]}  {len(b)} keys")

same, diff, only_a, only_b = [], [], [], []
for k in sorted(set(a) | set(b)):
    if k not in a:
        only_b.append(k)
    elif k not in b:
        only_a.append(k)
    elif a[k].shape != b[k].shape:
        diff.append((k, "shape", tuple(a[k].shape), tuple(b[k].shape)))
    elif torch.equal(a[k], b[k]):
        same.append(k)
    else:
        d = (a[k] - b[k]).abs().max().item()
        diff.append((k, f"maxdiff={d:.4g}", "", ""))

print(f"\nidentical: {len(same)}   changed: {len(diff)}   "
      f"only in A: {len(only_a)}   only in B: {len(only_b)}")
print("\n--- changed tensors ---")
for k, why, s1, s2 in diff:
    print(f"  {k:34s} {why} {s1} {s2}")
print("\n--- only in A ---", only_a[:10])
print("--- only in B ---", only_b)

if "actor_logstd" in a and "actor_logstd" in b:
    print("\nactor_logstd A:", a["actor_logstd"].flatten().tolist())
    print("actor_logstd B:", b["actor_logstd"].flatten().tolist())
