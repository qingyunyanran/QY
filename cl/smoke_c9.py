# -*- coding: utf-8 -*-
"""Integration smoke test on the REAL 6-head checkpoint.

Confirms that the trained c6v4 run can be continued into a 9-head run and that
an appended task's expert only touches its own head.
"""
import torch

from ppo_er_unified import (Agent, UNIFIED_OBS_DIM, expand_state_dict_load,
                            load_task_expert)

CK = "ckpts/ppo_er_unified/"
BASE = CK + "ppo_er_unified_task0_PegInsertionSide-v1_c6v4_refit_best.pt"

sd = torch.load(BASE, map_location="cpu")
heads = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor_heads_extra")})
print(f"ckpt {BASE.split('/')[-1]}: extra heads {heads}, {len(sd)} keys")

agent = Agent(UNIFIED_OBS_DIM, 8, n_heads=9, private_base_ids=(0,))
kept, fresh, missing = expand_state_dict_load(agent, sd, verbose=True,
                                             source=BASE.split("/")[-1])
print(f"kept={sorted(kept)} fresh={sorted(fresh)} missing={missing}")

# --- append the PokeCube expert as task 6 (head index 5) -------------------
ex_path = CK + "ppo_er_unified_task0_PokeCube-v1_ex48c_PokeCube_best.pt"
raw = torch.load(ex_path, map_location="cpu")
ex = {k[len("actor_mean."):]: v for k, v in raw.items() if k.startswith("actor_mean.")}

x = torch.randn(2, UNIFIED_OBS_DIM)
with torch.no_grad():
    base_before = agent.actor_mean(x).clone()
    feat_before = agent.actor_mean[:Agent.SHARED_BASE_LAYERS](x).clone()

load_task_expert(agent, 6, ex, include_base=False)

with torch.no_grad():
    base_after = agent.actor_mean(x)
    # heads consume the 256-dim TRUNK feature, not the 48-dim observation
    feat = agent.actor_mean[:Agent.SHARED_BASE_LAYERS](x)
    head6 = agent.actor_heads_extra[5](feat)

ref = ex["0.weight"] @ x.t() + ex["0.bias"][:, None]
ref = ex["2.weight"] @ torch.tanh(ref) + ex["2.bias"][:, None]
feat_ex = torch.tanh(ref).t()                      # expert's own 256-dim feature
ref = ex["4.weight"] @ torch.tanh(ref) + ex["4.bias"][:, None]
ref = (ex["6.weight"] @ torch.tanh(ref) + ex["6.bias"][:, None]).t()

with torch.no_grad():
    head6_on_expert_feat = agent.actor_heads_extra[5](feat_ex)
    head6_on_cl_base = head6

ok_base = torch.equal(base_before, base_after)
ok_feat = torch.equal(feat_before, feat)
ok_head = torch.allclose(head6_on_expert_feat, ref, atol=1e-5)
drift = (head6_on_cl_base - ref).abs().max().item()
print(f"shared base untouched by the expert load : {ok_base}")
print(f"trunk features identical                 : {ok_feat}")
print(f"head 6 == expert head, same input space  : {ok_head}")
print(f"expert head on the CL base's features    : max|diff|={drift:.3f} "
      f"(expected != 0: the head was trained on the EXPERT base's features, "
      f"so appending gives a calibrated-ish init, not the exact expert)")
print("OK" if (ok_base and ok_feat and ok_head
              and kept == {0, 1, 2, 3, 4} and fresh == {5, 6, 7})
      else "FAIL")
