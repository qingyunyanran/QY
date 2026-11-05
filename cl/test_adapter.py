import torch
from ppo_er_unified import Agent, UNIFIED_OBS_DIM

A, OBS = 8, UNIFIED_OBS_DIM

# 1) single-head stays byte-compatible with the historical layout
a1 = Agent(OBS, A)
assert [type(m).__name__ for m in a1.actor_mean] == \
    ["Linear", "Tanh", "Linear", "Tanh", "Linear", "Tanh", "Linear"], a1.actor_mean
assert a1._trunk is None
assert a1.head_mean(torch.randn(4, OBS)).shape == (4, A)

# 2) multi-head: base shared, adapter private, head 0 keeps actor_mean keys
a3 = Agent(OBS, A, n_heads=3)
sd = a3.state_dict()
assert "actor_mean.0.weight" in sd and "actor_heads_extra.0.0.weight" in sd
assert len(a3._trunk) == Agent.SHARED_BASE_LAYERS == 4
assert len(a3.actor_heads_extra[0]) == 3

x = torch.randn(5, OBS)
outs = [a3.head_mean(x, t) for t in range(3)]
assert all(o.shape == (5, A) for o in outs)
assert not torch.allclose(outs[0], outs[1])
tid = torch.tensor([0, 1, 2, 1, 0])
mixed = a3.head_mean(x, tid)
for row, t in enumerate(tid.tolist()):
    assert torch.allclose(mixed[row], outs[t][row], atol=1e-6)

# 3) a single-head teacher reproduces exactly in task 0
teacher = Agent(OBS, A)
t_actor = {k[len("actor_mean."):]: v for k, v in teacher.state_dict().items()
           if k.startswith("actor_mean.")}
a0 = Agent(OBS, A, n_heads=3)
a0.actor_mean.load_state_dict(t_actor, strict=True)
with torch.no_grad():
    assert torch.allclose(a0.head_mean(x, 0), teacher.actor_mean(x), atol=1e-6)

# 4) teacher -> task k adapter remap (the warm start for non-outlier tasks)
nb = Agent.SHARED_BASE_LAYERS
body = {k: v for k, v in t_actor.items() if int(k.split(".")[0]) < nb}
adapt = {str(int(k.split(".")[0]) - nb) + k[k.index("."):]: v
         for k, v in t_actor.items() if int(k.split(".")[0]) >= nb}
a0.actor_mean.load_state_dict(body, strict=False)
a0.actor_heads_extra[1].load_state_dict(adapt, strict=True)
with torch.no_grad():
    for t in (0, 2):
        assert torch.allclose(a0.head_mean(x, t), teacher.actor_mean(x), atol=1e-6), t

# 5) private base for an outlier task: whole net, exact teacher load, and the
#    trunk optimiser group must not contain it (it moves at the full lr)
a4 = Agent(OBS, A, n_heads=3, private_base_ids=(0,))
assert "private_nets.0.0.weight" in a4.state_dict()
assert a4.head_mean(x, 0).shape == (5, A)
a4.private_nets["0"].load_state_dict(t_actor, strict=True)
with torch.no_grad():
    assert torch.allclose(a4.head_mean(x, 0), teacher.actor_mean(x), atol=1e-6)
    assert not torch.allclose(a4.head_mean(x, 1), teacher.actor_mean(x))
trunk_ids = {id(p) for p in a4.trunk_parameters()}
assert not any(id(p) in trunk_ids for p in a4.private_nets.parameters())
mixed = a4.head_mean(x, torch.tensor([0, 1, 0, 2, 1]))
with torch.no_grad():
    assert torch.allclose(mixed[0], teacher.actor_mean(x)[0], atol=1e-6)
    assert not torch.allclose(mixed[1], teacher.actor_mean(x)[1])

print("OK: single-head compatible, adapter exact for every task, "
      "private base isolated")
