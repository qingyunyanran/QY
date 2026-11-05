# -*- coding: utf-8 -*-
"""Test: continuing a 6-head run into a 9-head one keeps every old head intact.

What must hold:
  1. shared trunk / existing adapters / private net / critic are byte-identical
     to the 6-head checkpoint,
  2. the three APPENDED heads keep their fresh initialisation (so the new tasks
     do not silently inherit head 5 of another task),
  3. a checkpoint whose heads are not a contiguous prefix is refused,
  4. a checkpoint with MORE heads than the run is refused,
  5. the single-head path still loads a plain policy with strict=True.
"""
import io
import torch

from ppo_er_unified import Agent, UNIFIED_OBS_DIM, expand_state_dict_load

OK, FAIL = [], []


def check(name, fn):
    try:
        fn()
        OK.append(name)
        print(f"  ok   {name}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, e))
        print(f"  FAIL {name}: {type(e).__name__}: {e}")


def mk(n_heads, private=(), seed=0):
    torch.manual_seed(seed)
    return Agent(UNIFIED_OBS_DIM, 8, n_heads=n_heads, private_base_ids=private)


def rand_state(agent):
    """Distinct, non-default values so an accidental no-op load is visible.

    NOTE: the values must also be written INTO `agent` -- state_dict() returns
    live parameter tensors, so replacing dict entries alone would leave the
    module on its layer_init weights and a forward comparison between "source"
    and "loaded copy" would compare two unrelated networks.
    """
    sd = agent.state_dict()
    g = torch.Generator().manual_seed(1234)
    sd = {k: torch.randn(v.shape, generator=g) for k, v in sd.items()}
    agent.load_state_dict(sd, strict=True)
    return sd


SOURCE_KEYS = [
    "critic.0.weight", "critic.6.weight",
    "actor_mean.0.weight", "actor_mean.0.bias", "actor_mean.2.weight",
    "actor_mean.4.weight", "actor_mean.6.weight", "actor_mean.6.bias",
    "actor_heads_extra.0.0.weight", "actor_heads_extra.0.2.bias",
    "actor_heads_extra.4.0.weight", "actor_heads_extra.4.2.weight",
    "private_nets.0.0.weight", "private_nets.0.6.weight",
    "actor_logstd",
]


def case_expand_keeps_old_heads():
    src = mk(6, private=(0,), seed=1)
    sd = rand_state(src)
    dst = mk(9, private=(0,), seed=2)
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    kept, fresh, missing = expand_state_dict_load(dst, sd, verbose=True,
                                                 source="<6-head test ckpt>")
    now = dst.state_dict()
    for k in SOURCE_KEYS:
        assert k in sd, f"test itself is stale: {k} not in source"
        assert torch.equal(now[k], sd[k]), f"old key changed by the load: {k}"
    # appended heads must be untouched (fresh init from seed=2)
    for i in (5, 6, 7):
        for k in ("0.weight", "0.bias", "2.weight", "2.bias"):
            key = f"actor_heads_extra.{i}.{k}"
            assert torch.equal(now[key], before[key]), f"new head overwritten: {key}"
    assert kept == {0, 1, 2, 3, 4}, kept
    assert fresh == {5, 6, 7}, fresh


def case_head_forward_stays_per_task():
    """After the load, head 5 must still be a DIFFERENT function from head 4."""
    src = mk(6, private=(0,), seed=1)
    sd = rand_state(src)
    dst = mk(9, private=(0,), seed=2)
    expand_state_dict_load(dst, sd, verbose=False, source="<test>")
    x = torch.randn(4, 256)
    h4 = dst.actor_heads_extra[4](x)
    h5 = dst.actor_heads_extra[5](x)
    assert not torch.allclose(h4, h5), "head 5 copied head 4 -- new task would " \
                                       "start from the previous task's policy"
    # and the shared trunk must match the source trunk
    trunk_x = torch.randn(4, UNIFIED_OBS_DIM)
    with torch.no_grad():
        a = src.actor_mean[:4](trunk_x)
        b = dst.actor_mean[:4](trunk_x)
    assert torch.allclose(a, b, atol=1e-6), "shared trunk drifted on load"


def case_refuse_non_prefix():
    good = rand_state(mk(6, private=(0,), seed=3))
    holed = {k: v for k, v in good.items()
             if not k.startswith("actor_heads_extra.2.")}
    dst = mk(9, private=(0,), seed=4)
    try:
        expand_state_dict_load(dst, holed, verbose=False, source="<holed>")
    except RuntimeError:
        return
    raise AssertionError("a non-contiguous head layout was accepted")


def case_refuse_too_many_heads():
    sd = rand_state(mk(6, private=(0,), seed=5))
    dst = mk(3, private=(0,), seed=6)
    try:
        expand_state_dict_load(dst, sd, verbose=False, source="<too many>")
    except RuntimeError:
        return
    raise AssertionError("a checkpoint with MORE heads than the run was accepted")


def case_refuse_action_width():
    sd = rand_state(mk(6, private=(0,), seed=7))
    dst = Agent(UNIFIED_OBS_DIM, 7, n_heads=9, private_base_ids=(0,))
    try:
        expand_state_dict_load(dst, sd, verbose=False, source="<7-action>")
    except RuntimeError:
        return
    raise AssertionError("a 7-action checkpoint was loaded into an 8-action head")


def case_single_head_strict_still_works():
    sd = rand_state(mk(1, seed=8))
    dst = mk(1, seed=9)
    dst.load_state_dict(sd, strict=True)
    for k, v in sd.items():
        assert torch.equal(dst.state_dict()[k], v), k


def case_roundtrip_through_file():
    """The real path: torch.save -> torch.load -> expand (no in-memory shortcut)."""
    src = mk(6, private=(0,), seed=10)
    sd = rand_state(src)
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    loaded = torch.load(buf, map_location="cpu")
    dst = mk(9, private=(0,), seed=11)
    expand_state_dict_load(dst, loaded, verbose=False, source="<file>")
    for k in SOURCE_KEYS:
        assert torch.equal(dst.state_dict()[k], sd[k]), k


check("expand 6 -> 9 keeps old heads, refreshes new ones", case_expand_keeps_old_heads)
check("head 5 is an independent function, trunk identical", case_head_forward_stays_per_task)
check("refuses a non-prefix head layout", case_refuse_non_prefix)
check("refuses a checkpoint with more heads", case_refuse_too_many_heads)
check("refuses a 7-action checkpoint", case_refuse_action_width)
check("single-head strict load unchanged", case_single_head_strict_still_works)
check("save/load round trip then expand", case_roundtrip_through_file)

print(f"\n{len(OK)} passed, {len(FAIL)} failed")
if FAIL:
    for n, e in FAIL:
        print(f"  {n}: {e}")
    raise SystemExit(1)
print("OK: head expansion (6 -> 9) verified")
