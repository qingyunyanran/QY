# -*- coding: utf-8 -*-
"""Test: appending a task to a consolidated multi-head run.

`load_task_expert` drops a single-task expert (flat 7-layer actor_mean) into the
head of one task. Two modes matter:

  include_base=True   the classic warm start: expert base + expert head.
  include_base=False  append mode: ONLY the head moves, the shared base keeps
                      the earlier tasks. This is the case the 9-task run needs,
                      because the base holds 6 already-learned tasks.

Checks:
  1. adapter-only leaves every base layer, head 0 and the critic byte-identical,
     while head k-1 becomes the expert's last two layers,
  2. the resulting head-0 / head-k functions differ where their weights differ
     (i.e. the expert really landed in head k-1),
  3. include_base=True DOES overwrite the base (the behaviour we must not use
     when appending, but must keep for standalone warm starts),
  4. an out-of-range task index is refused instead of silently writing head 0,
  5. an expert without any adapter layers is refused.
"""
import torch

from ppo_er_unified import Agent, UNIFIED_OBS_DIM, load_task_expert

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


def rand_actor(agent, seed=1234):
    """Random expert actor (keys '0.*' .. '6.*'), as stored in a ckpt."""
    sd = agent.actor_mean.state_dict()
    g = torch.Generator().manual_seed(seed)
    out = {k: torch.randn(v.shape, generator=g) for k, v in sd.items()}
    agent.actor_mean.load_state_dict(out, strict=True)
    return out


def base_keys():
    return [f"actor_mean.{i}.{p}" for i in range(4) for p in ("weight", "bias")
            if p == "weight" or i % 2 == 0]


def case_adapter_only_keeps_base():
    dst = mk(9, private=(0,), seed=2)
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    expert = rand_actor(mk(1, seed=1))
    load_task_expert(dst, 7, expert, include_base=False)
    now = dst.state_dict()
    # every key that is not the target head must be untouched
    for k in before:
        if k.startswith("actor_heads_extra.6."):
            continue
        assert torch.equal(now[k], before[k]), f"append mode changed {k}"
    # head 6 == expert's layers 4 and 6
    for src, local in (("4.weight", "0.weight"), ("4.bias", "0.bias"),
                       ("6.weight", "2.weight"), ("6.bias", "2.bias")):
        assert torch.equal(now[f"actor_heads_extra.6.{local}"], expert[src]), \
            f"head 6 {local} != expert {src}"


def case_expert_lands_in_its_own_head():
    """Head 6 must be the expert; heads 0..5 must stay their own functions."""
    dst = mk(9, private=(0,), seed=3)
    expert = rand_actor(mk(1, seed=4), seed=77)
    load_task_expert(dst, 7, expert, include_base=False)
    x = torch.randn(8, 256)
    h6 = dst.actor_heads_extra[6](x)
    for other in (0, 5):
        assert not torch.allclose(h6, dst.actor_heads_extra[other](x)), \
            f"head 6 matches head {other} -- expert went to the wrong head"


def case_include_base_overwrites_base():
    """include_base=True copies the expert's SHARED BASE (layers 0-3 only).

    actor_mean[4:] is task 0's adapter, not base: a warm start must not plant
    the expert's head there either, otherwise task 0 would silently become the
    new task.
    """
    dst = mk(9, private=(0,), seed=5)
    before = {k: v.clone() for k, v in dst.actor_mean.state_dict().items()}
    expert = rand_actor(mk(1, seed=6))
    load_task_expert(dst, 2, expert, include_base=True)
    now = dst.actor_mean.state_dict()
    nb = Agent.SHARED_BASE_LAYERS
    for k in before:
        if int(k.split(".")[0]) < nb:
            assert torch.equal(now[k], expert[k]), f"include_base left {k} unchanged"
        else:
            assert torch.equal(now[k], before[k]), f"task 0 adapter touched: {k}"
    # head 1 got the expert's adapter
    assert torch.equal(dst.actor_heads_extra[1].state_dict()["0.weight"],
                       expert["4.weight"])


def case_refuse_bad_index():
    dst = mk(9, private=(0,), seed=7)
    expert = rand_actor(mk(1, seed=8))
    for bad in (0, 10):
        try:
            load_task_expert(dst, bad, expert, include_base=False)
        except (IndexError, KeyError):
            continue
        raise AssertionError(f"task_idx={bad} was accepted")


def case_refuse_expert_without_adapter():
    dst = mk(9, private=(0,), seed=9)
    only_base = {f"{i}.weight": torch.zeros(1) for i in range(4)}
    try:
        load_task_expert(dst, 3, only_base, include_base=False)
    except RuntimeError:
        return
    raise AssertionError("an expert with no adapter layers was accepted")


check("append mode keeps base, head 0 and critic identical",
      case_adapter_only_keeps_base)
check("expert lands in its own head only", case_expert_lands_in_its_own_head)
check("include_base=True still overwrites the base",
      case_include_base_overwrites_base)
check("refuses task_idx 0 and out-of-range", case_refuse_bad_index)
check("refuses an expert with no adapter layers",
      case_refuse_expert_without_adapter)

print(f"\n{len(OK)} passed, {len(FAIL)} failed")
if FAIL:
    for n, e in FAIL:
        print(f"  {n}: {e}")
    raise SystemExit(1)
print("OK: appending an expert head (task 8) leaves earlier tasks intact")
