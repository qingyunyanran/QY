#!/bin/bash
# Re-train the three candidate anchors with a bigger budget, warm-started from
# the 8M-step checkpoints.
#
# Why gamma=0.99 for RollBall / PullCubeTool but not PokeCube:
#   RollBall success = the ball STOPS near the goal; the push happens early and
#   the payoff arrives 30-80 steps later, so the default 0.8 (effective horizon
#   ~5 steps) cannot see it. PullCubeTool needs a grasp + hook + 50-70 step drag
#   inside a 100-step horizon. PokeCube already reached 85% with the default
#   gamma, so it is continued unchanged (only the budget grows).
set -u
cd /root/cl
PY=/root/miniconda3/envs/ms3/bin/python
CK=ckpts/ppo_er_unified
LOG=logs

echo "===== start $(date '+%F %T') ====="

echo "===== [1/3] PokeCube-v1  8M -> 16M (default gamma)  $(date '+%T') ====="
$PY -u ppo_er_unified.py \
  --task-sequence PokeCube-v1 --total-steps 16000000 \
  --pretrain-ckpt PokeCube-v1=$CK/ppo_er_unified_task0_PokeCube-v1_ex48c_PokeCube_best.pt \
  --num-envs 128 --num-steps 64 --num-minibatches 16 --eval-freq 25 --eval-eps 20 --stop-sr 85 \
  --run-tag poke16m --seed 42 \
  > $LOG/train_poke16m.log 2>&1
echo "===== [1/3] done rc=$? $(date '+%T') ====="
grep -E '\[best\]' $LOG/train_poke16m.log | tail -2

echo "===== [2/3] RollBall-v1  8M -> 30M (gamma 0.99)  $(date '+%T') ====="
$PY -u ppo_er_unified.py \
  --task-sequence RollBall-v1 --total-steps 30000000 \
  --gamma 0.99 \
  --pretrain-ckpt RollBall-v1=$CK/ppo_er_unified_task0_RollBall-v1_ex48_RollBall_best.pt \
  --num-envs 128 --num-steps 64 --num-minibatches 16 --eval-freq 25 --eval-eps 20 --stop-sr 85 \
  --run-tag roll30m --seed 42 \
  > $LOG/train_roll30m.log 2>&1
echo "===== [2/3] done rc=$? $(date '+%T') ====="
grep -E '\[best\]' $LOG/train_roll30m.log | tail -2

echo "===== [3/3] PullCubeTool-v1  8M -> 30M (gamma 0.99)  $(date '+%T') ====="
$PY -u ppo_er_unified.py \
  --task-sequence PullCubeTool-v1 --total-steps 30000000 \
  --gamma 0.99 \
  --pretrain-ckpt PullCubeTool-v1=$CK/ppo_er_unified_task0_PullCubeTool-v1_ex48_PullCubeTool.pt \
  --num-envs 128 --num-steps 64 --num-minibatches 16 --eval-freq 25 --eval-eps 20 --stop-sr 85 \
  --run-tag tool30m --seed 42 \
  > $LOG/train_tool30m.log 2>&1
echo "===== [3/3] done rc=$? $(date '+%T') ====="
grep -E '\[best\]' $LOG/train_tool30m.log | tail -2

echo "===== all done $(date '+%F %T') ====="
ls -1 $CK | grep -E 'poke16m|roll30m|tool30m'
echo DONE_MARKER > logs/train_new3.done
