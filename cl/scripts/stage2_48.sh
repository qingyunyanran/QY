#!/bin/bash
# Stage 2 (48-dim): 5-task ER, StackCube last.
#
# Optimizer schedule is matched to the archived 4-task baseline on purpose:
#   batch 128 x 64 = 8,192 transitions (= baseline 16 x 512)
#   16 minibatches of 512            (= baseline 16)
#   4 epochs                         (= baseline 4)
# So the same step budget gives the same number of PPO iterations and gradient
# steps; only the simulator batching differs.
#
# StackCube is the one task with a reference policy, so it enters the shared
# actor by distillation (LwF-style anchor) instead of by warm-start:
#   * warm-starting overwrote the shared actor and wiped the four earlier tasks
#     (measured: 0% on all four), because all five tasks share one policy net;
#   * the anchor keeps the expert as a frozen teacher and adds
#     coef * MSE(actor_mean(obs), expert_mean(obs)) on StackCube batches only,
#     with the PPO policy term switched off for that phase (the teacher is the
#     authority) and the PPO KL gate relaxed (distillation must move the policy).
# Verified on a 2-task probe (PushCube + StackCube): 90-100% / 100%.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
cd /root/cl
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1
mkdir -p logs results/ppo_er_unified ckpts/ppo_er_unified
exec python -u ppo_er_unified.py \
  --task-sequence PushCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,StackCube-v1 \
  --task-steps 2000000,2000000,6000000,4000000,2500000 \
  --replay-ratio 0.5 \
  --buffer-per-task 200000 \
  --num-envs 128 \
  --num-steps 64 \
  --num-minibatches 16 \
  --eval-freq 50 \
  --eval-eps 20 \
  --sim-backend gpu \
  --seed 42 \
  --run-tag gpu5_48h \
  --anchor-task StackCube-v1=/root/cl/ckpts/ppo_official_unified/StackCube-v1_seed1_official48_native.pt \
  --anchor-coef 2.0
