#!/bin/bash
# Stage 3 (48-dim): SeqFT control -- no experience replay, same task order,
# same per-task budgets, same optimizer schedule and the same StackCube
# distillation anchor as stage2_48.sh. Only the replay buffer differs, so
# ER vs SeqFT isolates the contribution of replay.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
cd /root/cl
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1
mkdir -p logs results/ppo_er_unified ckpts/ppo_er_unified
exec python -u ppo_er_unified.py \
  --seqft \
  --task-sequence PushCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,StackCube-v1 \
  --task-steps 2000000,2000000,6000000,4000000,2500000 \
  --buffer-per-task 200000 \
  --num-envs 128 \
  --num-steps 64 \
  --num-minibatches 16 \
  --eval-freq 50 \
  --eval-eps 20 \
  --sim-backend gpu \
  --seed 42 \
  --run-tag gpu5_48h_seqft \
  --anchor-task StackCube-v1=/root/cl/ckpts/ppo_official_unified/StackCube-v1_seed1_official48_native.pt \
  --anchor-coef 2.0
