#!/bin/bash
# Adapter smoke: Peg(warm 400k) -> Stack(700k).
# Must show BOTH: Peg stays high through the Stack phase (the per-task head on a
# fully shared trunk lost it: 35% at end of c6v3_mh) AND Stack actually learns
# (with a 0.1 trunk it never left 0% inside 700k).
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

CK=ckpts/ppo_er_unified
echo "=============== adapter smoke start $(date) ==============="
python ppo_er_unified.py \
  --task-sequence PegInsertionSide-v1,StackCube-v1 \
  --task-steps 400000,700000 \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --eval-freq 10 --eval-eps 20 \
  --mtd-coef 2.0 \
  --multi-head --trunk-lr-mult 0.1 \
  --pretrain-ckpt "PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt" \
  --anchor-task "PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt,StackCube-v1=$CK/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt" \
  --run-tag smoke_ad >> logs/smoke_ad.log 2>&1
echo "--------------- adapter smoke done (exit=$?) $(date) ---------------"
