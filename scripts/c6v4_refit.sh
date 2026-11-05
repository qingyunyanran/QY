#!/bin/bash
# Targeted refit on the finished c6v4 checkpoint, with the shared base held
# still so the four tasks that already pass cannot move:
#   Peg  300k  -- its private net trains again (85% -> 62% under MTD), then it
#                 is frozen for the remaining phases of this run as well
#   Pick 600k  -- the one shared-base task that came in under target (75%)
#   others 8k  -- a single update each, just so the loop reaches them
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

CK=ckpts/ppo_er_unified
SEQ=PegInsertionSide-v1,StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1
STEPS=300000,8192,8192,600000,8192,8192
ANCHORS="PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt,\
StackCube-v1=$CK/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt,\
PullCube-v1=$CK/ppo_er_unified_task1_PullCube-v1_c5_er_mtd_best.pt,\
PickCube-v1=$CK/ppo_er_unified_task2_PickCube-v1_c5_er_mtd_best.pt,\
LiftPegUpright-v1=$CK/ppo_er_unified_task3_LiftPegUpright-v1_c5_er_mtd_best.pt,\
PushCube-v1=$CK/ppo_er_unified_task4_PushCube-v1_c5_er_mtd_best.pt"

echo "=============== c6v4 refit start $(date) ==============="
python ppo_er_unified.py \
  --task-sequence "$SEQ" \
  --task-steps "$STEPS" \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --eval-freq 10 --eval-eps 20 \
  --mtd-coef 2.0 \
  --multi-head --private-base PegInsertionSide-v1 --trunk-lr-mult 0.0 \
  --init-ckpt "$CK/ppo_er_unified_task5_PushCube-v1_c6v4_ad.pt" \
  --anchor-task "$ANCHORS" \
  --run-tag c6v4_refit >> logs/c6v4_refit.log 2>&1
echo "--------------- c6v4 refit done (exit=$?) $(date) ---------------"
