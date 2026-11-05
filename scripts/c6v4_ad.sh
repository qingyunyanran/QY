#!/bin/bash
# 6-task ER+MTD with shared base + per-task adapter, PegInsertionSide isolated.
# Order: Peg first (its expert restores into its own network, no sharing).
# Evidence for the split:
#   fully shared trunk, per-task head : Peg 49 / Stack 27  (mean 74.0)
#   + head refit, trunk frozen        : Peg 56 / Stack 58  (mean 80.2)
#   shared base + adapter, trunk 0.1  : Peg 85-95, Stack 0 (never learns)
#   shared base + adapter, trunk 0.5  : Peg 60-80, Stack 75
# => the two tasks need opposite base plasticity, so the base moves freely for
#    the 5 simple tasks and Peg keeps its own.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

CK=ckpts/ppo_er_unified
SEQ=PegInsertionSide-v1,StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1
STEPS=600000,2500000,2500000,2500000,2500000,2500000
ANCHORS="PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt,\
StackCube-v1=$CK/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt,\
PullCube-v1=$CK/ppo_er_unified_task1_PullCube-v1_c5_er_mtd_best.pt,\
PickCube-v1=$CK/ppo_er_unified_task2_PickCube-v1_c5_er_mtd_best.pt,\
LiftPegUpright-v1=$CK/ppo_er_unified_task3_LiftPegUpright-v1_c5_er_mtd_best.pt,\
PushCube-v1=$CK/ppo_er_unified_task4_PushCube-v1_c5_er_mtd_best.pt"
WARM="PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt"

echo "=============== c6v4 adapter+private-base start $(date) ==============="
python ppo_er_unified.py \
  --task-sequence "$SEQ" \
  --task-steps "$STEPS" \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --eval-freq 25 --eval-eps 20 \
  --mtd-coef 2.0 \
  --multi-head --private-base PegInsertionSide-v1 \
  --pretrain-ckpt "$WARM" \
  --anchor-task "$ANCHORS" \
  --run-tag c6v4_ad >> logs/c6v4_ad.log 2>&1
echo "--------------- c6v4 done (exit=$?) $(date) ---------------"
