#!/bin/bash
# Which trunk_lr_mult keeps an already-finished task's head alive over a long
# chain? Baseline (mult=1.0, smoke_mh.log) over Peg -> Stack: Peg 100 -> 30 -> 20
# -> 80 -> 45 -> 70 over 600k steps, i.e. a big transient dip.
# Here: Peg(warm 400k) -> Stack(700k) -> Pull(700k), and Peg is read at the end.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

CK=ckpts/ppo_er_unified
SEQ=PegInsertionSide-v1,StackCube-v1,PullCube-v1
STEPS=400000,700000,700000
ANCHORS="PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt,\
StackCube-v1=$CK/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt,\
PullCube-v1=$CK/ppo_er_unified_task1_PullCube-v1_c5_er_mtd_best.pt"
WARM="PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt"

for MULT in 0.1 0.01; do
  TAG=probe_t${MULT/./}
  echo "=============== $TAG start $(date) ==============="
  python ppo_er_unified.py \
    --task-sequence "$SEQ" \
    --task-steps "$STEPS" \
    --num-envs 128 --num-steps 64 --num-minibatches 16 \
    --eval-freq 10 --eval-eps 20 \
    --mtd-coef 2.0 \
    --multi-head --trunk-lr-mult "$MULT" \
    --pretrain-ckpt "$WARM" \
    --anchor-task "$ANCHORS" \
    --run-tag "$TAG" >> logs/$TAG.log 2>&1
  echo "--------------- $TAG done (exit=$?) $(date) ---------------"
done
echo "=============== all probes done $(date) ==============="
