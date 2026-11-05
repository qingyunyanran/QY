#!/bin/bash
# Is Stack's 0% a plasticity problem or a capacity problem?
# Continue the adapter smoke ckpt with trunk_lr_mult=0.5 for Stack only and
# watch both numbers: if Stack takes off AND Peg holds, 0.5 is the setting; if
# Stack still sits at 0, the top-of-network adapter is the limit and the shared
# base itself has to be per task.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

CK=ckpts/ppo_er_unified
echo "=============== plasticity test (mult=0.5) start $(date) ==============="
python ppo_er_unified.py \
  --task-sequence PegInsertionSide-v1,StackCube-v1 \
  --task-steps 8192,600000 \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --eval-freq 10 --eval-eps 20 \
  --mtd-coef 2.0 \
  --multi-head --trunk-lr-mult 0.5 \
  --init-ckpt "$CK/ppo_er_unified_task1_StackCube-v1_smoke_ad.pt" \
  --anchor-task "PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt,StackCube-v1=$CK/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt" \
  --run-tag ad_t05 >> logs/ad_t05.log 2>&1
echo "--------------- plasticity test done (exit=$?) $(date) ---------------"
