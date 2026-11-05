#!/bin/bash
# 8-task ER+MTD: continue the finished 6-task run with the 2 tasks that cleared
# the 80% anchor gate (100 independent episodes):
#   PokeCube-v1      85.0%   (73 -> 78 -> 85 after +8M)
#   PullCubeTool-v1  99.0%   (0 -> 99, it only found the solution in the last 10M)
# RollBall-v1 (44.0%) is EXCLUDED: a task below its own 80% bar drags the CL
# mean down and its anchor would inject a weak teacher into the replay rows.
#
#   --init-ckpt ... --init-allow-expand   the 5 shared-base heads + Peg's private
#                                         network come back verbatim; the 2
#                                         appended heads start fresh. Head index
#                                         is positional, so the old 6 tasks stay
#                                         first in the sequence.
#   --pretrain-adapter-only               each new expert drops only its top two
#                                         layers into its own head, leaving the
#                                         shared base carrying the 6 old tasks.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

CK=ckpts/ppo_er_unified
PY=/root/miniconda3/envs/ms3/bin/python
SEQ=PegInsertionSide-v1,StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1,PokeCube-v1,PullCubeTool-v1
STEPS=8192,8192,8192,8192,8192,8192,3000000,3000000
ANCHORS="PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt,\
StackCube-v1=$CK/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt,\
PullCube-v1=$CK/ppo_er_unified_task1_PullCube-v1_c5_er_mtd_best.pt,\
PickCube-v1=$CK/ppo_er_unified_task2_PickCube-v1_c5_er_mtd_best.pt,\
LiftPegUpright-v1=$CK/ppo_er_unified_task3_LiftPegUpright-v1_c5_er_mtd_best.pt,\
PushCube-v1=$CK/ppo_er_unified_task4_PushCube-v1_c5_er_mtd_best.pt,\
PokeCube-v1=$CK/ppo_er_unified_task0_PokeCube-v1_poke24m_best.pt,\
PullCubeTool-v1=$CK/ppo_er_unified_task0_PullCubeTool-v1_tool30m_best.pt"
WARM="PokeCube-v1=$CK/ppo_er_unified_task0_PokeCube-v1_poke24m_best.pt,\
PullCubeTool-v1=$CK/ppo_er_unified_task0_PullCubeTool-v1_tool30m_best.pt"

echo "=============== c8v1 adapter+private-base start $(date '+%F %T') ==============="
$PY ppo_er_unified.py \
  --task-sequence "$SEQ" \
  --task-steps "$STEPS" \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --eval-freq 10 --eval-eps 20 \
  --mtd-coef 2.0 \
  --multi-head --private-base PegInsertionSide-v1 --trunk-lr-mult 0.5 \
  --init-ckpt "$CK/ppo_er_unified_task0_PegInsertionSide-v1_c6v4_refit_best.pt" \
  --init-allow-expand \
  --pretrain-ckpt "$WARM" \
  --pretrain-adapter-only \
  --anchor-task "$ANCHORS" \
  --stop-sr 85 \
  --run-tag c8v1_ad > logs/c8v1_ad.log 2>&1
echo "=============== c8v1 done rc=$? $(date '+%F %T') ==============="
grep -E '\[best\]' logs/c8v1_ad.log | tail -3
echo DONE > logs/c8v1_ad.done
