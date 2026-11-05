#!/bin/bash
# c8v2: same 8-task append as c8v1, but the SHARED BASE IS FROZEN.
#
# Why the change: c8v1 (--trunk-lr-mult 0.5) ended with Stack 0% / Pull 5% /
# Pick 0% / Lift 15% -- the base was pulled towards the two new tasks while the
# six old tasks got only ONE update each, far too few to re-adapt their heads to
# a moving base. The MTD term (measured 0.0037) was an order of magnitude too
# weak to hold it.
#
# With the base frozen, the old tasks cannot degrade by construction: their base
# layers and adapters are byte-identical to the 6-task checkpoint, and Peg keeps
# its private network. The two new tasks train ONLY their adapter (a 256->256->8
# head, i.e. standard linear probing on a frozen backbone), which is why they
# get a longer budget.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

CK=ckpts/ppo_er_unified
PY=/root/miniconda3/envs/ms3/bin/python
SEQ=PegInsertionSide-v1,StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1,PokeCube-v1,PullCubeTool-v1
STEPS=8192,8192,8192,8192,8192,8192,4000000,4000000
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

echo "=============== c8v2 frozen-base start $(date '+%F %T') ==============="
$PY ppo_er_unified.py \
  --task-sequence "$SEQ" \
  --task-steps "$STEPS" \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --eval-freq 50 --eval-eps 20 \
  --mtd-coef 2.0 \
  --multi-head --private-base PegInsertionSide-v1 --trunk-lr-mult 0.0 \
  --init-ckpt "$CK/ppo_er_unified_task0_PegInsertionSide-v1_c6v4_refit_best.pt" \
  --init-allow-expand \
  --pretrain-ckpt "$WARM" \
  --pretrain-adapter-only \
  --anchor-task "$ANCHORS" \
  --stop-sr 85 \
  --run-tag c8v2_ad > logs/c8v2_ad.log 2>&1
echo "=============== c8v2 done rc=$? $(date '+%F %T') ==============="
grep -E '\[best\]' logs/c8v2_ad.log | tail -3
echo DONE > logs/c8v2_ad.done
