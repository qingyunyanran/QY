#!/bin/bash
# PickCube refit with a SLOWLY MOVING shared base (trunk-lr-mult 0.2).
#
# c8v3 (real PG, trunk frozen) lifted Pick only 64 -> 67%: the bottleneck is the
# frozen 256-dim CL feature, not the head. So let the shared base take small
# steps during the Pick phase and rely on multi-teacher distillation to keep the
# other seven heads glued to their experts while the feature space drifts.
# 0.2 (not the 0.5 of the earlier c8v1 pass, which wiped out five tasks):
# forgetting scaled with trunk drift speed.
#
# ORDER MATTERS: heads are positional, so repeat the c8v2 order exactly --
# Peg first (that is why its private net is key "0").
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
mkdir -p logs ckpts/ppo_er_unified results/ppo_er_unified
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

CK=ckpts/ppo_er_unified
SEQ=PegInsertionSide-v1,StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1,PokeCube-v1,PullCubeTool-v1
STEPS=8192,8192,8192,2000000,8192,8192,8192,8192

ANCHORS="PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt,\
StackCube-v1=$CK/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt,\
PullCube-v1=$CK/ppo_er_unified_task0_PullCube-v1_ex48_PullCube_best.pt,\
LiftPegUpright-v1=$CK/ppo_er_unified_task0_LiftPegUpright-v1_ex48_LiftPeg_best.pt,\
PushCube-v1=$CK/ppo_er_unified_task0_PushCube-v1_gpu5_48h.pt,\
PokeCube-v1=$CK/ppo_er_unified_task0_PokeCube-v1_poke24m_best.pt,\
PullCubeTool-v1=$CK/ppo_er_unified_task0_PullCubeTool-v1_tool30m_best.pt"

echo "=============== PickCube refit, trunk 0.2 (8-task) $(date) ==============="
python ppo_er_unified.py \
  --task-sequence "$SEQ" \
  --task-steps "$STEPS" \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --multi-head \
  --eval-freq 25 --eval-eps 20 \
  --stop-sr 85 \
  --gamma 0.99 \
  --mtd-coef 2.0 \
  --anchor-task "$ANCHORS" \
  --private-base PegInsertionSide-v1 \
  --init-ckpt "$CK/ppo_er_unified_task7_PullCubeTool-v1_c8v3_pickpg.pt" \
  --init-allow-expand \
  --trunk-lr-mult 0.2 \
  --run-tag c8v4_picktrunk >> logs/c8v4_picktrunk.log 2>&1
echo "--------------- train done (exit=$?) $(date) ---------------"

FINAL=$CK/ppo_er_unified_task7_PullCubeTool-v1_c8v4_picktrunk.pt
if [ -f "$FINAL" ]; then
  echo "--------------- eval 100 eps $(date) ---------------"
  python eval_ckpt_5tasks.py "$FINAL" --eps 100 --multi-head \
    --private-base PegInsertionSide-v1 \
    --tasks "$SEQ" \
    > logs/c8v4_picktrunk_eval100.log 2>&1
  echo "eval exit=$?"
  grep -E "SR=|mean SR" logs/c8v4_picktrunk_eval100.log
fi
echo "ALL C8V4 DONE $(date)"
