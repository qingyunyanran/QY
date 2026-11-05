#!/bin/bash
# Consolidation pass after the trunk moved (c8v4 -> c8v5).
#
# c8v4 let the shared base drift (trunk_lr_mult 0.2) during the Pick phase and
# Pick jumped 64 -> 98%: the frozen 256-dim feature really was the bottleneck.
# The cost was that every other adapter had only ONE update to re-fit the new
# feature space, so Stack 69 / LiftPeg 49 / Poke 56 / Tool 77 (mean 79.0).
#
# So: freeze the new base (trunk 0.0) and give each of those seven tasks a short
# DISTILLATION re-alignment (they stay in --anchor-task, so pg_loss is zeroed and
# the head is fitted to its expert on the CURRENT trunk features). Pick is left
# out of the anchor table so its 98% head is not touched: 8192 steps.
#
# Order is positional (heads are indexed), so it repeats the c8v2/c8v4 order.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
mkdir -p logs ckpts/ppo_er_unified results/ppo_er_unified
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

CK=ckpts/ppo_er_unified
SEQ=PegInsertionSide-v1,StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1,PokeCube-v1,PullCubeTool-v1
STEPS=200000,200000,200000,8192,200000,200000,200000,200000

ANCHORS="PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt,\
StackCube-v1=$CK/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt,\
PullCube-v1=$CK/ppo_er_unified_task0_PullCube-v1_ex48_PullCube_best.pt,\
LiftPegUpright-v1=$CK/ppo_er_unified_task0_LiftPegUpright-v1_ex48_LiftPeg_best.pt,\
PushCube-v1=$CK/ppo_er_unified_task0_PushCube-v1_gpu5_48h.pt,\
PokeCube-v1=$CK/ppo_er_unified_task0_PokeCube-v1_poke24m_best.pt,\
PullCubeTool-v1=$CK/ppo_er_unified_task0_PullCubeTool-v1_tool30m_best.pt"

echo "=============== consolidation on the drifted base $(date) ==============="
python ppo_er_unified.py \
  --task-sequence "$SEQ" \
  --task-steps "$STEPS" \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --multi-head \
  --eval-freq 10 --eval-eps 20 \
  --stop-sr 85 \
  --gamma 0.99 \
  --mtd-coef 2.0 \
  --anchor-task "$ANCHORS" \
  --private-base PegInsertionSide-v1 \
  --init-ckpt "$CK/ppo_er_unified_task7_PullCubeTool-v1_c8v4_picktrunk.pt" \
  --init-allow-expand \
  --trunk-lr-mult 0.0 \
  --run-tag c8v5_cons >> logs/c8v5_cons.log 2>&1
echo "--------------- train done (exit=$?) $(date) ---------------"

FINAL=$CK/ppo_er_unified_task7_PullCubeTool-v1_c8v5_cons.pt
if [ -f "$FINAL" ]; then
  echo "--------------- eval 100 eps $(date) ---------------"
  python eval_ckpt_5tasks.py "$FINAL" --eps 100 --multi-head \
    --private-base PegInsertionSide-v1 \
    --tasks "$SEQ" \
    > logs/c8v5_cons_eval100.log 2>&1
  echo "eval exit=$?"
  grep -E "SR=|mean SR" logs/c8v5_cons_eval100.log
fi
echo "ALL C8V5 DONE $(date)"
