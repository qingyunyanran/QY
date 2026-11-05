#!/bin/bash
# PickCube refit on the frozen 8-task base.
#
# Evidence: the 8-task ckpt and the 6-task ckpt have BIT-IDENTICAL
# actor_mean[0:4] and actor_heads_extra[0:6] (diff_ckpt.py: 28 keys identical,
# only critic / actor_logstd / Peg private net moved). PickCube still scores 64%
# on BOTH -> it was never at 80%; the earlier "6/6 >= 80%" Pick number was
# measured wrong. 64% is the ceiling of the DISTILLED head: a 256->256->8 head
# fitted to expert actions on CL base features that are not the expert's own.
#
# So: give Pick REAL policy gradient instead of distillation (drop it from
# --anchor-task; the anchor branch is per current task), keep the shared base
# frozen (--trunk-lr-mult 0.0) so nothing else can move, and give every other
# task a single update (8192 steps) so their heads are untouched but the run
# still exercises the full positional 8-head mapping.
#
# ORDER MATTERS: heads are positional, so this must repeat the c8v2 order
# exactly -- Peg first (that is why its private net is key "0").
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

# every task except PickCube keeps its teacher; PickCube is trained by PG.
ANCHORS="PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt,\
StackCube-v1=$CK/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt,\
PullCube-v1=$CK/ppo_er_unified_task0_PullCube-v1_ex48_PullCube_best.pt,\
LiftPegUpright-v1=$CK/ppo_er_unified_task0_LiftPegUpright-v1_ex48_LiftPeg_best.pt,\
PushCube-v1=$CK/ppo_er_unified_task0_PushCube-v1_gpu5_48h.pt,\
PokeCube-v1=$CK/ppo_er_unified_task0_PokeCube-v1_poke24m_best.pt,\
PullCubeTool-v1=$CK/ppo_er_unified_task0_PullCubeTool-v1_tool30m_best.pt"

echo "=============== PickCube refit (8-task, frozen base) $(date) ==============="
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
  --init-ckpt "$CK/ppo_er_unified_task7_PullCubeTool-v1_c8v2_ad.pt" \
  --init-allow-expand \
  --trunk-lr-mult 0.0 \
  --run-tag c8v3_pickpg >> logs/c8v3_pickpg.log 2>&1
echo "--------------- train done (exit=$?) $(date) ---------------"

FINAL=$CK/ppo_er_unified_task7_PullCubeTool-v1_c8v3_pickpg.pt
if [ -f "$FINAL" ]; then
  echo "--------------- eval 100 eps $(date) ---------------"
  python eval_ckpt_5tasks.py "$FINAL" --eps 100 --multi-head \
    --private-base PegInsertionSide-v1 \
    --tasks "$SEQ" \
    > logs/c8v3_pickpg_eval100.log 2>&1
  echo "eval exit=$?"
  grep -E "SR=|mean SR" logs/c8v3_pickpg_eval100.log
fi
echo "ALL C8V3 DONE $(date)"
