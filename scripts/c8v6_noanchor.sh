#!/bin/bash
# Ablation: can PokeCube / PullCubeTool be learned IN the sequence WITHOUT an
# expert teacher?
#
# Matched to c8v2 on purpose -- same init, same order, same frozen base, same
# 4M steps per new task. The ONLY difference is that Poke and Tool are dropped
# from --anchor-task, so pg_loss is no longer zeroed for them: their heads learn
# from real policy gradient on the shared base's features instead of being
# fitted to poke24m / tool30m with MSE.
#
# Interpretation note: if this lands far below c8v2's 87 / 98, it shows how much
# of the "8/8" comes from expert distillation rather than from CL proper. It is
# also a BUDGET comparison: the two experts needed 24M / 30M of single-task PPO,
# so 4M of in-sequence RL is a hard test, not a like-for-like one.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
mkdir -p logs ckpts/ppo_er_unified results/ppo_er_unified
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

CK=ckpts/ppo_er_unified
SEQ=PegInsertionSide-v1,StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1,PokeCube-v1,PullCubeTool-v1
STEPS=8192,8192,8192,8192,8192,8192,4000000,4000000

# Poke and Tool DELIBERATELY absent: no teacher for the two tasks under test.
ANCHORS="PegInsertionSide-v1=$CK/ppo_er_unified_task0_PegInsertionSide-v1_peg75m_best.pt,\
StackCube-v1=$CK/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt,\
PullCube-v1=$CK/ppo_er_unified_task0_PullCube-v1_ex48_PullCube_best.pt,\
LiftPegUpright-v1=$CK/ppo_er_unified_task0_LiftPegUpright-v1_ex48_LiftPeg_best.pt,\
PushCube-v1=$CK/ppo_er_unified_task0_PushCube-v1_gpu5_48h.pt"

echo "=============== ablation: no teacher for Poke/Tool $(date) ==============="
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
  --init-ckpt "$CK/ppo_er_unified_task7_PullCubeTool-v1_c8v5_cons.pt" \
  --init-allow-expand \
  --trunk-lr-mult 0.0 \
  --run-tag c8v6_noanchor >> logs/c8v6_noanchor.log 2>&1
echo "--------------- train done (exit=$?) $(date) ---------------"

FINAL=$CK/ppo_er_unified_task7_PullCubeTool-v1_c8v6_noanchor.pt
if [ -f "$FINAL" ]; then
  echo "--------------- eval 100 eps $(date) ---------------"
  python eval_ckpt_5tasks.py "$FINAL" --eps 100 --multi-head \
    --private-base PegInsertionSide-v1 \
    --tasks "$SEQ" \
    > logs/c8v6_noanchor_eval100.log 2>&1
  echo "eval exit=$?"
  grep -E "SR=|mean SR" logs/c8v6_noanchor_eval100.log
fi
echo "ALL C8V6 DONE $(date)"
