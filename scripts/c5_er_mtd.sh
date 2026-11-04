#!/bin/bash
# ER + MULTI-TEACHER DISTILLATION, same 5-task sequence / same budget as c5_cl.sh.
#
# Why: plain ER (replay_ratio 0.5 + BC coef 0.05) forgot the early tasks.
# Measured end state of the plain run (100 episodes each):
#   StackCube 25% | PullCube 100% | PickCube 57% | LiftPeg 76% | PushCube 100%
# so StackCube lost 62 pp and PickCube 41 pp.
#
# MTD adds one term to the PPO loss: every replay minibatch (which by
# construction comes from an OLD task) is distilled from THAT task's frozen
# expert, not from the current task's expert. The training data is therefore
# the old task's own observations, paired with the exact action the expert
# takes there -- strictly stronger than the BC-on-stale-actions term.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export PYTHONUNBUFFERED=1
mkdir -p logs
E=ckpts/ppo_er_unified

SC=$E/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt
PU=$E/ppo_er_unified_task0_PullCube-v1_ex48_PullCube_best.pt
PC=$E/ppo_er_unified_task2_PickCube-v1_gpu5_48h.pt
LP=$E/ppo_er_unified_task0_LiftPegUpright-v1_ex48_LiftPeg_best.pt
PS=$E/ppo_er_unified_task0_PushCube-v1_gpu5_48h.pt
for f in "$SC" "$PU" "$PC" "$LP" "$PS"; do
  [ -f "$f" ] || { echo "MISSING EXPERT: $f"; exit 1; }
done

SEQ=StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1
ANCH="StackCube-v1=$SC,PullCube-v1=$PU,PickCube-v1=$PC,LiftPegUpright-v1=$LP,PushCube-v1=$PS"
LAST=$E/ppo_er_unified_task4_PushCube-v1

# The GPU is shared: wait for the plain ER / SeqFT pass to finish first so the
# throughput numbers of both runs stay comparable (one job == one GPU).
echo "waiting for c5_cl.sh to finish ... $(date)"
while pgrep -f "bash c5_cl.sh" >/dev/null; do sleep 30; done
echo "GPU free, starting MTD $(date)"

echo "=============== ER-MTD $(date) ==============="
python -u ppo_er_unified_mtd.py --task-sequence "$SEQ" --total-steps 2500000 \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --eval-freq 25 --eval-eps 20 \
  --anchor-task "$ANCH" --anchor-coef 2.0 --mtd-coef 2.0 \
  --run-tag c5_er_mtd > logs/c5_er_mtd.log 2>&1
echo "--------------- ER-MTD done (exit=$?) $(date) ---------------"

python -u eval_ckpt_5tasks.py "${LAST}_c5_er_mtd.pt" \
  --eps 100 --tasks "$SEQ" > logs/c5_er_mtd_eval100.log 2>&1
echo "--------------- ER-MTD eval100 done (exit=$?) $(date) ---------------"
echo "ALL MTD RUNS DONE $(date)"
