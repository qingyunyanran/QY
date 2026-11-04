#!/bin/bash
# CONSOLIDATION PASS over the same 5-task sequence, continuing the ER+MTD run.
#
# Why: ER+MTD reached 93.4% (100 episodes) but StackCube -- trained FIRST, so
# it absorbed every later update -- ended at 76%, 4 pp under the 80% bar.
# The archived 94.8% run got 90% on StackCube because StackCube was LAST there.
# A second pass over the sequence reproduces that condition without throwing
# away the ER+MTD result: the 4 maintenance tasks run for 0.3M steps each (just
# enough to re-seed the replay buffer with their own data and re-distil their
# expert), then StackCube gets 1.5M steps as the LAST task, with the full
# replay buffer and the 5-teacher MTD term keeping the others intact.
#
# Total: 0.3*4 + 1.5 = 2.7M steps (~30 min), not a 5x rerun.
#
# Also re-evaluates the archived 94.8% checkpoint over 100 episodes: that number
# was recorded with the in-training 20-episode evaluator, so it is NOT
# comparable with the 100-episode numbers reported for the new runs.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export PYTHONUNBUFFERED=1
E=ckpts/ppo_er_unified

SC=$E/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt
PU=$E/ppo_er_unified_task0_PullCube-v1_ex48_PullCube_best.pt
PC=$E/ppo_er_unified_task2_PickCube-v1_gpu5_48h.pt
LP=$E/ppo_er_unified_task0_LiftPegUpright-v1_ex48_LiftPeg_best.pt
PS=$E/ppo_er_unified_task0_PushCube-v1_gpu5_48h.pt
INIT=$E/ppo_er_unified_task4_PushCube-v1_c5_er_mtd.pt

SEQ=StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1
ANCH="StackCube-v1=$SC,PullCube-v1=$PU,PickCube-v1=$PC,LiftPegUpright-v1=$LP,PushCube-v1=$PS"
LAST=$E/ppo_er_unified_task4_PushCube-v1

for f in "$SC" "$PU" "$PC" "$LP" "$PS" "$INIT"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done

echo "=============== archived 94.8% ckpt, 100-ep re-eval $(date) ==============="
python -u eval_ckpt_5tasks.py "$SC" "$PU" "$PC" "$LP" "$PS" \
  --eps 100 --tasks "$SEQ" > logs/c5_expert_eval100.log 2>&1
python -u eval_ckpt_5tasks.py "$E/ppo_er_unified_task4_StackCube-v1_gpu5_48h.pt" \
  --eps 100 --tasks "$SEQ" > logs/c5_arch48h_eval100.log 2>&1
echo "--------------- arch eval100 done (exit=$?) $(date) ---------------"

echo "=============== consolidation pass $(date) ==============="
python -u ppo_er_unified_mtd.py --task-sequence "$SEQ" \
  --task-steps 300000,300000,300000,300000,1500000 \
  --init-ckpt "$INIT" \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --eval-freq 25 --eval-eps 20 \
  --anchor-task "$ANCH" --anchor-coef 2.0 --mtd-coef 2.0 \
  --run-tag c5_cons > logs/c5_cons.log 2>&1
echo "--------------- consolidation done (exit=$?) $(date) ---------------"

python -u eval_ckpt_5tasks.py "${LAST}_c5_cons.pt" \
  --eps 100 --tasks "$SEQ" > logs/c5_cons_eval100.log 2>&1
echo "--------------- cons eval100 done (exit=$?) $(date) ---------------"

/root/miniconda3/envs/ms3/bin/python summarize_c5.py \
  --tasks "$SEQ" \
  --expert "StackCube-v1=87,PullCube-v1=100,PickCube-v1=98,LiftPegUpright-v1=95,PushCube-v1=100" \
  --run "ER=results/ppo_er_unified/ppo_er_unified_results_c5_er.json=logs/c5_er_eval100.log" \
  --run "SeqFT=results/ppo_er_unified/ppo_er_unified_results_c5_seqft.json=logs/c5_seqft_eval100.log" \
  --run "ER+MTD=results/ppo_er_unified/ppo_er_unified_results_c5_er_mtd.json=logs/c5_er_mtd_eval100.log" \
  --run "ER+MTD+Cons=results/ppo_er_unified/ppo_er_unified_results_c5_cons.json=logs/c5_cons_eval100.log" \
  --out logs/c5_summary.md > logs/c5_summary.log 2>&1
echo "--------------- summary done (exit=$?) $(date) ---------------"
echo "ALL CONSOLIDATION RUNS DONE $(date)"
