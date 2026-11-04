#!/bin/bash
# Final evidence pass: re-evaluate the five experts over 100 episodes each
# (the numbers the report quotes as the upper bound) and build the summary.
#
# Waits for the MTD run first -- one job at a time on the single GPU.
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
SEQ=StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1

echo "waiting for c5_er_mtd.sh ... $(date)"
while pgrep -f "bash c5_er_mtd.sh" >/dev/null; do sleep 30; done
echo "GPU free $(date)"

echo "=============== expert eval100 $(date) ==============="
python -u eval_ckpt_5tasks.py "$SC" "$PU" "$PC" "$LP" "$PS" \
  --eps 100 --tasks "$SEQ" > logs/c5_expert_eval100.log 2>&1
echo "--------------- expert eval100 done (exit=$?) $(date) ---------------"

/root/miniconda3/envs/ms3/bin/python summarize_c5.py \
  --tasks "$SEQ" \
  --expert "StackCube-v1=87,PullCube-v1=100,PickCube-v1=98,LiftPegUpright-v1=95,PushCube-v1=100" \
  --run "ER=results/ppo_er_unified/ppo_er_unified_results_c5_er.json=logs/c5_er_eval100.log" \
  --run "SeqFT=results/ppo_er_unified/ppo_er_unified_results_c5_seqft.json=logs/c5_seqft_eval100.log" \
  --run "ER+MTD=results/ppo_er_unified/ppo_er_unified_results_c5_er_mtd.json=logs/c5_er_mtd_eval100.log" \
  --out logs/c5_summary.md > logs/c5_summary.log 2>&1
echo "--------------- summary done (exit=$?) $(date) ---------------"

python -u report_cl_matrix.py \
  results/ppo_er_unified/ppo_er_unified_results_c5_er.json \
  results/ppo_er_unified/ppo_er_unified_results_c5_seqft.json \
  results/ppo_er_unified/ppo_er_unified_results_c5_er_mtd.json \
  > logs/c5_matrix.log 2>&1
echo "--------------- matrix done (exit=$?) $(date) ---------------"
echo "ALL REPORT RUNS DONE $(date)"
