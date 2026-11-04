#!/bin/bash
# Single-task experts for the 5-task COMPLEX sequence (48-dim unified obs).
# StackCube is NOT here: its expert is the already-verified 48-dim checkpoint
# from the first benchmark round.
#
# Each task: 8M-step cap (user budget), 128 envs x 64 steps (8192 per update,
# 16 minibatches -> the archived baseline optimiser schedule), early stop once
# the task holds >= 85% for two consecutive evals (20 eps each).
#
# Logs are raw (-u, no pipe) so `tail` shows live progress: piping through grep
# buffers 4 KB blocks and made a 4-minute run look hung.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export PYTHONUNBUFFERED=1
mkdir -p logs

run() {
  name=$1
  tag=$2
  echo "=============== $name (budget 8M, stop_sr=85) $(date) ==============="
  python -u ppo_er_unified.py \
    --task-sequence "$name" \
    --total-steps 8000000 \
    --num-envs 128 --num-steps 64 --num-minibatches 16 \
    --eval-freq 25 --eval-eps 20 \
    --stop-sr 85 \
    --run-tag "$tag" > "logs/expert_$tag.log" 2>&1
  echo "=============== $name done (exit=$?) $(date) ==============="
}

run PokeCube-v1        ex48_PokeCube
run StackPyramid-v1    ex48_StackPyramid
run PegInsertionSide-v1 ex48_PegInsertion
run PlugCharger-v1     ex48_PlugCharger
echo "ALL EXPERTS DONE"
