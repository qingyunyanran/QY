#!/bin/bash
# Round-3 expert queue (48-dim unified obs).
#
#   1. PlaceSphere-v1    H=50,  dense reward
#   2. PullCubeTool-v1   H=100, dense reward (grasp the L-tool, hook the cube)
#   3. PegInsertionSide-v1 H=100 -- RETRY: the first 8M-step run was stuck at
#      SR=0% (reward ~3.5/10 = grasped + partly aligned, never inserted), so the
#      budget is doubled. If it is still 0% at 16M the task is a precision wall,
#      not a budget problem, and gets swapped out.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export PYTHONUNBUFFERED=1
mkdir -p logs

run() {
  name=$1
  tag=$2
  steps=${3:-8000000}
  echo "=============== $name (budget $steps, stop_sr=85) $(date) ==============="
  python -u ppo_er_unified.py \
    --task-sequence "$name" \
    --total-steps "$steps" \
    --num-envs 128 --num-steps 64 --num-minibatches 16 \
    --eval-freq 25 --eval-eps 20 \
    --stop-sr 85 \
    --run-tag "$tag" > "logs/expert_$tag.log" 2>&1
  echo "=============== $name done (exit=$?) $(date) ==============="
}

run PlaceSphere-v1       ex48_PlaceSphere
run PullCubeTool-v1      ex48_PullCubeTool
run PegInsertionSide-v1  ex48b_PegInsertion 16000000
echo "ALL EXPERTS ROUND3 DONE"
