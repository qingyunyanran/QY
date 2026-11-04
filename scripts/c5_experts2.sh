#!/bin/bash
# Round-2 expert queue (48-dim unified obs).
#
#  1. PokeCube-v1        H=100 (0->100 step budget doubled after the H=50 run
#                        plateaued at 65-70%), dense reward -> no shaping needed
#  2. PegInsertionSide-v1 H=100, dense reward -> no shaping needed
#
# StackPyramid-v1 / PlugCharger-v1 are NOT in this queue: both are
# reward_mode="none" (SUPPORTED_REWARD_MODES = {"none","sparse"}), i.e. a 0/1
# success signal, which PPO cannot bootstrap from. They need a shaping wrapper
# and are handled separately.
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

run PokeCube-v1         ex48b_PokeCube
run PegInsertionSide-v1 ex48_PegInsertion
echo "ALL EXPERTS ROUND2 DONE"
