#!/bin/bash
# Round-5 experts: the two tasks that still need a documented >= 80% checkpoint
# under the *current* code (best-checkpoint saving was added after their first
# run, so their old artefacts only kept the LAST weights).
#
#   PokeCube-v1       H=100  -- best SR ever observed was 85%, but the saved
#                                checkpoint was the 70% final one.
#   LiftPegUpright-v1 H=50   -- 100% in the valid5 run, 75% at the end of the
#                                48h run; same "last != best" problem.
#
# TurnFaucet-v1 / AssemblingKits-v1 / PlugCharger-v1 / StackPyramid-v1 were
# rejected at the reward-mode audit (SUPPORTED_REWARD_MODES == sparse/none).
# PickSingleYCB-v1 was rejected because its YCB assets are not on the machine
# and huggingface.co is unreachable from it.
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

run PokeCube-v1       ex48c_PokeCube
run LiftPegUpright-v1 ex48_LiftPeg
echo "ALL EXPERTS ROUND5 DONE"
