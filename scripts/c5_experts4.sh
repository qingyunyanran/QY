#!/bin/bash
# Round-4 expert queue (48-dim unified obs).
#
#   1. PullCubeTool-v1  H=100  dense reward -- grasp the L-shaped tool, hook the
#      cube and drag it back close to the robot base (success = cube within
#      0.6 m of the base, a GENEROUS criterion).
#   2. RollBall-v1      H=80   dense reward -- push a free ball onto the goal
#      point; the ball keeps rolling, so this is contact- and timing-sensitive.
#   3. PullCube-v1      H=50   dense reward -- insurance / fallback slot. Same
#      family as PushCube (100% SR in round 1) so it is the cheapest way to get
#      a 5th task that is provably trainable >= 80%.
#
# Dropped after the reward/precision audit (see the report):
#   StackPyramid  - reward_mode none,sparse (0/1 only), reward == 0.000 for 490k
#   PlugCharger   - reward_mode none,sparse + 0.75 mm pegs in 0.5 mm clearance
#   PlaceSphere   - 5 mm tolerance for balancing a sphere on a flat plate,
#                   reward plateaued at 0.31 with SR 0% after 6.3M steps
#   PegInsertionSide - 8M steps SR 0% (grasped + aligned, never inserted);
#                   ManiSkill's own baseline marks PPO(state) as "does not work"
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

run PullCubeTool-v1 ex48_PullCubeTool
run RollBall-v1     ex48_RollBall
run PullCube-v1     ex48_PullCube
echo "ALL EXPERTS ROUND4 DONE"
