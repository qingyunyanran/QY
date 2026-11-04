#!/bin/bash
# 5-TASK CONTINUAL LEARNING on the unified 48-dim observation interface.
#
# Every task in this sequence has a single-task expert whose success rate was
# MEASURED OVER 100 EPISODES (not the 20-episode estimate printed during
# training, which is noisy by +-11 pp at 1 sigma):
#
#   StackCube-v1       87%   grasping + stacking, two cubes, 100-step horizon
#   PullCube-v1       100%   drag the cube towards the base
#   PickCube-v1        98%   pick and place, 2.5 cm tolerance
#   LiftPegUpright-v1  95%   regrasp and rotate a 20 cm peg upright
#   PushCube-v1       100%   push the cube into the goal region
#
# Tasks that were audited and REJECTED, with the reason kept in the report:
#   StackPyramid / PlugCharger / TurnFaucet / AssemblingKits
#       SUPPORTED_REWARD_MODES == ("none"/"sparse") -> the only signal is the
#       0/1 success flag; StackPyramid measured reward == 0.000 for 490k steps.
#   PickSingleYCB
#       needs the YCB asset pack; huggingface.co is unreachable from this host.
#   PlaceSphere / PegInsertionSide / PullCubeTool / RollBall
#       dense reward but millimetre / contact-rich success tests; 8M steps gave
#       0%, 0%, 0% and <=10% respectively (ManiSkill's own PPO baseline marks
#       PegInsertionSide and PlugCharger as "does not work").
#   PokeCube
#       learned (72% over 100 episodes) but stays below the 80% expert bar.
#
# STRUCTURE
#   ER    -- every task enters by distilling its own single-task expert into the
#            shared actor while the ER replay buffer keeps the earlier tasks
#            (--anchor-task teacher table + replay_ratio 0.5).
#   SeqFT -- identical sequence, identical budgets, replay OFF and no teacher
#            distillation. The only difference is the retention mechanism, so
#            the gap between the two end-state matrices is the CL effect and not
#            a budget artefact.
#
# 2.5M steps/task = the budget of the archived easy-sequence run, so the new
# numbers stay comparable with it.
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

echo "=============== ER $(date) ==============="
python -u ppo_er_unified.py --task-sequence "$SEQ" --total-steps 2500000 \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --eval-freq 25 --eval-eps 20 \
  --anchor-task "$ANCH" --anchor-coef 2.0 \
  --run-tag c5_er > logs/c5_er.log 2>&1
echo "--------------- ER done (exit=$?) $(date) ---------------"

python -u eval_ckpt_5tasks.py "${LAST}_c5_er.pt" \
  --eps 100 --tasks "$SEQ" > logs/c5_er_eval100.log 2>&1
echo "--------------- ER eval100 done (exit=$?) $(date) ---------------"

echo "=============== SeqFT $(date) ==============="
python -u ppo_er_unified.py --task-sequence "$SEQ" --total-steps 2500000 \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --eval-freq 25 --eval-eps 20 \
  --seqft \
  --run-tag c5_seqft > logs/c5_seqft.log 2>&1
echo "--------------- SeqFT done (exit=$?) $(date) ---------------"

python -u eval_ckpt_5tasks.py "${LAST}_c5_seqft.pt" \
  --eps 100 --tasks "$SEQ" > logs/c5_seqft_eval100.log 2>&1
echo "--------------- SeqFT eval100 done (exit=$?) $(date) ---------------"

python -u report_cl_matrix.py \
  results/ppo_er_unified/ppo_er_unified_results_c5_er.json \
  results/ppo_er_unified/ppo_er_unified_results_c5_seqft.json \
  > logs/c5_matrix.log 2>&1
echo "ALL CL RUNS DONE $(date)"
