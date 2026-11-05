#!/bin/bash
# Queued behind train_new3.sh: give PokeCube-v1 another 8M steps.
#
# Why: poke16m_best scored 95% on the 20-episode training eval but only 78% on
# the independent 100-episode check (the 20-episode estimate is biased high on
# a stochastic poking task). So this run (a) CONTINUES from that best ckpt
# instead of restarting, and (b) evaluates on 50 episodes, so the checkpoint
# selection itself is less noisy.
set -u
cd /root/cl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ms3
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

CK=ckpts/ppo_er_unified
LOG=logs
PY=/root/miniconda3/envs/ms3/bin/python

# Wait for the three-phase retrain to finish (it writes this marker last).
while [ ! -f "$LOG/train_new3.done" ]; do
  sleep 60
done

echo "===== [extra] PokeCube-v1 +8M from poke16m_best $(date '+%F %T') ====="
$PY ppo_er_unified.py \
  --task-sequence PokeCube-v1 --total-steps 8000000 \
  --init-ckpt "$CK/ppo_er_unified_task0_PokeCube-v1_poke16m_best.pt" \
  --num-envs 128 --num-steps 64 --num-minibatches 16 \
  --eval-freq 25 --eval-eps 50 --stop-sr 85 \
  --run-tag poke24m --seed 43 > "$LOG/train_poke24m.log" 2>&1
echo "===== [extra] done rc=$? $(date '+%T') ====="
grep -E '\[best\]|avg SR' "$LOG/train_poke24m.log" | tail -3
echo DONE > "$LOG/train_poke24m.done"
