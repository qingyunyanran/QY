#!/bin/bash
# Evaluate one checkpoint on the 8 dense-reward tasks (no PlaceSphere).
# Usage: _ev8.sh <ckpt> [eps]   -> per-task SR and mean episode steps
set -u
PY=/root/miniconda3/envs/ms3/bin/python
cd /root/cl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CK=$1
EPS=${2:-100}
TASKS=PegInsertionSide-v1,StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1
TASKS=$TASKS,PushCube-v1,PokeCube-v1,PullCubeTool-v1
SD_EVAL_CKPT=$CK SD_STACK=${SD_STACK:-5} SD_EVAL_EPS=$EPS SD_TASKS=$TASKS \
  $PY _sd_eval_gpu.py 2>&1 | grep -E "SR=|mean="
