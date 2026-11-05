#!/bin/bash
# 100-episode SR re-evaluation of the three candidate anchors, one process each
# (a fresh sim per task, so no cross-task state can leak into the numbers).
set -u
cd /root/cl
PY=/root/miniconda3/envs/ms3/bin/python
CK=ckpts/ppo_er_unified

run() {
  echo "=== $1  ($2) ==="
  $PY eval_ckpt_5tasks.py --eps 100 --action-dim 8 --tasks "$1" "$2" 2>&1 \
    | grep -E "load:|SR=|mean SR|Error|error" 
}

run PokeCube-v1     $CK/ppo_er_unified_task0_PokeCube-v1_ex48c_PokeCube_best.pt
run RollBall-v1     $CK/ppo_er_unified_task0_RollBall-v1_ex48_RollBall_best.pt
run PullCubeTool-v1 $CK/ppo_er_unified_task0_PullCubeTool-v1_ex48_PullCubeTool.pt
echo "=== done ==="
