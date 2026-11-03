# 5 任务持续学习结果（48 维统一观测）

任务序：**PushCube → PullCube → PickCube → LiftPegUpright → StackCube**
（观测：48 维统一语义槽位；控制：`pd_joint_delta_pos`；GPU：RTX 4080 SUPER）

## 1. 终评（每任务 100 episodes，independent re-eval）

用独立进程、独立环境重新加载权重评测（`cl/eval_ckpt_5tasks.py`），不复用训练日志中的数字：

| 任务 | ER（replay 0.5） | SeqFT（replay 0） |
|---|---|---|
| PushCube-v1 | **100%** | 0% |
| PullCube-v1 | **100%** | 0% |
| PickCube-v1 | **92%** | 0% |
| LiftPegUpright-v1 | **90%** | 0% |
| StackCube-v1 | **92%** | 90% |
| **平均** | **94.8%** | **18.0%** |

- 两组预算、任务序、优化器调度（128 envs × 64 steps，16 minibatch × 4 epoch，lr 3e-4，γ=0.8，λ=0.9，clip 0.2）、
  seed、StackCube 蒸馏 anchor 完全一致，**唯一差别是 `replay_ratio`（0.5 vs 0）**。
- 步数：`2M / 2M / 6M / 4M / 2.5M`，合计 16.5M。
- 耗时：ER 约 105 min（2.0–3.0k sps），SeqFT 约 95 min（2.7–3.7k sps）。

## 2. 遗忘矩阵（行 = 学完该任务后；列 = 评测任务；空 = 尚未学过）

### ER

| 学完 | Push | Pull | Pick | LiftPeg | Stack |
|---|---|---|---|---|---|
| PushCube | 100 | — | — | — | — |
| PullCube | 100 | 100 | — | — | — |
| PickCube | 100 | 100 | 100 | — | — |
| LiftPegUpright | 100 | 100 | 90 | 85 | — |
| StackCube | 100 | 100 | 90 | 80 | 95 |

### SeqFT

| 学完 | Push | Pull | Pick | LiftPeg | Stack |
|---|---|---|---|---|---|
| PushCube | 100 | — | — | — | — |
| PullCube | 15 | 100 | — | — | — |
| PickCube | 30 | 100 | 100 | — | — |
| LiftPegUpright | 30 | 0 | 0 | 100 | — |
| StackCube | 0 | 0 | 0 | 0 | 80 |

SeqFT 的遗忘严格按任务顺序倒序发生（学 Pull 掉 Push、学 Pick 掉 Pull、学 LiftPeg 掉 Pick、学 Stack 掉 LiftPeg），
ER 在整个序列中旧任务保持 90–100%。

## 3. 关键：StackCube 怎么进共享 actor

5 个任务共用同一个 actor，因此**热启动会整体覆盖策略**（实测：4 个旧任务瞬间归零，
冻结 actor 时 KL=0.0000 也可直接印证）。改为**冻结教师蒸馏**：

- 官方 48 维检查点只作教师，不写入 actor；
- StackCube 批次上加 `coef * MSE(actor_mean(obs), expert_mean(obs))`，`coef=2.0`；
- 蒸馏阶段**关掉 PPO 策略项**（优势来自仍在拟合的价值头，与教师信号互相抵消）；
- 蒸馏阶段**放开 PPO 的 KL 早停门限到 0.5**。

最后一条是决定性因素：KL 门限 0.02 时每个 epoch 在第 1 个 minibatch 就 early-stop，
等效学习率只剩 1/4，蒸馏只会停在 MSE≈0.5（StackCube 0%）；放开后 MSE 0.88 → 0.030，
StackCube 从 0% 到 92–100%。

## 4. 复现

```bash
# ER
bash cl/scripts/stage2_48.sh
# SeqFT 对照
bash cl/scripts/stage3_48.sh
# 独立复评（100 eps/task）
python cl/eval_ckpt_5tasks.py ckpts/..._gpu5_48h.pt --eps 100
```

产物：`cl/results/ppo_er_unified/ppo_er_unified_results_gpu5_48h{,_seqft}.json`
（权重 `*.pt` 被 `.gitignore` 排除）。

## 5. 注意事项

1. **评测采样量**：20 eps 的抖动可达 ±10 个点（StackCube 在 20 eps 报 90%，50 eps 报 74%，100 eps 报 92%）。
   对外数字统一用 100 eps。
2. **结果 JSON 结构**：`per_task_results[*].eval_log` 指向同一个累积列表（训练脚本把同一 list 塞进每个条目），
   取遗忘矩阵时需按 `task` 字段分组取最后一条，不能直接取 `eval_log[-1]`。
