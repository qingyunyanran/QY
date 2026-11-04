# 5 任务持续学习（PPO + Replay / 蒸馏）实验报告

实验日期：2026-11-04　　代码：`cl/ppo_er_unified.py` + `cl/unified_obs.py`（48 维统一观测）
硬件：单卡 RTX 4080 SUPER，`sim_backend=gpu`，128 envs × 64 steps

---

## 1. 结论摘要

* 最终模型 **ER+MTD+Cons**：5 个任务 **全部 ≥ 80%**，平均 **93.4%**（100 回合/任务的独立进程评测）。

| 方法 | Stack | Pull | Pick | LiftPeg | Push | **mean** |
|---|---|---|---|---|---|---|
| 单任务专家（上界） | 87% | 100% | 98% | 95% | 100% | **96.0%** |
| SeqFT（纯序列微调，无 replay） | 0% | 0% | 0% | 0% | 100% | 20.0% |
| ER（replay，无教师） | 25% | 100% | 57% | 76% | 100% | 71.6% |
| ER + MTD（多教师蒸馏） | 76% | 100% | 99% | 92% | 100% | 93.4% |
| **ER + MTD + 巩固轮（交付）** | **85%** | **100%** | **94%** | **88%** | **100%** | **93.4%** |
| 归档 run `gpu5_48h`（同口径复评） | 87% | 100% | 88% | 84% | 100% | 91.8% |

* 三组对照的结论：
  1. **无 replay 的序列微调会造成彻底的灾难性遗忘**（前 4 个任务全归零，只有最后学的任务保留）。
  2. **纯 ER 能救回 71.6%**，但对"早期、低成功率的复杂任务"（StackCube）明显不够（87% → 25%）。
  3. **加入多教师蒸馏（MTD）后遗忘基本消除**（mean 93.4%），代价是最后一个任务（PushCube）训练时对早期任务的保护仍靠 replay + 蒸馏共同维持。
  4. **巩固轮**用 2.7M 步（4×0.3M 温习 + 1.5M StackCube 收尾）把唯一的短板 StackCube 从 76% 拉到 **85%**，其余任务损失 ≤5 pp，全部达标。
* 归档 run 的 94.8% 是 **20 回合**口径；换算到 100 回合口径为 **91.8%**，与本次 ER+MTD 的 93.4% 相比**并未退步**。原先"新 run 变差"的疑虑来自评测回合数不同，不是模型变差。

---

## 2. 实验设置

* **观测**：48 维统一语义槽位（`cl/unified_obs.py`），无 one-hot、无映射层；原生 48 维任务（StackCube）逐位不变。
* **任务序列**：`StackCube-v1 → PullCube-v1 → PickCube-v1 → LiftPegUpright-v1 → PushCube-v1`
  （StackCube 排第一：它是 48 维映射的参考实现，且是最难的任务，放在最前才能观察最严重的遗忘。）
* **每个任务预算**：CL 阶段统一 2.5M 步（305 iterations，`--task-steps 2500000`），远低于单任务 8M 上限。
* **算法**：PPO（128 envs × 64 steps = 8,192/update，16 minibatches）+ replay_ratio 0.5、buffer 200k/任务。
* **教师**：每个任务一份单任务专家 ckpt（见 §7），训练该任务时作为 anchor（`--anchor-task`, coef=2.0），并对 replay 中所有历史任务的样本做多教师蒸馏（`--mtd-coef 2.0`）。
* **评测口径**（重要）：所有汇报数字都来自 **独立进程、100 回合/任务** 的 `eval_ckpt_5tasks.py`。训练日志里的 20 回合评测噪声大且偏乐观（同一权重曾出现 65% / 35% / 25% 三种结果），**不作为结论依据**。

---

## 3. 训练期遗忘矩阵（20 回合，仅用于看趋势）

行 = 该任务训完时的评测，列 = 到此为止见过的任务。

### ER

| after | Stack | Pull | Pick | LiftPeg | Push |
|---|---|---|---|---|---|
| Stack | 85% | – | – | – | – |
| Pull | 85% | 100% | – | – | – |
| Pick | 70% | 100% | 100% | – | – |
| LiftPeg | 75% | 95% | 40% | 95% | – |
| Push | 65% | 100% | 70% | 70% | 100% |

### SeqFT

| after | Stack | Pull | Pick | LiftPeg | Push |
|---|---|---|---|---|---|
| Stack | 0% | – | – | – | – |
| Pull | 0% | 100% | – | – | – |
| Pick | 0% | 0% | 0% | – | – |
| LiftPeg | 0% | 5% | 0% | 100% | – |
| Push | 0% | 0% | 0% | 0% | 100% |

### ER + MTD

| after | Stack | Pull | Pick | LiftPeg | Push |
|---|---|---|---|---|---|
| Stack | 85% | – | – | – | – |
| Pull | 95% | 100% | – | – | – |
| Pick | 80% | 95% | 90% | – | – |
| LiftPeg | 85% | 100% | 90% | 85% | – |
| Push | 70% | 100% | 90% | 85% | 100% |

### ER + MTD + 巩固轮

| after | Stack | Pull | Pick | LiftPeg | Push |
|---|---|---|---|---|---|
| Stack（0.3M） | 85% | – | – | – | – |
| Pull（0.3M） | 75% | 100% | – | – | – |
| Pick（0.3M） | 75% | 100% | 95% | – | – |
| LiftPeg（0.3M） | 70% | 100% | 90% | 90% | – |
| Push + Stack 收尾（1.5M） | 85% | 100% | 90% | 85% | 100% |

---

## 4. 逐任务遗忘量（100 回合，相对单任务专家）

| Task | 专家 | ER | SeqFT | ER+MTD | **ER+MTD+Cons** |
|---|---|---|---|---|---|
| StackCube | 87% | −62 pp | −87 pp | −11 pp | **−2 pp** |
| PullCube | 100% | ±0 | −100 pp | ±0 | **±0** |
| PickCube | 98% | −41 pp | −98 pp | +1 pp | **−4 pp** |
| LiftPegUpright | 95% | −19 pp | −95 pp | −3 pp | **−7 pp** |
| PushCube | 100% | ±0 | ±0 | ±0 | **±0** |

---

## 5. 48 维与 42 维观测的区别（回答导师提问）

**不是"加了物体旋转位姿"。** 42 维里物体位姿本来就是 7 维（xyz + 四元数），这个格式没变。

真正的差别在尾部：42 维的 `[39:42]` 只有 3 维 goal，48 维把它扩成 `[39:48]` 的 9 维 tail，用来容纳：

* 第三个物体位姿（slot3，7 维）——StackPyramid / PlugCharger 需要 cubeC、goal_pose；
* 多相对向量（rel，3 个 × 3 维）——StackCube 需要 tcp−cubeA、tcp−cubeB、cubeB−cubeA；
* 辅助标量（aux）——如 PegInsertionSide 的 peg_half_size / hole_radius、RollBall 的 ball_vel。

**前 39 维在 42 和 48 两种布局里逐位相同**（qpos 9 / qvel 9 / tcp_pose 7 / slot1 7 / slot2 7）。简单任务（Push/Pull/Pick）新增的那 6 维是**纯零填充**，不携带信息；`slot3` 与 `goal+rel` 互斥（代码里有断言），所以扩容不会引入歧义。

对 StackCube 尤其关键：早期版本把它的 9 维当成"冗余线性差"删掉后，从零训练完全学不动，官方权重加载后 SR 直接从 97.6% 掉到 0%；补回 9 维后与官方 state obs 逐位相同，专家权重可免手术加载（87%）。`cl/verify_48d.py` 对 8 个任务做了逐位绑定校验，全绿。

---

## 6. 复杂任务淘汰清单（为什么最终只有这 5 个）

单任务预算 8M 步（`--total-steps 8000000 --stop-sr 85`），逐任务实测：

| 任务 | 结果 | 淘汰原因 |
|---|---|---|
| PegInsertionSide | 8M 跑满（72 min），**SR = 0%** | 卡在最后几毫米的插入精度；官方 baseline 同样标 ❌ |
| PullCubeTool | 8M 跑满，末次 **SR = 0%**（avg_R 100.6） | 工具使用型任务，稀疏/延迟奖励下 PPO 无法起步 |
| RollBall | 8M 跑满，best **SR = 10%** | 球体滚动控制对 delta-pos 动作不友好 |
| PlaceSphere | **SR = 0%** @ 6.3M | 成功判据极严（xy ≤ 5 mm, z ≤ 5 mm） |
| PokeCube | 100 回合 **72%**（20 回合曾报 85%） | 达不到 80% |
| StackPyramid / PlugCharger / TurnFaucet / AssemblingKits | 未跑 | 纯稀疏奖励或无 dense reward，PPO 无梯度信号 |
| PickSingleYCB | 代码已接通 48 维槽位并校验 | 服务器无外网，YCB 资产下载失败 |
| PushT | 运行时报错 | `render_backend="none"` 下 `render_shapes` 为 None |
| TwoRobotStackCube | 未跑 | action 是 dict，与单臂 8 维动作接口不兼容 |

入选的 5 个任务都能在 8M 内达到 ≥87%：StackCube 87% / PullCube 100% / PickCube 98% / LiftPegUpright 95% / PushCube 100%。

---

## 7. 代码改动

1. **48 维通用槽位映射**（`cl/unified_obs.py`）：`_tail_plan(cfg)` 按任务配置生成 tail 计划（`slot3` / `goal+rel` / `aux`，互斥 + 溢出断言 `off <= 48`）；新增 `verify_48d.py` 校验 8 个任务逐位绑定。
2. **MTD（Multi-Teacher Distillation）**（`cl/ppo_er_unified.py`）：
   * `teacher_nets[task_idx]` 在加载 anchor 时登记该任务专家；
   * replay 按 `task_ids` 分组（`torch.unique`），每组用对应教师的 `actor_mean` 做 MSE；
   * `loss += mtd_coef * mtd_loss`，日志追加 `mtd={...}`；
   * CLI `--mtd-coef`（默认 0.0，关闭时行为与改动前完全一致）。
3. **`--init-ckpt`**：任务 0 开始前用 `strict=True` 完整加载 state dict（含 critic），与 `--pretrain-ckpt`（仅 actor、冻结）区分，用于巩固轮续训。
4. results JSON 增加 `anchor_coef` / `mtd_coef` / `init_ckpt` 字段，便于回溯。
5. best-ckpt 保留：评测刷新最高 SR 时另存 `*_best.pt`。

---

## 8. 复现命令

```bash
# 单任务专家（每个任务一遍）
python cl/ppo_er_unified.py --task StackCube-v1 --total-steps 8000000 --stop-sr 85 \
    --run-tag expert_stack

# 三条 CL 链路
bash c5_cl.sh        # ER  → eval100 → SeqFT → eval100 → 遗忘矩阵
bash c5_er_mtd.sh    # 等 c5_cl 结束 → ER+MTD → eval100
bash c5_report.sh    # 等 mtd 结束 → 5 专家 100 回合复评 → 汇总
bash c5_cons.sh      # 归档 ckpt 同口径复评 → 巩固轮（0.3×4 + 1.5M）→ eval100

# 单次评测
python cl/eval_ckpt_5tasks.py --ckpt <path> --eps 100
```

---

## 9. 限制与注意事项

1. **评测回合数必须统一**。20 回合标准误约 ±9 pp，100 回合约 ±4 pp。同一权重在不同回合数下可差 10+ pp（StackCube 65% vs 25% 的极端例子）。本报告只用 100 回合数字。
2. **含 anchor 时 `pg_loss` 被置零**（框架原有设计）：因此 ER+MTD 实际上是 **replay + 多教师策略蒸馏**（critic 仍由 value regression 训练），不是纯 PPO 策略梯度。若要保留 PPO 梯度，需要把"教师"与"anchor"解耦——目前两者共用同一张表，属于已知设计限制。
3. **单 seed**。每条链路只跑一次，pp 级差异（如 Pick 99% vs 94%）落在噪声范围内，不应过度解读。
4. **巩固轮的 1.5M 收尾把 StackCube 放在最后**，等价于再做一次序列微调；它能奏效是因为同时开了 4 个历史任务的 0.3M 温习 + 全教师 MTD，不能简单理解为"多训 StackCube 就行"。
5. 归档 run 与本 run 之间有三个变量同时变化（任务顺序、每任务预算 2/2/6/4/2.5M vs 统一 2.5M、纯 ER 从零 PPO vs 蒸馏式），因此两者差异不能归因到单一因素。

---

## 10. 工件清单

| 文件 | 说明 |
|---|---|
| `ckpts/ppo_er_unified/*_c5_er.pt` | ER 终态（5 任务，末任务权重） |
| `ckpts/ppo_er_unified/*_c5_seqft.pt` | SeqFT 终态 |
| `ckpts/ppo_er_unified/*_c5_er_mtd.pt` | ER+MTD 终态 |
| **`ckpts/ppo_er_unified/*_c5_cons.pt`** | **交付模型（ER+MTD+巩固轮，5/5 ≥ 80%）** |
| `ckpts/ppo_er_unified/*_gpu5_48h.pt` | 归档 run 权重（100 回合复评 91.8%） |
| `logs/c5_er.log` / `c5_seqft.log` / `c5_er_mtd.log` / `c5_cons.log` | 训练日志 |
| `logs/c5_*_eval100.log` | 100 回合复评日志 |
| `logs/c5_summary.md` | 自动生成的对照表与遗忘矩阵 |
| `results/ppo_er_unified/ppo_er_unified_results_c5_*.json` | 结构化结果（含 task_sequence、anchor_coef、mtd_coef、init_ckpt） |
