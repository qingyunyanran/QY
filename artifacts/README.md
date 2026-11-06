# artifacts — 8 任务持续学习的可复现产物

对应报告：[`REPORT_CL8.md`](../REPORT_CL8.md)。所有 SR 均为**独立进程 100 回合**评估（`eval_ckpt_5tasks.py --eps 100`）。

任务序列（head 是**位置索引**，顺序不能改）：

```
0 PegInsertionSide → 1 StackCube → 2 PullCube → 3 PickCube
→ 4 LiftPegUpright → 5 PushCube → 6 PokeCube → 7 PullCubeTool
```

## 文件

| 文件 | 说明 |
|---|---|
| `ckpts/..._c8v5_cons.pt` | **主结果**：8/8 达标，均值 92.0%。c8v4（Pick 段放开底座 ×0.2）后经 7 任务蒸馏重对齐 |
| `ckpts/..._c8v6_noanchor.pt` | 对照：Poke / Tool **不用专家教师**（各 4M 真实 PG），8/8 仍达标，均值 91.4% |
| `results/..._c8v5_cons.json` | 训练过程记录（逐 eval 的 per-task SR、步数、早停点）|
| `results/..._c8v6_noanchor.json` | 同上 |
| `logs/..._eval100.log` | 100 回合终评的原始输出（可直接核对报告表格）|

SHA256（前 16 位）：

```
79548C3935A67E7B  ckpts/ppo_er_unified_task7_PullCubeTool-v1_c8v5_cons.pt   3668865 B
790F22F8F0F8678D  ckpts/ppo_er_unified_task7_PullCubeTool-v1_c8v6_noanchor.pt 3669101 B
83620EEF31A65068  logs/c8v5_cons_eval100.log   1643 B
2125A546C5FFF396  logs/c8v6_noanchor_eval100.log 1647 B
9D796C709F692888  results/ppo_er_unified_results_c8v5_cons.json  113001 B
EF449C73AC972040  results/ppo_er_unified_results_c8v6_noanchor.json 225433 B
```

## 终评成绩

| 任务 | c8v5_cons（有教师）| c8v6_noanchor（Poke/Tool 无教师）|
|---|---|---|
| PegInsertionSide | 84 | 87 |
| StackCube | 89 | 81 |
| PullCube | 100 | 100 |
| PickCube | 96 | 95 |
| LiftPegUpright | 84 | 91 |
| PushCube | 100 | 100 |
| PokeCube | 87 | **80**（无教师）|
| PullCubeTool | 96 | **97**（无教师）|
| **均值** | **92.0** | **91.4** |

## 如何复现评估

```bash
python eval_ckpt_5tasks.py artifacts/ckpts/ppo_er_unified_task7_PullCubeTool-v1_c8v5_cons.pt \
  --tasks PegInsertionSide-v1,StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1,PokeCube-v1,PullCubeTool-v1 \
  --multi-head --private-base PegInsertionSide-v1 --eps 100
```

环境：ManiSkill3（SAPIEN GPU），RTX 4080 SUPER，Python 3（conda env `ms3`）。
`--tasks` 的顺序必须与上表一致；`--private-base PegInsertionSide-v1` 对应 `private_nets.0`（Peg 排序列第一）。
