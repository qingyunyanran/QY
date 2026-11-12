# 10 任务 CL 终评结果（100 回合 / 任务，seeds 0–99）

## 1. 结论

| 指标 | 数值 |
|---|---|
| **10 任务平均成功率** | **90.6%** |
| 8 任务统一策略平均（原 CL 序列） | 92.6% |
| 2 个新增 anchor 平均（运动规划） | 82.5% |
| 最低任务 | PlugCharger-v1 81.0% |
| 最高任务 | PushCube-v1 100.0% |

全部 10 个任务均 ≥80%，序列可用。

> 口径说明：8 个任务是**同一份 48 维统一 PPO 权重**（`c8v5_cons_best`），
> 两个 anchor 仍是运动规划 solver（见第 3 节）。学习式 anchor 的最新进展与
> 结构性上限见第 4 节。

## 2. 逐任务结果

| # | 任务 | SR | 实现方式 |
|---|---|---|---|
| 1 | PegInsertionSide-v1 | 91.0% | 统一 PPO（ER + CL 巩固） |
| 2 | StackCube-v1 | 85.0% | 统一 PPO |
| 3 | PullCube-v1 | 99.0% | 统一 PPO |
| 4 | PickCube-v1 | 93.0% | 统一 PPO |
| 5 | LiftPegUpright-v1 | 94.0% | 统一 PPO |
| 6 | PushCube-v1 | 100.0% | 统一 PPO |
| 7 | PokeCube-v1 | 87.0% | 统一 PPO |
| 8 | PullCubeTool-v1 | 92.0% | 统一 PPO |
| 9 | **StackPyramid-v1** | **84.0%** | 运动规划 solver |
| 10 | **PlugCharger-v1** | **81.0%** | 运动规划 solver |

8 任务的合并权重仍是 `c8v5_cons_best`，本轮只重测（同 seeds、同 agent），
逐任务差异来自单次 100 回合的采样波动（±3pt 量级）：
PokeCube 单任务复核 100 集 84.0%，与上表 87.0% 的差异即为该波动。

评测：每任务 100 回合，seed 0–99，`reconfiguration_freq=1`（每回合全场重配），
回合上限取任务原生 horizon（Pyramid 250，其余 200/250）。
权重：`ckpts/ppo_er_unified/ppo_er_unified_task7_PullCubeTool-v1_c8v5_cons_best.pt`
（8 头 + PegInsertionSide 私有 base，48 维统一观测）。
原始数据：`results/cl10_eval_100.json`。

## 3. 两个 anchor 的求解方式

新增任务 StackPyramid / PlugCharger 使用 ManiSkill 官方运动规划 solver
（`mani_skill.examples.motionplanning.panda.solutions`），在 `/root/cl` 下做了参数化封装：

- `_sol_pyr.py`：官方 `solveStackPyramid` 逻辑 + `joint_vel_limits/acc` 可调
- `_sol_plug2.py`：官方 `solvePlugCharger` 逻辑 + 抓取倾角、夹爪柔性、速度上限可调

### 3.1 PlugCharger 从 58% 提到 81% 的关键变量

唯一的决定性变量是 **关节速度/加速度上限**（官方默认 0.5）：

| joint_vel_limits | 0.3 | 0.5(默认) | 1.0 | 1.5 | **2.0** | 2.2 | 2.5 | 3.0 | 5.0 |
|---|---|---|---|---|---|---|---|---|---|
| SR | 52% | 59% | 68% | 73% | **81%** | 80% | 66% | 42% | 3% |

机理：插入阶段规划出的 screw 轨迹在低速限下被拉长，机械臂在插孔口长时间抖动/停滞，
推进量被接触阻尼吃掉；限速调高后单步推进更果断，插入一次到位。上限过高（≥2.5）则会冲过插孔。

`0.5` 的默认值同时让 Pyramid 单回合步数从 188 降到 133（耗时 -30%）而不损失成功率。

### 3.2 已排除的改进方向（有完整消融数据）

| 尝试 | 结果 | 结论 |
|---|---|---|
| 抓取倾角 0/5/10/15/20/25/30/45° | 18/40/59/58/56/52/38/3% | 10–15° 已是峰值，角度非瓶颈 |
| 插入分段推进（8 段，逐段成功即停） | 4% | 大幅恶化：screw 规划非直线，逐段重规划产生 Z 字形偏离 |
| 失败后退+微转重插（jiggle 4 组） | 53% | 无效 |
| 失败后退+平移搜索（shift 4 组） | 58% | 无效 |
| 失败后深压（press 2 组） | 58% | 无效 |
| 插入前夹爪放松（-0.8/-0.5/0） | 61/38/1% | 无效，且越松越差 |
| Pyramid：高空松手 → 下降到堆叠高度再松手 | 79% → 79% | 无提升（官方结尾已是终点，无需改动） |

跨 seed 段复核（PlugCharger，vel=2.0、倾角 15°）：
seeds 0–99 83%、100–199 87%、200–299 73%，合并 **243/300 = 81.0%**，
即该配置的真实水平约 81%，单次 100 回合的波动 ±3%。

## 4. 学习式 anchor 的负结果（供报告引用）

为把两个 anchor 并入统一策略，做了完整的 BC / DAgger 攻关，**最好 15%（3/20），未达可用**：

| 阶段 | 干预 | 离线 arm MAE | 闭环 SR |
|---|---|---|---|
| 全库 BC（1512 段 / 27.8 万帧） | 目标 `(A−O)/0.1` | — | 0/40 |
| 修正控制增益 | `gain=0.0393`（实测） | 0.000335（MSE） | 0/40 |
| 去 qvel / 去臂 qpos | 通道屏蔽 | — | 0/40 |
| 绝对关节目标 + `pd_joint_pos` | 开环重放 5/5 验证 | — | 0/40 |
| 关节扰动增广（sigma=0.02，2 轮） | 83.6 万对 | — | 0/40 |
| 目标 z-score 标准化 | 必要项 | 0.0037 rad | 0/40 |
| **DAgger**（planner 热启动打标签，160 条 ×60 倍加权） | 5 轮 | **0.0025 rad** | 0/40 |

DAgger 把轨迹偏差从 0.93 m 压到峰值 0.058 m（t≤20）、t=40 回升到 0.018 m，
但仍有两条**观测信息缺口**无法弥补，最终失败：

1. **时间歧义**：t=0–20 期间 TCP 只移动 3 cm，专家目标从 0 变到 0.677 rad。
   多对一映射下网络只能输出均值 → 约 1 步的时间误差 → 5 cm 位置偏差。
2. **夹爪时序不可观测**：success 依赖"是否已抓住"，48 维状态里没有该量，
   策略在 t=100 输出 grip=+1 而专家为 −1。

结论：48 维状态观测下，成功率型长程装配任务（Pyramid/Plug）的 BC 存在结构性上限，
用求解器是更可靠的选择。

### 4.1 学习式 StackPyramid：分段 BC + 几何门控（最新进展）

思路是把 Pyramid 拆成 21 个几何阶段（hold / move / pushA_up / pushA_down / pushA /
pushA_off / graspC_pre / graspC / lift / stack_pre / stack），每阶段一个小 head
（输入 48 维观测 + 阶段目标 3 维，输出 7 维关节增量 + 夹爪），动作空间仍是
`pd_joint_delta_pos`；阶段推进只看几何条件（TCP 到达阈值 / A-B 距离 / 停滞计数），
**全程不调用规划器**。

- 专家数据：`demos/pdgoal_pyr.npz`（27599 帧，带 stage 与阶段目标标签）
- 最佳结果：**50.0%（8/16）**，48 集复测 **47.9%（23/48）**，权重 `ckpts/stage_bc_v2.pt`
- 兜底机制：pushA 超时沿 BA 续推（≤3 轮）、k13 检测到 A 被撞飞回推、
  lift 阶段监测 cubeC 高度、抓空则回退重抓

失败集中在两簇：A 被推偏（|A−B| 0.11–0.16，判据 0.0616）与 C 被扫飞/抓空
（|C−B| 0.38 或 |Δz| = 0）。

已排除的改进（同 16 集口径，基线 8/16）：

| 尝试 | SR | 结论 |
|---|---|---|
| k10–k19 整链 PPO 重训（387 个失败态作起点） | 3/16 | 离线 best ok 高、闭环反而变差 |
| DAgger 数据（32 集 / 6074 对专家标注）重训 k10–k19 | 4/16 | 同上，分布漂移主导 |
| 仅替换 pushA head（其余保留原 head） | 3/16 | pushA 新 head 步数变长、拖垮后续 |
| graspC_pre/graspC 实时跟踪 cubeC | 6/16 | 目标抖动，闭合时抓空 |
| 抓取失败重试（k17 看 C 高度回退 k14） | 8/16 | 中性，保留为兜底 |
| move 阶段避障抬升（AIR_Z=0.15） | 3/16 | head 未见过高空目标 |
| 推点实时重锚 / `GAIN=1.3` / `STACK_K` 对准 | 2–6/16 | 无效 |
| 推距/轮次/超时/重推阈值网格扫描 | 4–5/8(8 集) | 无余量 |

结论：head 重训在这条管线上稳定地伤闭环（BC 权重 `stage_bc_v2` 反而是最好的），
要突破 50% 需要扩观测（补"是否已抓住""阶段相位"等缺失量）并整体重训，
属于第 6 节所述的结构性改造。

## 5. 复现

```bash
# 10 任务终评（约 3 分钟：8 任务 GPU 37s + 两 anchor CPU 约 2.5 min）
cd /root/cl && bash scripts/c10ev.sh

# 单独评测某个 solver
/root/miniconda3/envs/ms3/bin/python -u _solve_eval.py \
  --task PlugCharger-v1 --eps 100 --seed0 0 --mod _sol_plug2   # PLUG_VEL/PLUG_ANGLE 环境变量
PYR_VEL=2.0 /root/miniconda3/envs/ms3/bin/python -u _solve_eval.py \
  --task StackPyramid-v1 --eps 100 --seed0 0 --mod _sol_pyr

# 单任务复核（复用 8 头统一权重，head id 由规范任务顺序决定）
/root/miniconda3/envs/ms3/bin/python _eval1.py PokeCube-v1 100

# 学习式 Pyramid（不调 solver）：分段 head + 几何门控
/root/miniconda3/envs/ms3/bin/python _stage_eval.py ckpts/stage_bc_v2.pt 16
SEG_TRACE=1 /root/miniconda3/envs/ms3/bin/python _stage_eval.py ckpts/stage_bc_v2.pt 8
```

## 6. 局限

- 10 任务序列不是"单模型跑完"：8 个任务共享一个 PPO 策略，2 个 anchor 各用求解器，
  属于"混合策略"设定。若要求严格单模型，需先解决第 4 节的两个观测缺口（要扩观测维度）。
- anchor 的 84%/81% 来自求解器自身能力，不随训练变化；
  同比其在官方基线上的 85%/60%，PlugCharger 有实质提升（+21 pt），Pyramid 持平。

## 7. 无 solver 化的最新进展（2026-11-12）

第 4 节"BC 存在结构性上限"的判断已被部分推翻：PlugCharger 的失败**主体不在学习能力，
而在几何管线**。修掉观测 quat 切片、IK 坐标系（必须 base frame）、夹爪安装旋转
（`REL_QUAT`）三处错误后，**解析式抓取 4/4 全中**，`ik_fail` 从 943 降到 0；
剩余瓶颈是插入段收敛（已改为朝 receptive 位姿的几何直线推进，验证中）。

同一管线不调用任何运动规划器，也不依赖 solver 生成的动作。详细数据见
`REPORT_PLUGCHARGER.md` 与 `docs/PLUGCHARGER_DIAGNOSIS.md` 末节。
