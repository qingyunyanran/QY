# 8 任务 Benchmark 说明（ManiSkill3 稠密奖励子集）

> 版本：2026-11-19
> 范围：**8 个稠密奖励操作任务**（PegInsertionSide / StackCube / PullCube / PickCube /
> LiftPegUpright / PushCube / PokeCube / PullCubeTool）。PlaceSphere 已从套件中移除
> （原因见 §8）；9 任务严格单头的历史记录保留在 `CL_10TASK_RESULTS.md`。
> 本文档模仿 ManiSkill 官方 Benchmarking 页的写法：先给**任务表 + 评测协议**，
> 再给**baseline 结果表**，最后说明每个 baseline 的做法与复现命令。

---

## 1. 套件概览

| 项 | 值 |
|---|---|
| 仿真器 | ManiSkill3（SAPIEN / PhysX） |
| 机器人 | Panda（Franka），7 关节 + 2 指夹爪 |
| 任务数 | 8（全部稠密奖励、全部单物体或双物体桌面操作） |
| 统一动作空间 | 8 维 `pd_joint_delta_pos`（7 关节增量 + 1 夹爪） |
| 统一观测空间 | 48 维语义槽位（§3）+ 3 或 5 帧堆叠 |
| 评测规模 | **100 回合 / 任务**，seeds 0–99 |
| 评测后端 | **GPU**（PhysX GPU）；同 ckpt 在 CPU 上 PegInsertionSide 只有 15% |
| 单模型要求 | 单策略头、固定参数量、无任务标签、无 one-hot |

**为什么只保留这 8 个。** 它们是唯一能同时满足"稠密奖励 + 可由纯 BC 单头表达 +
专家成功率 ≥80%"的任务。StackPyramid 的专家被动作分辨率卡住（见
`DATASET_AND_METHODS.md` 的任务覆盖表）；PlugCharger 的最优解是"策略接近 + IK 安全网"，不是纯策略；
RollBall 的专家本身 0%、采不到数据。这三个任务无法进入纯 BC 口径，已剔除。

---

## 2. 任务表

| # | Task | 目标（一句话） | Success 判据 | `max_episode_steps` | 专家平均步长 |
|---|---|---|---|---|---|
| 1 | `PegInsertionSide-v1` | 把圆柱 peg 插入侧向开口的盒子孔位 | peg 插入深度 > 阈值且未脱出 | 100 | 40 |
| 2 | `StackCube-v1` | 抓 cubeA、叠到 cubeB 上、松手稳定 | 两 cube 接触稳定且 cubeA 高于 cubeB | 100 | 19 |
| 3 | `PullCube-v1` | 把 cube 沿桌面拉到目标附近 | cube 与目标球距离 < 阈值 | 50（默认） | 18 |
| 4 | `PickCube-v1` | 抓起 cube 到目标球位置 | cube 在目标球半径内且已抬起 | 50（默认） | 22 |
| 5 | `LiftPegUpright-v1` | 把倒下的 peg 立起来 | peg 轴与竖直方向夹角 < 阈值 | 50（默认） | 25 |
| 6 | `PushCube-v1` | 把 cube 推入目标区域 | cube 与目标球距离 < 阈值 | 50（默认） | 13 |
| 7 | `PokeCube-v1` | 把 cube 推到目标区域（接触点低、friction 低） | cube 与目标球距离 < 阈值 | 100 | 28 |
| 8 | `PullCubeTool-v1` | 用杆状工具把 cube 拉近 | cube 与目标球距离 < 阈值 | 100 | 34 |

> `max_episode_steps` 取自 `ppo_er_unified.TASK_INFO`。StackCube / PegInsertionSide /
> PokeCube / PullCubeTool 覆盖了注册默认值（50 → 100），因为专家本身就要 19–40 步、
> 学生更慢；另外 4 个任务保留原生 50。**评测与训练用的是同一套 horizon**，与官方
> benchmark 默认值不同，因此本文的 SR 不能直接与官方 leaderboard 横向比较（§7）。

**全部 8 个任务使用同一个 `pd_joint_delta_pos`。** 官方 baseline 里 Push/Pick 用
`pd_ee_delta_pose`、Pull/Poke 用 joint 增量，混用会让"统一接口"不成立；实测 joint
增量在 8 个任务上都能训到 ≥80%，所以统一为 joint 增量。

---

## 3. 统一观测：48 维语义槽位

原始 state-obs 各任务维度不同（Push/Pull 35、Pick 42、LiftPegUpright 32），差异只来自
对象数量与"相对向量"（`tcp_to_obj`、`obj_to_goal` 等）。因此按**物理含义**对齐到固定槽位：

| 槽位 | 维度 | 内容 |
|---|---|---|
| `[0:9]` | 9 | qpos：7 关节 + 2 指 |
| `[9:18]` | 9 | qvel：同序 |
| `[18:25]` | 7 | `tcp_pose`（xyz + quat） |
| `[25:32]` | 7 | 物体槽 1（被操作主物体） |
| `[32:39]` | 7 | 物体槽 2（第二物体 / 工具；单物体任务置零） |
| `[39:48]` | 9 | 任务尾部：目标 xyz / 相对向量 / 任务参数（见下） |

尾部 `[39:48]` 的填充规则：

| 任务 | 填充内容 |
|---|---|
| Push / Pull / Pick / Poke | 目标位置 xyz（3）+ 零 |
| LiftPegUpright | 全零（成功条件隐含在槽 1 的姿态里） |
| StackCube | 三条原生相对向量 `tcp→cubeA`、`tcp→cubeB`、`cubeA→cubeB`（9 维，与官方 48 维逐位相同） |
| PegInsertionSide | `peg_half_size`(3) + `box_hole_radius`(1)，均逐回合随机化 |

**关键测量**：把这 48 维零填充扩到 56 维，单头 SR 从 92.4% 到 93.5%（§5），
说明"加宽输入槽位"本身零成本——这是后来能承载新任务的前提。

---

## 4. 数据集

### 4.1 来源

| 项 | 值 |
|---|---|
| **初始数据**标签来源 | **官方 mplib 运动规划 solver**（`mani_skill/examples/motionplanning/panda/solutions/*.py`），不训练、无参数；8 任务 SR 85–100，均值 92.6（§4.2） |
| **DAgger** 标签来源 | `c8v5_cons`——一个 8-head PPO 的 CL 模型，8 任务均值 **92.0%**；在学生访问到的状态上现算标签 |
| 为什么不全程用 solver | solver 每步重规划几十–几百 ms，且在学生漂移状态上不一定有解；PPO 网络前向是微秒级，能对任意状态打标签 |
| 格式 | 每个任务一个 `.npz`：`obs (N,48) float32`、`act (N,8) float32`、`ep`、`tid` |
| 初始数据规模 | **7,425 轨迹 / 182,328 帧**（`demos/sd_8task_1k.npz`，8 任务，见下表） |
| DAgger 数据池 | 各轮累计 **22,625 条轨迹**；全轮合并 **732,708 帧**（`demos/sd_8t_dag_66.npz`） |
| 交付链训练集 | `demos/sd_dagger_r5.npz`，**121,769 帧**（48 维，tid 0–7）；各任务 6,295–25,485 帧，**不均衡**，训练时按任务均衡采样 |
| 保存 | 全部帧入库（含失败/走偏），不按成功过滤 |

| Task | 轨迹数 | 帧数 | 平均步长 |
|---|---|---|---|
| PegInsertionSide | 910 | 36,280 | 40 |
| StackCube | 865 | 16,520 | 19 |
| PullCube | 999 | 17,542 | 18 |
| PickCube | 936 | 20,919 | 22 |
| LiftPegUpright | 914 | 22,549 | 25 |
| PushCube | 1,000 | 12,590 | 13 |
| PokeCube | 831 | 23,113 | 28 |
| PullCubeTool | 970 | 32,815 | 34 |
| **合计** | **7,425** | **182,328** | — |

**交付链训练集（r5）各任务帧数 vs 初始数据**（`np.unique(tid)` 实测）：

| tid | Task | 初始 npz 帧数 | r5 训练集帧数 | 比例 |
|---|---|---|---|---|
| 0 | PegInsertionSide | 36,280 | 25,485 | 0.70 |
| 1 | StackCube | 16,520 | 16,779 | 1.02 |
| 2 | PullCube | 17,542 | 8,883 | 0.51 |
| 3 | PickCube | 20,919 | 12,003 | 0.57 |
| 4 | LiftPegUpright | 22,549 | 14,391 | 0.64 |
| 5 | PushCube | 12,590 | 6,295 | 0.50 |
| 6 | PokeCube | 23,113 | 19,424 | 0.84 |
| 7 | PullCubeTool | 32,815 | 18,509 | 0.56 |
| — | **合计** | **182,328** | **121,769** | **0.67** |

> r5 训练集帧数合计**低于**初始数据，且 StackCube 反而略增（1.02）——说明它是
> "按每任务配额抽取的合并集"，不是全量拼接。因此不能拿"累计轨迹数 × 平均步长"
> 反推帧数；表里的帧数一律以 npz 实测为准。

### 4.2 官方 solver 成功率（初始数据采集侧）

| Task | SR |
|---|---|
| PushCube | 100 |
| PullCube | 99 |
| LiftPegUpright | 94 |
| PickCube | 93 |
| PullCubeTool | 92 |
| PegInsertionSide | 91 |
| PokeCube | 87（我们自建规划器，官方无此 solver） |
| StackCube | 85 |
| **8 任务均值** | **92.6** |

> 这是**初始数据侧的专家上限**。DAgger 的老师换成 `c8v5_cons`（8 任务均值 92.0）
> 之后，学生 r5 到 92.4%，已经与老师打平——说明这条链的天花板由老师决定，
> 而不是网络容量。

---

## 5. Benchmark 结果

### 5.1 逐任务 Success Rate

100 回合 / 任务，seeds 0–99，GPU 后端，`STACK=5`（solver 行为单帧，无需堆叠）。

| Method | 模型 | Peg | Stack | Pull | Pick | LiftPeg | Push | Poke | Tool | **mean** | min | ≥80% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Official solver (mplib) | 规划器，无参数 | 91 | 85 | 99 | 93 | 94 | 100 | 87 | 92 | **92.6** | 85 | 8/8 |
| BC + DAgger r5（48 维，`sd_single_d5`） | 单头 768 hidden / STACK=5（240 维） | 89 | 83 | 100 | 97 | 92 | 100 | 84 | 94 | **92.4** | 83 | 8/8 |
| **BC + DAgger r5，拓宽到 56 维（交付）** | **单头 768 hidden / STACK=5** | **87** | **87** | **100** | **97** | **92** | **100** | **87** | **98** | **93.5** | **87** | **8/8** |

交付模型：`ckpts/sd_single_d5_p56.pt`。逐任务区间 **87–100**；最弱的是
PegInsertionSide、StackCube、PokeCube 三项并列 87，仍高于 80% 门槛 7 个点。

### 5.2 全部 baseline 均值对照

| # | Method | 学习 | 模型形态 | 任务数 | 8 任务均值 | min | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | Official solver (mplib) | 否 | 规划器 | 8 | **92.6** | 85 | 无参数，不能做 CL |
| 2 | BC，仅初始数据 | 是 | 单头 | 8 | **65.5** | — | 单一策略覆盖 8 任务，无 DAgger |
| 3 | BC + DAgger r1 | 是 | 单头 | 8 | 78.4 | — | 9,025 轨迹 |
| 4 | BC + DAgger r2 | 是 | 单头 | 8 | 83.4 | — | 12,225 轨迹 |
| 5 | BC + DAgger r3 | 是 | 单头 | 8 | 86.4 | — | 15,425 轨迹 |
| 6 | BC + DAgger r4 | 是 | 单头 | 8 | 91.1 | — | 18,625 轨迹 |
| 7 | BC + DAgger r5 | 是 | 单头 | 8 | **92.4** | 83 | 22,625 条轨迹（训练集 121,769 帧），与老师打平 |
| 8 | **BC + DAgger r5 → 56 维（交付）** | 是 | **单头** | **8** | **93.5** | **87** | **8/8 ≥80%** |
| 9 | PPO + ER + MTD（`c8v2_ad`，trunk 冻结） | 是 | multi-head | 8 | 88.7 | 71 | CL 路线最好一对 |
| 10 | PPO + ER + MTD（`c8v5_cons`） | 是 | multi-head | 8 | 92.0 | — | DAgger 的老师 |
| 11 | PPO + ER + MTD（`c8v4_picktrunk`，trunk 漂移 0.2） | 是 | multi-head | 8 | 79.0 | — | trunk 漂移有害 |
| 12 | PPO naive（`c8v7`，`replay_ratio=0`） | 是 | multi-head | 8 | 76.6 | — | 无 ER、无 MTD |
| 13 | PPO + ER + MTD vs naive（同 seed 配对） | 是 | multi-head | 8 | 78.8 vs **80.7** | — | **p ≈ 0.49，ER+MTD 无正贡献** |
| 14 | TD-MPC2 + MPPI（仅 StackCube） | 是 | model-based | 1 | 0–7 | — | 见 §6.5 |

> 第 9–13 行是 **multi-head**：每个任务有自己的读出层，违反"严格单模型"约束，
> 只能作为 CL 方法的对照，不能与第 8 行直接比较。第 8 行才是交付口径。

### 5.3 DAgger 学习曲线

> 轨迹数为"每任务采集条数"的累加值，与 npz 帧数不是同一口径（见 §4.1）。

| 轮次 | 每轮新增（条/任务） | 累计轨迹 | 8 任务均值 |
|---|---|---|---|
| 初始 BC | 831–1,000 | 7,425 | 65.5 |
| r1 | 200 | 9,025 | 78.4 |
| r2 | 400 | 12,225 | 83.4 |
| r3 | 400 | 15,425 | 86.4 |
| r4 | 400 | 18,625 | 91.1 |
| **r5** | 500 | **22,625** | **92.4** |

r4→r5 只 +1.3 点，且 r5 = 老师的 92.0%，说明**继续迭代收益已饱和**；第 6 轮
与权重平均（SWA）都试过，均为负结果。

### 5.4 Episode length

> **未测**。本文所有 SR 都是在 GPU 后端评测的（CPU 上 PegInsertionSide 只有 15%，
> 因为 PhysX CPU 与 GPU 在亚毫米级插孔上不一致）。2026-11-19 补测时实例 GPU
> 掉线（`/dev/nvidia*` 不存在、`torch.cuda.is_available()=False`），未能在交付模型
> 上取得 episode length。
>
> 已有参考（9 任务模型 `sd_9t_v22`，同协议）在评测时**只记录了 SR，未落盘
> episode length**，所以本表不提供"平均步数"这一列；仅数据侧专家步长（13–40 步、
> 见 §4.1）可作为"学生应当达到什么量级"的参考。

---

## 6. 每个 baseline 的具体做法

### 6.1 Official solver（mplib 运动规划）

不训练、无参数。每个任务调用 `mani_skill/examples/motionplanning/panda/solutions/`
下的官方解：内部用 `mplib` 做抓取位姿采样 → IK → 碰撞检测 → 分段轨迹，执行时把
规划出的关节轨迹转成 8 维 delta 动作。用途有两个：生成初始 182,328 帧数据；以及
作为"任务本身可解"的证明。

**PokeCube 没有官方 solver**，用自建规则规划器（推到目标点即可），87%。这解释了
它在所有 baseline 里都是最弱项之一。

### 6.2 单头 BC（behavior cloning）

| 项 | 初版（§3 阶段） | 交付链（`sd_single_d5`） |
|---|---|---|
| 输入 | obs 3 帧堆叠 → 144 维 | obs 5 帧堆叠 → 240 维 |
| 主干 | `Linear→Tanh→Linear→Tanh→Linear`，hidden 256 | 同结构，hidden 768 |
| 输出 | 8 维动作（`tanh` 后乘 `ACTION_LIMIT`） | 同 |
| 损失 | 监督回归（NLL，`nll=True`） | 同 |
| 优化器 | Adam，lr 3e-4，200 epochs，batch 2048 | Adam，lr 1e-4，120 轮 |
| 采样 | **按任务均衡**：`per_batch = BS // n_tasks` | 同 |

**为什么必须按任务均衡**：PullCube 平均 18 步、PullCubeTool 34 步，按帧均匀采样会让
长任务获得近 2 倍梯度，短任务被压制。均衡后每个 batch 里 8 个任务各出 256 条（初版）。

**为什么把多帧堆叠**：单帧输入下策略看不到"运动趋势"，而统一接口里没有任务标签，
堆叠是唯一的隐式时序信息。3 帧 → 5 帧、hidden 256 → 768 是交付链相对初版的两处
改动，与 5 轮 DAgger 叠加后到 92.4%。**本文没有做这两项的单因素消融**，因此不把
增益单独归给其中任一项——这是下一阶段可以补的对照（§9-D）。

**单头的关键约束**：8 个任务共用一个输出头、一套参数，只能靠观测本身的物理差异
区分任务——这正是"无任务标签"的检验方式。

### 6.3 DAgger（5 轮）

```
初始 BC  →  学生在环境里跑  →  老师在学生访问到的每个状态上打标签
        →  新数据并入数据集  →  重训（含旧数据）  →  重复
```

与纯 BC 的差别只有一句：**数据里的状态是学生自己走出来的，不是专家走出来的**。
实现上把"走偏 / 恢复 / 失败"的帧全部保留——`_sd_dagger.py` 的注释写得很直白：
*"a state where the student is already off-track is precisely the state that needs
a label"*。

| 轮次 | 新增轨迹/任务 | 累计轨迹 | 均值 |
|---|---|---|---|
| r1 | 200 | 9,025 | 78.4 |
| r2 | 400 | 12,225 | 83.4 |
| r3 | 400 | 15,425 | 86.4 |
| r4 | 400 | 18,625 | 91.1 |
| r5 | 500 | 22,625 | **92.4** |

**收敛判据**：r5 = 92.4% ≈ 老师 92.0%，说明误差不再来自"学生没学到老师"，而来自
老师本身。第 6 轮、权重平均（SWA）都是负结果，链条在此停止。

### 6.4 PPO + ER + MTD（CL 主实验，multi-head）

**用途**：这是"任务逐个到达、旧数据可能不允许无限留存"的持续学习设定，与 §6.2 的
联合训练是两件不同的事。

| 项 | 值 |
|---|---|
| 算法 | PPO（on-policy） |
| replay | `replay_ratio = 0.5`，旧任务样本以**监督项**进入 loss（不是混进策略梯度） |
| anchor | 当前任务 rollout 的观测 → 当前任务专家，系数 2.0 |
| ER / BC | replay 里的旧**动作**，系数 0.05 |
| MTD | replay 里的旧**观测** → **该样本所属任务的冻结专家**，系数 2.0 |
| 模型 | shared base 256-256 + 每任务私有 adapter（multi-head） |
| 序列 | Peg → Stack → Pull → Pick → LiftPeg → Push → Poke → Tool |

结果：`c8v2_ad`（trunk 冻结）**88.7%**（min 71）、`c8v5_cons`（蒸馏重对齐）**92.0%**、
`c8v4_picktrunk`（trunk 允许漂移 0.2）79.0%、naive（无 ER/MTD）76.6%。

**负结果（必须一起报）**：同 seed 配对下 ER+MTD 78.8 vs naive 80.7，`p ≈ 0.49`——
在 8 任务序列上没有统计显著的正贡献。ER 真正的作用是"防止旧任务坍塌"（Stack 在
5 任务 ER 运行中从 87% 掉到 25%），但 MTD 的额外蒸馏没有换来增益。

### 6.5 TD-MPC2（负结果）

StackCube 单任务，`model_size=5`，2M–4M 环境步，MPPI 规划：

| 配置 | SR |
|---|---|
| policy-only（直接执行 actor） | 0%（reward 10.4 / 63.7） |
| MPPI N=50 | 0%（reward 145.1） |
| MPPI N=30（`best_strong`） | **7%** |
| MPPI N=30（`w3resume`） | 3% |

reward 上去了但成功率为 0——稠密奖励与 success 判据脱钩，这是 model-based 路线在这批
任务上失败的直接原因（详见 `DATASET_AND_METHODS.md` §5）。

---

## 7. 与 ManiSkill 官方 benchmark 的可比性

**不能和官方 leaderboard 直接横向比较。** 四处不同：

| 项 | 官方 benchmark | 本套件 |
|---|---|---|
| 控制模式 | 各任务官方推荐（`pd_ee_delta_pose` / `pd_joint_delta_pos` 混用） | 统一 `pd_joint_delta_pos` |
| horizon | 注册默认值 | 4 个任务覆盖为 100（专家本身要 19–40 步） |
| 观测 | 各任务原生 state / RGB-D | 统一 48 维语义槽位 + 多帧堆叠 |
| 评估 | 每任务 100 回合 | 同（seeds 0–99） |

所以本表的 SR 只在"这套统一接口内部"可比：**同一协议下**比较 solver / BC / DAgger /
CL / TD-MPC2 五种路线。跨论文比较时，必须同时报告 horizon 与控制模式。

**可以对外主张的三件事**：

1. 单一策略头、一套参数、无任务标签，8 个任务 **93.5%（min 87，8/8 ≥80%）**；
2. 与老师策略打平（学生 92.4 vs 老师 92.0），且这条结论有 5 轮 DAgger 曲线支撑；
3. 所有 baseline 在同一评测脚本、同一 seed 集合、同一 GPU 后端下测得。

## 8. 已知限制

| 项 | 现状 | 影响 |
|---|---|---|
| Episode length | **未测**（GPU 掉线，§5.4） | 不能报"平均步数"，只能报 horizon |
| PlaceSphere | 10 类任务里唯一失败的（9 任务口径 0–3%） | 已从本套件移除；根因是输入尺度互斥，见 `CL_10TASK_RESULTS.md` §11.7 |
| StackPyramid / PlugCharger / RollBall | 专家侧不可用（动作分辨率 / 需 IK 安全网 / 专家 0%） | 未进入套件 |
| 标签上界 | 老师 92.0–92.6% | 学生 93.5% 含评测噪声，本质上"与老师打平"，不是超过老师 |
| 观测 | 纯 state（48 维特权信息） | 实机不可直接部署，需要视觉版本 |
| 随机性 | 每任务 100 回合、单 seed 集合 | 单任务 SR 的 95% 置信区间约 ±5 pt，87 vs 92 的差异不显著 |
| PokeCube solver | 自建规则规划器 | 它的 87% 不是官方数字 |
| CL 负结果 | ER+MTD 相对 naive `p ≈ 0.49` | 不能声称 ER+MTD 有效 |

## 9. 下一阶段候选方向（供与导师讨论）

按"投入 / 风险 / 可发表性"排序，每条都附上依据。

**A. 视觉输入版本（state → RGB-D）**
- 依据：当前 48 维全是特权信息（物体真实位姿、关节角），实机拿不到；这是本套件
  离"可用"最远的一环。
- 做法：保持 8 任务与统一接口不变，把 obs 换成 RGB-D + proprio，先跑通 Push/Pull/Pick
  三个容易的，再上 Peg/Stack。
- 风险：视觉版成功率通常比 state 低 10–20 pt；单头跨任务在视觉上更难。
- 预期产出：一份"state vs vision"对照表，正好补上本文缺的那一列。

**B. CL 方法改进（ER+MTD 无正贡献之后）**
- 依据：§6.4 的 `p ≈ 0.49` 是清晰的负结果，说明"监督式 replay 蒸馏"在这个序列上不够。
- 候选：EWC / 参数隔离（LoRA、adapter）与本文"固定参数量"约束的冲突需要先定义清楚
  ——若允许多一份固定大小的 adapter 池，"固定参数量"仍有意义；若不允许，只能走正则路线。
- 依据补充：9 任务实验里 EWC λ=0.03 把 Peg 从 44 拉到 77（+33），说明正则在**混训**
  上有效——值得搬到 CL 序列上验证。

**C. 失败任务的专项**
- StackPyramid：动作分辨率不足（专家 87%→学生 0），可试子步动作 / 更长 horizon。
- PlugCharger：最佳解是"策略接近 + IK 安全网"，这是混合系统，需要单独定义评测口径
  （安全网触发次数也要报）。
- RollBall：专家 0%、0 帧，先解决"任务可解"再谈学习。

**D. 数据效率**
- 依据：全轮 22,625 条轨迹（数据池 732,708 帧、交付训练集 121,769 帧）换 92.4%。
  若只留 r4–r5 的新增数据（约 230,927 帧）能否保住 90%？这直接决定"采数据成本"
  这一项。
- 成本低（只重训，不重采），适合作为下一阶段的第一个小实验。

**E. 精细任务与粗任务的架构分离**
- 依据：PlaceSphere 的失败不是数据或容量，而是"5 mm 精度需要全维标准化，而 8 任务
  依赖 raw 量纲区分任务"（`CL_10TASK_RESULTS.md` §11.6–11.7）。
- 若下一阶段要纳入精细任务，必须在"输入表示"层面做设计（例如把精细量显式算成
  相对差、或加一个固定宽度的双通路），而不是继续加数据。

## 10. 复现

```bash
# 0) 数据：初始 8 任务 npz（182,328 帧）+ 5 轮 DAgger 合并集
#    合并后的训练集：demos/sd_dagger_r5.npz（121,769 帧，48 维，tid 0–7）

# 1) 单头 BC + DAgger（交付链）
SD_DATA=demos/sd_dagger_r5.npz SD_STACK=5 SD_HIDDEN=768 SD_NORM=0 \
  SD_EPOCHS=120 SD_LR=1e-4 python _sd_train.py        # -> ckpts/sd_single_d5.pt

# 2) 48 -> 56 维零填充（等价拓宽，新增列权重置零）
python _ps_up56.py                                  # -> ckpts/sd_single_d5_p56.pt

# 3) 8 任务评测（100 回合/任务，GPU）
bash _ev8.sh ckpts/sd_single_d5_p56.pt 100

# 4) solver 基线（官方 motionplanning 解）
python mani_skill/examples/motionplanning/panda/run.py -e PegInsertionSide-v1 -n 100

# 5) CL 主实验（multi-head）
#    see _cl_*.sh；关键参数：replay_ratio=0.5, anchor=2.0, bc=0.05, mtd=2.0
```

> 注意 `python` 指远端 conda 环境 `/root/miniconda3/envs/ms3/bin/python`；并行评测需
> `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。

## 附录 A：交付物清单

| 文件 | 内容 | 位置 |
|---|---|---|
| `ckpts/sd_single_d5_p56.pt` | **交付模型**（8 任务单头，280 维输入，hidden 768） | 服务器 `/root/cl/ckpts/`，10.7 MB，未入仓库 |
| `ckpts/sd_single_d5.pt` | 同链 48 维版（240 维输入），92.4% | 服务器 `/root/cl/ckpts/`，10.5 MB，未入仓库 |
| `demos/sd_dagger_r5.npz` | 合并训练集（121,769 帧，48 维，tid 0–7） | 服务器 `/root/cl/demos/`，18.0 MB，未入仓库 |
| `_shapes.sh` | 数据集体检（帧数 / tid 分布 / 文件清单），本文所有帧数字由此得出 | 仓库 `scripts/_shapes.sh` |
| `_sd_train.py` | BC / DAgger 训练脚本（支持 EWC、warm start、按任务均衡采样） | 仓库 `scripts/_sd_train.py` |
| `_sd_eval_gpu.py` + `_ev8.sh` | 8 任务评测（GPU 后端、STACK 历史缓冲） | 仓库 `scripts/_sd_eval_gpu.py`、`scripts/_ev8.sh` |
| `DATASET_AND_METHODS.md` | 数据集与方法全量说明（含负结果） | 仓库 `docs/DATASET_AND_METHODS.md` |
| `CL_10TASK_RESULTS.md` | CL 与 9 任务严格单头的完整实验记录 | 仓库 `docs/CL_10TASK_RESULTS.md` |

> 仓库 <https://github.com/qingyunyanran/QY>，`main` 分支（2026-11-26 起；提交号以仓库首页为准，附录有改动时会变）。三个二进制文件（两个 `.pt`、一个 `.npz`，合计约 39 MB）不放进 git：`*.pt` 已被 `.gitignore` 排除，数据集体积也不适合进仓库历史；需要时直接从服务器 `/root/cl/` 取，或按需转 Release 附件。

---

*本文所有 SR 均为 100 回合/任务、seeds 0–99、GPU 后端；除特别标注外不做多次重复，
单任务 SR 的统计噪声约 ±5 pt。*
