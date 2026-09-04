# ManiSkill3 持续学习

基于 [ManiSkill3](https://www.maniskill.ai/) 机械臂操作任务的持续学习（Continual Learning, CL）实验。

按导师制定的路线推进：**统一观测接口 → 单任务 RL baseline → 持续学习**。目前第二阶段已完成：在统一 42 维观测接口下，4 个任务使用**一行未改的官方 PPO 算法**全部达到 **100% 成功率**。

## 统一 42 维观测接口

ManiSkill3 的 Panda 任务原始状态维度各不相同（Push/Pull 35 维、Pick 42 维、LiftPegUpright 32 维）。我们没有引入可学习的映射层（早期实验表明这会损失约 30 个百分点的单任务成功率），而是按**物理语义**把观测对齐到固定槽位：

```
[ 0:18]  本体感知        Panda qpos(9) + qvel(9)          —— 所有任务完全一致
[18:25]  tcp_pose        末端执行器 xyz + 四元数（7 维）
[25:32]  物体槽位 1      主操作物位姿（7 维）
[32:39]  物体槽位 2      第二物体 / 工具（预留槽位，补零）
[39:42]  goal_pos        目标点 xyz（3 维；无目标点的任务补零）
```

设计决策：
- **删除相对向量**（`tcp_to_obj_pos`、`obj_to_goal_pos`、`is_grasped`）：它们只是绝对位姿的线性差/函数，不含新信息。PushCube/PullCube 原生就不含任何相对向量，两个任务都训练到 100% SR。
- **保留全部绝对位姿**（tcp / 物体 / 目标）：场景完全由这些绝对量定位，不做任何改动。
- 无对应物体的槽位补零，但槽位语义在所有任务间严格一致。

实现：`cl/unified_obs.py`（`build_unified_obs_batch`）。

## 任务集（4 个任务）

全部使用 Panda 机器人、`pd_joint_delta_pos` 控制器、状态观测。

| 任务 | 原始观测维度 | 回合步数 | 技能 |
|------|------------|---------|------|
| PushCube-v1 | 35 | 50 | 把立方体推到目标点 |
| PullCube-v1 | 35 | 50 | 把立方体拉向机器人 |
| PickCube-v1 | 42 | 50 | 抓起立方体并移到目标点 |
| LiftPegUpright-v1 | 32 | 50 | 把平躺的销钉扶直立起 |

### 为什么是 4 个任务（CPU 算力边界）

训练在 **CPU（16 个向量化环境，AMD RX 7600，无 CUDA）**上进行。两阶段 / 多刚体接触类任务在此预算下无法收敛：

| 被排除的任务 | 尝试步数 | 最佳 SR | 说明 |
|-------------|---------|---------|------|
| StackCube-v1 | 10M | 0% | 双立方体精确叠放 |
| PokeCube-v1 | 8.8M（GPU） | ~10% | 工具使用，两阶段 |
| PlaceSphere-v1 | 2M | 0% | 抓取并放入容器 |
| PullCubeTool-v1 | 10M | 12.5%（8 次 eval 蒙对 1 次） | L 形工具钩取 + 拉拽 |

参照：GTP-FA（arXiv:2606.03385）中这些任务需要 2× RTX 4090、50M 步、2048 个并行环境才能收敛。失败原因是算力预算，不是观测接口问题——奖励曲线持续上升，但完整接触链条始终无法稳定学会。

## 单任务 Baseline（官方 PPO，算法零改动）

脚本：`cl/ppo_official_unified.py` —— fork 自官方 `examples/baselines/ppo/ppo.py`，**只插入了 42 维观测转换层**，算法、超参、网络结构全部保持原样。CPU 适配：16 个环境、512 步 rollout（batch 8192）。

| 任务 | 训练步数 | 最佳 SR |
|------|---------|---------|
| PushCube-v1 | 2M | **100%** |
| PullCube-v1 | 2M | **100%** |
| PickCube-v1 | 5.12M（10M 步训练内收敛） | **100%** |
| LiftPegUpright-v1 | 2M | **100%** |

Checkpoint：`cl/ckpts/ppo_official_unified/{env}_seed1_{latest,final}.pt`，结果见 `{env}_seed1_results.json`。

### 复现

```powershell
conda activate diffcl10
cd cl
python ppo_official_unified.py --env-id PushCube-v1       --total-timesteps 2000000
python ppo_official_unified.py --env-id PullCube-v1       --total-timesteps 2000000
python ppo_official_unified.py --env-id PickCube-v1       --total-timesteps 10000000
python ppo_official_unified.py --env-id LiftPegUpright-v1 --total-timesteps 2000000
```

## 下一阶段：持续学习

单任务 baseline 成立后，CL 阶段将对比：
- **SeqFT** —— 顺序微调（灾难性遗忘基线）
- **ER（经验回放）** —— 维护旧任务转移数据的 replay buffer，在 PPO 更新时混入回放样本（导师指定的主方法）
- 参数隔离 / adapter 消融实验（早期 Route A 实验，见 `experiments/route_a/`）

CL 入口：`cl/ppo_er_unified.py`（同样基于 42 维统一接口）。

## 项目结构

```
maniskill-cl/
├── cl/
│   ├── unified_obs.py            # 42 维语义槽观测接口
│   ├── ppo_official_unified.py   # 官方 PPO + 统一观测（单任务 baseline）
│   ├── ppo_er_unified.py         # PPO + 经验回放（CL 阶段）
│   ├── cl_methods.py             # SeqFT / ER / DISTR 方法实现
│   ├── env_wrapper.py            # 多任务环境包装器
│   ├── config.py
│   ├── main.py
│   ├── evaluate.py
│   └── ckpts/                    # checkpoint 与结果 json
├── baselines/                    # 早期单任务 PPO 脚本
├── experiments/route_a/          # 参数隔离 / LoRA 消融实验
└── docs/PROGRESS_REPORT.md
```

## 环境

```powershell
conda activate diffcl10   # Python 3.10，ManiSkill3，torch（CPU），gymnasium
```

## 参考

- [ManiSkill3](https://www.maniskill.ai/) —— 通用操作基准
- [TAIL (ICLR 2024)](https://arxiv.org/abs/2310.05905) —— 任务感知增量学习
- GTP-FA（arXiv:2606.03385）—— GPU 规模 PPO 在 ManiSkill 任务集上的参考结果
