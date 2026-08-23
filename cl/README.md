# ManiSkill Continual Learning Framework

持续学习框架在 ManiSkill 平台上的实现，对比 SeqFT/ER/EWC 方法在机器人操作任务上的抗遗忘能力。

## 项目结构

```
continual_learning_v2/
├── cl_comparison.py              # 主 CL 对比实验脚本（SeqFT/ER/EWC + DAgger）
├── train_bc_experts.py           # BC 专家训练脚本
├── scripted_experts.py           # 脚本化专家实现
├── train_pushcube_ppo_v2.py      # PushCube PPO v2 训练脚本
├── train_pickcube_ppo_v2.py      # PickCube PPO v2 训练脚本
├── verify_pickcube_ppo_v2.py     # PickCube PPO v2 验证脚本
├── render_all_videos.py          # 渲染所有任务的专家视频
├── analyze_training_logs.py      # 分析 TensorBoard 训练日志
│
├── expert_models/                # 训练好的专家模型
│   ├── PushCube-v1/
│   │   ├── PushCube-v1_sb3_ppo.zip      # PPO 模型（100% 成功率）
│   │   ├── PushCube-v1_bc_expert.pt     # BC 模型
│   │   ├── vec_normalize.pkl            # 观测归一化参数
│   │   └── tb_logs/                     # TensorBoard 日志
│   ├── PickCube-v1/
│   │   ├── PickCube-v1_sb3_ppo.zip      # PPO 模型（训练中）
│   │   ├── PickCube-v1_bc_expert.pt     # BC 模型（100% 成功率）
│   │   └── tb_logs/
│   ├── StackCube-v1/
│   │   ├── StackCube-v1_bc_expert.pt    # BC 模型（85% 成功率）
│   │   └── tb_logs/
│   └── PegInsertionSide-v1/
│       └── PegInsertionSide-v1_bc_expert.pt  # BC 模型（79% 成功率）
│
└── videos/                       # 渲染的专家策略视频（生成后）
```

## 任务说明

| 任务 | 描述 | 单任务成功率 | RL训练状态 |
|------|------|------------|-----------|
| PushCube-v1 | 推动立方体到目标位置 | 100% (PPO) | ✅ 完成 |
| PickCube-v1 | 抓取立方体到目标位置 | 100% (BC) | ⚠️ PPO验证0% |
| StackCube-v1 | 堆叠两个立方体 | 85% (BC) | ❌ RL失败 |
| PegInsertionSide-v1 | 插入销钉 | 79% (MP replay) | ❌ RL失败 |

## 持续学习方法

### 已实现
- **SeqFT**：顺序微调（朴素方法）
- **ER (Experience Replay)**：经验回放，维护 replay buffer
- **EWC (Elastic Weight Consolidation)**：弹性权重巩固
- **DAgger**：在线模仿学习聚合

### 当前配置
```python
MAX_OBS_DIM = 48          # 最大观测维度
TASK_ID_DIM = 3           # 任务 one-hot 编码维度
ACTION_DIM = 7            # 动作维度
HIDDEN_DIM = 512          # 隐藏层维度
CL_EPOCHS = 200           # CL 训练轮数
CL_BATCH_SIZE = 256       # 批量大小
CL_LR = 1e-3              # 学习率
ER_BUFFER_SIZE = 5000     # ER replay buffer 大小
EWC_LAMBDA = 100.0        # EWC 正则化强度
n_demos per task = 200    # 每个任务的演示数量
```

## 运行实验

### 1. 渲染专家视频（分析任务表现）
```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
cd E:\munichi\continual_learning_v2
& "E:\My_programs\anaconda\envs\diffcl10\python.exe" render_all_videos.py
```

### 2. 分析训练日志（诊断 RL 失败原因）
```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
cd E:\munichi\continual_learning_v2
& "E:\My_programs\anaconda\envs\diffcl10\python.exe" analyze_training_logs.py
```

### 3. 运行 CL 对比实验
```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
cd E:\munichi\continual_learning_v2
# 完整实验（20 episodes evaluation）
& "E:\My_programs\anaconda\envs\diffcl10\python.exe" cl_comparison.py --eval-eps 20

# 快速测试（10 episodes evaluation）
& "E:\My_programs\anaconda\envs\diffcl10\python.exe" cl_comparison.py --eval-eps 10 --quick
```

## 最新实验结果

### CL 对比实验（PPO PushCube + 脚本化 PickCube/StackCube）
| 策略 | PushCube-v1 | PickCube-v1 | StackCube-v1 | Average | Forget |
|------|-------------|-------------|--------------|---------|--------|
| SeqFT | 0.0% | 0.0% | 0.0% | 0.0% | 57.5% |
| ER | 55.0% | 5.0% | 20.0% | 26.7% | 22.5% |
| EWC | 0.0% | 0.0% | 5.0% | 1.7% | 65.0% |

**分析**：
- SeqFT 完全失败：所有任务 0%，说明模型严重灾难性遗忘
- ER 表现最好：平均 26.7%，遗忘率 22.5%
- EWC 效果差：平均仅 1.7%

## 待解决问题

1. **PickCube PPO v2 验证 0%**：训练指标正常但策略完全失败
2. **StackCube/PegInsertionSide RL 训练失败**：成功率始终 0%
3. **ER 方法改进**：导师建议只测 ER 并改通

## 环境要求

- Python 3.10
- PyTorch (CPU version)
- ManiSkill
- Stable-Baselines3
- Gymnasium
- imageio (视频渲染)

### 安装
```bash
conda create -n diffcl10 python=3.10
conda activate diffcl10
pip install mani_skill stable_baselines3 gymnasium torch imageio
```

## 参考资源

- [ManiSkill 官方文档](https://maniskill.readthedocs.io/)
- [ContinualWorld 论文](https://arxiv.org/abs/2305.15164)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)

## 更新日志

- 2026-07-20: 创建项目结构，完成 CL 框架实现
- 2026-07-20: PushCube PPO v2 训练完成（5M steps）
- 2026-07-20: PickCube PPO v2 训练完成（5M steps，验证中）
- 2026-07-20: 完整 CL 对比实验完成
