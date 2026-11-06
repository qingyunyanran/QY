# 8 任务持续学习：在 6/6 达标之上再并入两个复杂任务（并说明 RollBall 为何被剔除）

日期：2026-11-05　机器：RTX 4080 SUPER（单卡）　框架：`cl/ppo_er_unified.py`

序列：`PegInsertionSide → StackCube → PullCube → PickCube → LiftPegUpright → PushCube → PokeCube → PullCubeTool`
观测：统一 48 维（qpos/qvel/tcp_pose/slot1/slot2/tail），8 个任务完全同构，无 one-hot、无 padding；动作统一 8 维（7 关节 + 夹爪，`pd_joint_delta_pos`）
训练：ER（replay 0.5、每任务 buffer 200k）+ MTD 蒸馏（coef 2.0）+ 专家 anchor（coef 2.0）+ 128 envs × 64 steps = 8192/update
方法定位：**expert-initialized CL（教师辅助 / 蒸馏式）**——先按任务训单任务 PPO 专家，再并入一条共享底座的序列；
教师**只在那个任务自己的训练窗口内生效**（`anchor` 分支按任务开关），不是全程在线。见 §2.7。
**所有报数均为独立进程的 100 回合评估**（训练内 20 回合只用于看趋势，见 §2.5，它会把 Pick 高估 20–35pp）

---

## 1. 结论

| 方案 | Peg | Stack | Pull | Pick | Lift | Push | Poke | Tool | 均值 | 达标 |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 任务 D 组（原报告，独立 100 回合复评）| 95 | 88 | 100 | 64 | 87 | 100 | – | – | 89.0 | 5/6 |
| c8v1：追加任务 + `trunk_lr_mult 0.5` | 0 | 0 | 5 | 0 | 15 | 60 | – | – | 41.3 | 0/8 |
| c8v2：追加任务 + `trunk_lr_mult 0.0`（底座冻结）| 93 | 83 | 99 | 64 | 81 | 100 | 87 | 98 | 88.1 | 7/8 |
| c8v3：c8v2 + Pick 真实 PG 2M（底座仍冻结）| 91 | 87 | 100 | 67 | 81 | 100 | 87 | 98 | 88.9 | 7/8 |
| c8v4：c8v3 + Pick 段放开底座 ×0.2 | 91 | 69 | 100 | **98** | 49 | 92 | 56 | 77 | 79.0 | 4/8 |
| **c8v5：c8v4 + 7 任务蒸馏重对齐（底座再冻结）** | **84** | **89** | **100** | **96** | **84** | **100** | **87** | **96** | **92.0** | **8/8** |
| c8v6：教师可省性——Poke / Tool **不用教师**，各 4M 真实 PG（底座冻结）| 87 | 81 | 100 | 95 | 91 | 100 | **80** | **97** | **91.4** | **8/8** |

- **核心结论 1（扩容可行，但底座必须冻结）**：6 任务 ckpt 可以直接续训到 8 任务，前提是 `--trunk-lr-mult 0.0`。
  换成 0.5 会在 1 次更新内把 5 个旧任务打成 0（c8v1）。
- **核心结论 2（Pick 的瓶颈是冻结特征，不是 head）**：底座一放开（c8v4，×0.2），Pick 从 64% 直接跳到 98%；
  证明蒸馏头/head-PG 补不回来的东西，是"256 维 CL 特征不是 Pick 专家自己的特征"。
- **核心结论 3（底座一动，旧头必须重对齐）**：同样是 c8v4，Stack 69 / LiftPeg 49 / Poke 56 / Tool 77 ——
  每个旧任务只有 1 次更新来适应新特征，来不及。补一轮**蒸馏重对齐**（底座再冻结，每任务 ≤200k）后全部回到 84–100%，
  **8/8 达标，均值 92.0%**（比 6 任务单阶段的 89.0% 还高）。
- **核心结论 4（零遗忘的逐位证明）**：c8v2 的 `actor_mean[0:4]` 与全部旧 adapter 和 6 任务 ckpt **逐位相同**
  （`diff_ckpt.py`：28 个张量 identical，只有 critic / actor_logstd（Δ0.012）/ Peg 私有网（Δ0.005，它自己那 1 次更新）在动）。
  同一份权重在 6 任务 ckpt 上复评 Pick 也是 64.0% —— **Pick 当年的 64% 是历史遗留，不是被新任务带坏的**。
- **核心结论 5（候选任务筛选）**：三个候选里 RollBall 在 30M 预算下只到 44%（训练期曾 70%），**不并入**；
  序列因此是 8 个任务而不是 9 个。Poke 需两轮补训（16M + 8M）才从 73% 到 85%。
- **核心结论 6（方法定位：expert-initialized CL，教师可按任务省略）**：先按任务训单任务专家、再并入共享底座序列，
  是**标准范式**（Policy Distillation / Actor-Mimic 一脉），它解决的是"单网络共存"而不是"无教师从零学习"。
  anchor 的真实作用是**把专家策略投影到新的共享特征上**（专家 head 直接拷 `max|diff| = 1.585`，不能用），
  代价是蒸馏 head 没有策略改进能力（Pick：蒸馏 64% → 真 RL 98%）+ 成本随任务数线性。
  对照消融（c8v6，Poke / Tool 不用教师）**8/8 仍全达标（均值 91.4%，Poke 80 / Tool 97）**：Tool 在序列内 0.61M 即早停，
  比单训的 20.5M 少 **33×** —— 教师**可省**，省掉的是 N 次单任务预训练的成本。

---

## 2. 证据链

### 2.1 候选任务必须与 48 维观测 / 8 维动作 / 同一 horizon 同构
扫描脚本对 ManiSkill3 全任务列表逐项检查：观测能否映射进 7 维 slot 槽位、动作维度是否正好 7 关节 + 夹爪、
`max_episode_steps` 与 `gamma` 是否需要单独设置。结论：

- 可用（结构完全同构）：`PokeCube-v1`、`PullCubeTool-v1`、`RollBall-v1`
- 需特殊处理：`PegInsertionSide-v1`（43 维原生观测、必须 `gamma=0.99`，已在 6 任务中作为私有底座异常任务处理）
- 不可用：`PickSingle*` / `PlaceSphere` 等需要 `base_pose` 之外的自由度或不同控制模式的任务

### 2.2 三个候选的专家起点与补训预算

用同一套统一观测先训出单任务专家（anchor），再测其 100 回合 SR：

| 任务 | 专家初测 | 补训预算 | 补训后 | 备注 |
|---|---|---|---|---|
| PokeCube-v1 | 73.0% | 16M（9.4M 早停） | 78.0% → 再补 8M 后 **85.0%** | 需要两轮 |
| PullCubeTool-v1 | 0.0% | 20.5M 早停 | **99.0%** | 前 20M 全在刷 reward，最后 10M 才找到真解 |
| RollBall-v1 | 4.0% | 30M（跑满） | **44.0%** | 训练期曾 70%，不达标，剔除 |

### 2.3 追加任务时，共享底座必须冻结
c8v1 与 c8v2 的唯一区别就是 `--trunk-lr-mult`：

| 配置 | Peg | Stack | Pull | Pick | Lift | Push | Poke | Tool | 均值 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0 | 0 | 5 | 0 | 15 | 60 | – | – | 41.3 |
| **0.0** | 93 | 83 | 99 | 64 | 81 | 100 | 87 | 98 | 88.1 |

机理：旧任务在本轮只跑 1 次更新（8192 步就进位到下一个任务），底座一动，旧 adapter 就没有机会重新对齐；
MTD 的损失量级只有 2e-3 到 4e-3，压不住底座的漂移。**冻结底座 = 新任务只在新增维度上学习。**

### 2.4 逐位 diff：扩容没有改动任何旧任务
`diff_ckpt.py` 对比 `c6v4_refit_best`（6 头）与 `c8v2_ad`（8 头）：

```
identical: 28   changed: 17   only in A: 0   only in B: 8
changed:  actor_logstd 0.0125 | critic.* 0.45 左右 | private_nets.0.* 0.005 量级
only in B: actor_heads_extra.5.* , actor_heads_extra.6.*   （Poke / Tool 的新头）
```

`actor_mean[0:4]`（共享底座）与 `actor_heads_extra[0:4]`（Peg/Stack/Pull/Pick/Lift/Push 的 adapter）**一个字节没动**。
这是"零遗忘"最硬的证据：8 任务 ckpt 在旧任务上的行为，等于 6 任务 ckpt。

### 2.5 Pick 的 64%：不是遗忘，是蒸馏头的天花板（并修正一处旧报数）
- 8 任务 ckpt 与 6 任务 ckpt 的 Pick 策略逐位相同，两者独立 100 回合都是 64%，因此与扩容无关。
- 同一 ckpt 再测 20 / 20 / 100 回合：70 / 60 / 65%（`pick_eval.py`）——真实值 64±4%。
- **旧报告里的 Pick 89% 复现不出来**：训练内 20 回合评估对 Pick 给出 85–100%（`per_task_results` 里 100/100/90），
  换成独立进程就掉到 64–70%。任务越吃精度，20 回合的小样本越会过估；本报告一律以独立 100 回合为准。
- 归因实验：让 Pick 的头做**真实 PG**而不是蒸馏（把它从 `--anchor-task` 里拿掉，anchor 分支是按当前任务生效的），
  底座仍冻结，2M 步 → 64% → **67%**。说明瓶颈在**冻结底座的 256 维特征**，不在 head 容量。

### 2.6 两阶段收尾：放开底座修 Pick，再重对齐救其余七个
**第一阶段（c8v4）**：在 c8v3 上重跑，只把 Pick 段的 `trunk_lr_mult` 提到 0.2，其余 7 个任务各 1 次更新、靠 MTD 保护。

- Pick **64 → 98%**（训练内 95–100%，独立 100 回合 98%）——瓶颈判断被证实：冻结的 256 维特征才是天花板。
- 但代价立刻显现：Stack 83→69、LiftPeg 81→49、Poke 87→56、Tool 98→77，均值 79.0%。
  原因和 c8v1 同构：**底座动了，而旧 adapter 只有 1 次更新去适应新特征**。
  0.2 只是把损伤从"全灭（41.3）"降到"部分（79.0）"，并没有消除它。

**第二阶段（c8v5）**：把底座重新冻结（`trunk_lr_mult 0.0`），给那七个任务各 ≤200k 步做**蒸馏重对齐**
（它们仍在 `--anchor-task` 里，`pg_loss` 被置零，head 只拟合"当前底座特征 → 专家动作"），Pick 保持 8192 步不动。

| 任务 | c8v4（底座漂移后）| c8v5（重对齐后）| 重对齐预算（早停点）|
|---|---|---|---|
| Peg | 91 | 84 | 163,840 |
| Stack | 69 | 89 | 200,000 |
| Pull | 100 | 100 | 163,840 |
| Pick | 98 | 96 | 8,192（不动）|
| LiftPeg | 49 | 84 | 163,840 |
| Push | 92 | 100 | 163,840 |
| Poke | 56 | 87 | 200,000 |
| Tool | 77 | 96 | 200,000 |
| **均值** | 79.0 | **92.0** | 约 12 分钟 |

结论：**"先放开底座学难任务，再冻结底座把旧任务的头贴回去"是可复用的两步法**，比全程冻结多花 12 分钟，
换来 Pick +32pp、均值 +3.1pp（相对 c8v2 的 88.1%），且 8/8 达标。
这也说明 6 任务报告里"Pick 需要重训"的判断是对的——只是当年重训的是 head，真正该动的是 trunk。

### 2.7 方法定位与教师可省性（c8v6 对照）

**本方法的定位是 expert-initialized CL（教师辅助 / 蒸馏式持续学习）**，与 Policy Distillation、Actor-Mimic 一类工作同属一脉：
先给每个任务单独训一个 PPO 专家（相当于"会做这个任务的示范者"），再把它们并入一条共享底座的序列。
这解决的是**部署形态**问题——8 个任务共存于一个网络、一份 ckpt、免切换，而不是"在没有教师的情况下从零长出新技能"。
所以下面四条 claim 必须分开写，各自有独立证据：

| claim | 证据 | 是否成立 |
|---|---|---|
| ① 单网络共存 8 个任务 | c8v5：8/8 ≥80，均值 92.0 | ✅ |
| ② 扩容不破坏旧任务 | c8v2 逐位 identical；c8v1 反例（0.5 → 41.3）| ✅ |
| ③ 教师可按任务省略 | c8v6：Poke 80 / Tool 97，8/8 仍达标 | ✅ |
| ④ "CL 算法本身抗遗忘" | 无——抗遗忘来自冻结底座 + 重对齐 | ❌ 不声称 |

#### 教师不是装饰：专家 head 不能直接拷

`smoke_c9.py`（真实 6 头 ckpt）实测：把专家 head 直接放到 CL 底座特征上跑，`max|diff| = 1.585`。
原因是专家是一个在**它自己那套底座**上训出来的 7 层扁平网络，第 4 层之后的 adapter 吃的是它自己的 256 维特征分布；
共享底座（已看过 6 个任务）的 256 维特征分布与它不同。所以 append **只提供初始化**，
`anchor` 在线蒸馏的真实作用是**把专家策略投影到新的共享特征上**。这也是 Pick 卡在 64% 的同一机制：
蒸馏头是在旧特征上拟合专家，特征一错配就封顶。

#### 教师的代价（两条要主动承认）

1. **蒸馏 head 没有策略改进能力**：anchor 存在时 `pg_loss *= 0`，该 head 全程只做"拟合专家动作"，
   天花板 = 专家水平 × 特征迁移损失。精度类任务必须走真 RL（Pick 64 → 98 是最直接的证明）。
2. **成本随任务数线性**：N 个任务要 N 次单任务预训练。任务数涨到几十个时这一项不可忽略 —— 这正是"教师可省性"值得测的动机。

#### 教师可省性（c8v6）：这两个任务不需要教师

设置：把 Poke / Tool 从 `--anchor-task` 删掉（anchor 分支按任务生效，`if anchor_net is not None` 在逐任务循环内），
它们的 `pg_loss` 恢复非零、head 走真实策略梯度；其余（初始化 c8v5、序列顺序、底座冻结 `trunk_lr_mult 0.0`、每任务 4M）与 c8v2 逐项对齐。

| 任务 | c8v2（有教师，4M 蒸馏）| c8v6（**无教师**，4M 真实 PG）| 单任务专家实际消耗 | 序列内早停点 |
|---|---|---|---|---|
| PokeCube | 87 | **80** | 9.42M（100 回合仅 73%）+ 补 3.07M = **12.5M** → 85% | 跑满 4M（训练内 35–95% 抖动）|
| PullCubeTool | 98 | **97** | **20.48M** 早停 → 99% | 614,400 步（训练内 90%）|
| 8 任务均值 | 88.1 | **91.4** | – | – |

- **结论**：Poke 80 / Tool 97，8/8 仍达标（均值 91.4）—— 两个任务在**没有教师**的情况下也过了 80% 线。
- **样本效率是这次对照最硬的数字**：Tool 从零 20.5M → 序列内 0.61M（**33×**）；Poke 从零 12.5M → 4M（**3×**）。
  这部分增益来自**前向迁移**（forward transfer）：底座已经在前 6 个任务上形成了可复用的操作先验，
  新任务不必从随机权重重新摸索。→ 在"把任务学会"这件事上，序列本身可以替代教师。
- **Poke 是唯一明显吃教师的**：87 vs 80（7pp，略超噪声），且训练内从 35 抖到 95%。教师在这里的价值是**降低方差**，
  而不只是抬分数；Tool 则几乎不需要教师（614k 早停 → 终评 97%）。
- 顺带一个反直觉的点：Poke 专家在 9.42M 时训练内 95%、独立 100 回合只有 73%，再补 3.07M 才到 85% ——
  **训练内评估对 Poke 同样不可信**，与 §2.5 的 Pick 是同一个坑。
- 其余 6 个任务在 c8v6 里各跑 8192 步（仍是 anchor 蒸馏、底座冻结），相对 c8v5 浮动 ±8pp
  （Peg 84→87、Stack 89→81、Lift 84→91），但全部 ≥80 —— 单次更新造成的蒸馏漂移在可接受范围内。

**工程建议**：教师按任务难度分配 —— 需要新技能或高精度接触的任务（Pick 一类）保留教师或至少走真 RL；
能被共享底座快速迁移的任务（Tool 一类）可以省掉那次单任务预训练。这样总成本不必随任务数线性膨胀。

---

## 3. 代码改动（`cl/ppo_er_unified.py`）

- `expand_state_dict_load(agent, sd, verbose, source)`：把短序列 ckpt 载入长序列。
  `strict=False` 加载；`unexpected` 非空即报错；校验 head 索引是 `range(n_heads-1)` 的**连续前缀**
  （头是位置索引，非前缀布局会被拒绝，而不是静默把 head 绑给错任务）；返回 `(kept, fresh, missing)`。
- `load_task_expert(agent, task_idx, actor_sd, include_base=True)`：把单任务专家（7 层扁平 `Sequential`）
  拆成 `body`（层 0–3）+ `adapt`（层 4+）写进 `actor_heads_extra[task_idx-1]`；
  `include_base=True` 只覆盖共享底座层 0–3，**不动** `actor_mean[4:]`（那是任务 0 的 adapter）。
- 新 CLI：`--init-allow-expand`（继续长序列）、`--pretrain-adapter-only`（只灌 adapter 不碰底座）。
- 修掉的真实 bug：`task_idx=0` 时 Python 负索引会把专家静默写进**最后一个 head** → 改为 `IndexError` 拒绝，并校验上界。
- 单测：`test_expand.py` 7 passed（6 头 → 9 头扩容）、`test_expert_load.py` 5 passed、`test_adapter.py` 仍通过。
- `smoke_c9.py`（真实 6 头 ckpt）：底座逐位不变、trunk 特征逐位相同、head 在同输入空间精确复现专家；
  但专家 head 跑在 CL 底座特征上 `max|diff| = 1.585` → **append 只提供良好初始化，仍需微调**。

### 踩过的脚本级坑（都是会静默出错的那种）

1. `--init-ckpt` 才是"整份状态字典续训"，`--pretrain-ckpt` 是 `TASK=PATH` 的单任务热启动，两者混用会直接报错或灌错头。
2. 续训必须同时给 `--multi-head`（否则 Agent 只建 1 个头）和 `--init-allow-expand`（否则 head 数不匹配）。
3. **c8v2 的真实顺序是 Peg 第一**（私有网键 `0` 就是证据）。头是位置索引，顺序错一位等于所有任务用错头。
4. PowerShell 下内联 `python -c "..."` / heredoc 必然解析失败 → 一律写 `.sh`/`.py` 文件再上传执行。
5. 轮询脚本 `grep -q "[3/3]"` 被当字符类，立即误命中 → 改固定串匹配，并自检 `waited=100s` 才放行。

---

## 4. 复现

```bash
# 1) 三个新任务的专家（anchor）
bash scripts/train_new3.sh          # Poke 16M / RollBall 30M / Tool 30M
bash scripts/train_poke_extra.sh    # Poke 再补 8M

# 2) 8 任务续训（底座冻结，约 25 分钟含两次长任务）
bash scripts/c8v2_ad.sh

# 3) Pick 定向补训（真实 PG，底座冻结，约 18 分钟）
bash scripts/c8v3_pickpg.sh

# 4) Pick 段放开底座 ×0.2（约 20 分钟）
bash scripts/c8v4_picktrunk.sh

# 5) 七任务蒸馏重对齐，收口 8/8（约 12 分钟）
bash scripts/c8v5_cons.sh

# 5b) 对照消融：Poke / Tool 不用教师，靠序列内真实 PG（约 25 分钟）
bash scripts/c8v6_noanchor.sh

# 6) 100 回合独立复评（--tasks 顺序必须与训练序列完全一致）
python eval_ckpt_5tasks.py \
  ckpts/ppo_er_unified/ppo_er_unified_task7_PullCubeTool-v1_c8v5_cons.pt \
  --tasks PegInsertionSide-v1,StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1,PokeCube-v1,PullCubeTool-v1 \
  --multi-head --private-base PegInsertionSide-v1 --eps 100

# 7) 单测 / 诊断
python test_expand.py && python test_expert_load.py && python test_adapter.py
python diff_ckpt.py          # 两个 ckpt 的逐位对比
python pick_eval.py CKPT --eps-list 20,100   # 单任务 SR
```

结果：`results/ppo_er_unified/ppo_er_unified_results_{c8v2_ad,c8v3_pickpg,c8v4_picktrunk,c8v5_cons,c8v6_noanchor}.json`
日志：`logs/{c8v1_ad,c8v2_ad,c8v3_pickpg,c8v4_picktrunk,c8v5_cons,c8v6_noanchor}.log` 与同名 `_eval100.log`

**已归档的权重与终评日志见 [`artifacts/`](artifacts/README.md)**：`c8v5_cons.pt`（8/8，均值 92.0）与
`c8v6_noanchor.pt`（教师可省性对照，91.4），含 SHA256 与一键复评命令。

---

## 5. 局限与下一步

1. **Pick 已达标，但靠的是两阶段**：64% 的天花板必须靠"放开底座"才破，而放开底座又会伤旧任务，
   所以要补一轮重对齐。更优雅的做法是把第二阶段的蒸馏重对齐做成每个难任务之后的**自动**步骤
   （见下一条），而不是手工编排两轮。
2. **RollBall 未达标**（44%）被剔除，序列是 8 个任务而不是 9 个。若要用它，需要 success shaping（当前奖励信号太稀疏）
   或更长预算；30M 预算下它没有证明能到 80%。
3. **"达标"的口径要说清**：本报告是"多任务共存 / 抗遗忘验证"——
   Peg 用的是整份私有网络 + 梯度冻结，等于一个受保护的静态专家，不是"完全共享底座"。
   完全共享**单一策略**（无 per-task head 与 adapter）的天花板仍是 56–62%（6 任务报告的 C 组）。
   本文的多头 + adapter 方案（8/8，91.4%）解决的是"共存"，不是"一个网络一套参数通吃 8 个任务"。
4. **重对齐轮应该固化成 `--consolidate`**：目前的 c8v5 是手工脚本（冻结底座 + 每任务 ≤200k 蒸馏 + `--stop-sr 85` 早停）。
   建议在每个任务结束后自动触发，阈值用"评估 ≥85% 连续两次"。
5. **评估协议要统一**：训练内 20 回合对精度类任务（Pick）会高估 20–35pp（曾经报 100%，独立 100 回合只有 64%），
   所以 `--stop-sr` 早停如果只看 20 回合会有系统性乐观偏差；建议训练内评估也提到 100 回合。
