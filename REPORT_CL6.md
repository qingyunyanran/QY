# 6 任务持续学习：从"共享 actor 过不去 Peg"到 6/6 达标

日期：2026-11-04 ~ 11-05　机器：RTX 4080 SUPER（单卡）　框架：`cl/ppo_er_unified.py`

序列：`PegInsertionSide → StackCube → PullCube → PickCube → LiftPegUpright → PushCube`
观测：统一的 48 维（qpos/qvel/tcp_pose/slot1/slot2/tail），6 个任务完全同构，无 one-hot
训练：ER（replay 0.5、每任务 buffer 200k）+ MTD 蒸馏（coef 2.0）+ 专家 anchor（coef 2.0）
预算：Peg 600k，其余各 2.5M；128 envs × 64 steps = 8192/update，16 minibatches
**所有报数均为独立进程的 100 回合评测**（训练内 20 回合仅用于看趋势，波动 ±10pp）

---

## 1. 结论

| 方案 | Peg | Stack | Pull | Pick | Lift | Push | 均值 | 达标 |
|---|---|---|---|---|---|---|---|---|
| A 共享 actor（单头） | 0% | 78% | 100% | 89% | 85% | 100% | 75.3% | 5/6 |
| B 每任务一个输出层（trunk ×0.1） | 49% | 27% | 100% | 87% | 81% | 100% | 74.0% | 4/6 |
| C B + 冻结 trunk 重训头 | 56% | 58% | 100% | 88% | 79% | 100% | 80.2% | 2/6 |
| **D 共享底座 + 每任务 adapter + Peg 私有底座 + 定向重训** | **91%** | **85%** | **100%** | **89%** | **85%** | **100%** | **91.7%** | **6/6** |

D 的拆解：全量 `c6v4_ad` 结束时 Peg 62% / Stack 84% / Pick 75%（均值 84.5%，4/6）；
只补一轮定向重训（Peg 300k + Pick 600k，共享底座冻结）后即上表的 6/6。

---

## 2. 失败链与每一步的证据

### 2.1 共享 actor 的硬冲突（A → B）
Peg 只有 43 维原生观测、必须 `gamma=0.99`（默认 0.8 的折扣视野约 5 步，插入收益在 50 步之后）。
热启动 92% 专家后，单头 6 任务序列里 Peg 在**下一个任务的 20 万步内从 95% 掉到 0%**，
且不是超参问题：`lr 3e-4→1e-4` + `target_kl 0.1→0.02` 仍只有 0–20%。
原因是共享输出层只能收敛到一个"折中策略"，而插入成功是一个毫米级吸引域，折中解落在域外。

### 2.2 只给每任务一个输出层还不够（B）
多头（共享 3 层 trunk + 每任务最后一层）让 Peg 在自己阶段到 95–100%，
但进入下一个任务阶段就先掉到 30/20%，再被 MTD 蒸馏拉回 45–80%——**泄漏在 trunk**。
降低 trunk 学习率（×0.1）能把 Peg 稳在 70–90%，但 Stack 在 655k 仍是 0%（trunk×1.0 时 598k 就到 80%）。

trunk 学习率扫描（Peg→Stack 探针，20 回合）：

| trunk_lr_mult | Peg 在 Stack 阶段 | Stack @600-700k |
|---|---|---|
| 1.0 | 100 → 30 → 20 → 80 → 45 | 80%（best） |
| 0.3 | 75 → 80 → 75 → 70 → 60 | 0% |
| 0.1 | 75 → 80 → 75 → 70 → 65 → 90 | 0% |

### 2.3 冻结 trunk 只能止损（C）
在 B 的终态上冻结 trunk、只重训 Peg 与 Stack 的头各 300k：Peg 35→56%、Stack 27→58%。
说明**特征已经被改掉了**，头的容量补不回来。

### 2.4 共享底座 + 每任务 adapter 把两侧的需求分开（D）
把共享范围从"3 层"缩到"2 层"，剩下 1 个隐层 + 输出层归各任务私有（adapter）：

| 配置 | Peg | Stack |
|---|---|---|
| adapter，trunk ×0.1 | 85–95%（全程不掉） | 0%（700k 仍是 0） |
| adapter，trunk ×0.5 | 60–80% | 75%（600k） |

两边的要求是**相反**的：Stack 要底座会动，Peg 要底座不动。
所以把 Peg 定为"异常任务"，给它一份完整私有网络，其余 5 个任务共享底座 + 各自 adapter：

- 私有网络 = 整个 actor（3 层 + 输出），热启动时**逐位复原** 75M 专家（单测校验 `atol=1e-6`）
- 任务结束后把私有网络的 `requires_grad` 关掉（否则仅靠 MTD 蒸馏也会把 Peg 从 85% 磨到 62%）
- 共享底座对 5 个简单任务保持完整塑性（`trunk_lr_mult=1.0`）

### 2.5 定向重训补齐最后两项
`c6v4_ad` 终态 Peg 62%（MTD 磨掉的）、Pick 75%（adapter 在漂移后的底座上没对齐）。
以该 ckpt 为起点再跑一轮：Peg 300k（私有网解冻、重新对齐）、Pick 600k，共享底座冻结（lr 0），
其余任务各 1 次更新。9 分钟就回到 100 回合下的 6/6。

---

## 3. 代码改动

`cl/ppo_er_unified.py`：

- `Agent(n_heads=N)`：`actor_mean[0:4]` 为共享底座，`actor_mean[4:7]` 为任务 0 的 adapter，
  `actor_heads_extra[k-1]` 为任务 k 的 adapter（`Linear(hidden,hidden) + Tanh + Linear(hidden,act)`）
- `Agent(private_base_ids=...)`：异常任务用 `private_nets[str(tid)]` 整份网络（`ModuleDict`）
- `head_mean(x, tid)`：单 int 或逐行 task id 张量，统一分发（replay/MTD 一个 minibatch 混多任务）
- `trunk_parameters()` + 参数分组优化器：`--trunk-lr-mult` 只作用于共享底座
- `--multi-head`、`--private-base`、`--trunk-lr-mult`、`--lr`、`--target-kl` 等 CLI
- 热启动仍只拷 actor（policy-only）：私有任务整份加载；共享任务拆成底座 + 该任务 adapter

`cl/eval_ckpt_5tasks.py`：新增 `--multi-head` / `--private-base`，`--tasks` 顺序必须与训练序列一致（头是按位置索引的）。
`cl/test_adapter.py`：单头兼容、adapter 逐位复原、私有底座隔离、逐行 task id 分发。

---

## 4. 复现

```bash
# 全量 6 任务（约 2h10m）
bash scripts/c6v4_ad.sh

# 定向重训（约 11 分钟）
bash scripts/c6v4_refit.sh

# 100 回合独立复评
python eval_ckpt_5tasks.py \
  ckpts/ppo_er_unified/ppo_er_unified_task5_PushCube-v1_c6v4_refit.pt \
  --tasks PegInsertionSide-v1,StackCube-v1,PullCube-v1,PickCube-v1,LiftPegUpright-v1,PushCube-v1 \
  --multi-head --private-base PegInsertionSide-v1 --eps 100

# 架构单测
python test_adapter.py
```

结果 JSON：`results/ppo_er_unified/ppo_er_unified_results_c6v4_ad.json`、`..._c6v4_refit.json`
日志：`logs/c6v4_ad.log`、`logs/c6v4_refit.log`、`logs/c6v4_eval100.log`、`logs/c6v4_refit_eval100.log`

---

## 5. 局限与下一步

1. **共享度**：Peg 目前用的是私有网络（架构级隔离），不是"完全共享底座 + 每任务头"。
   若要求严格共享，当前天花板是 56–62%（见 C 组）。真正能两者兼顾的办法是梯度正交/投影（OGD、GEM）
   或 HAT 式的硬注意力门控，成本明显更高。
2. **Pick 仍需重训**：说明 adapter 在底座漂移后没有自动跟上，MTD 的蒸馏权重（2.0）可能偏小。
3. **Peg 的 gamma**：本轮 Peg 依赖热启动 + 蒸馏，未跑 0.99 的在线微调；若要"从零学会 Peg"，
   仍需按官方 75M 预算 + `gamma=0.99` 单独训练。
4. 巩固轮目前的实现靠"冻结底座 + 分任务短重训"，可以固化成 `--consolidate` 参数，
   在每个任务结束后自动跑 50k 的头重对齐。
