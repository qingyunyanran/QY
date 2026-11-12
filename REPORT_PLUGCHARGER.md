# PlugCharger-v1 无 solver 化攻关：解析抓取 + IK 精确跟随

日期：2026-11-12　机器：RTX 4080 SUPER（单卡）　环境：ManiSkill3 / `pd_joint_delta_pos` / 统一 66 维观测

本报告是本轮"**把两个 anchor 从运动规划 solver 换成自主策略**"攻关的阶段性汇总，
供汇报使用。所有数字均为实测值，未做外推。

---

## 1. 总体进度（10 任务 CL）

| 项目 | 状态 |
|---|---|
| 8 任务统一 PPO（`c8v5_cons_best`），独立 100 回合/任务 | **8/8 达标，均值 92.0%** |
| 无教师消融（`c8v6_noanchor`） | **8/8 达标，均值 91.4%**（Poke 80 / Tool 97） |
| StackPyramid-v1 | solver 84%；学习式分段 head 最好 50% |
| **PlugCharger-v1** | solver 81%；**本轮学习式：抓取段已 100%，插入段攻关中** |
| 10 任务混合口径（8 学习 + 2 solver） | 90.6%，全任务 ≥80% |

结论不变：8 个操作任务可以合并进**一条共享底座 + 8 头**的统一策略；
两个装配型长程任务（Pyramid / PlugCharger）是当前唯一没跨过去的坎。

---

## 2. 本轮结论（PlugCharger）

**核心成果：抓取阶段的"能否夹住"已经不是问题——解析式抓取 4/4 全中，tcp 到抓取位姿误差 5 mm。**
卡了整整两天的"策略永远夹不住 / 差 2 mm 插不进"的因果链被彻底拆开，目前唯一未完成的是插入段的收敛。

| 阶段 | 原方案（BC/PPO 闭环） | 现方案（解析抓取 + IK） |
|---|---|---|
| 接近 + 抓取 | `dg` 全程 155 mm 不变（**从未夹住**） | `\|grasp\|` 达 **0.005 m（4/4）**，t≈80 步内完成 |
| IK 求解 | — | `ik_fail = 0`（此前 814~943） |
| 插入 | — | 已改为几何直线推进，**验证中** |
| eval SR | 19%（best, N=100） | 0%（N=4，抓取已通，插入未收敛） |

---

## 3. 八个量化根因（全部有实测支撑）

### 3.1 观测四元数切片错误（最致命）
66 维观测里 quat 在 `[42:46]`，早期代码把 `o[39:46]` 整段 7 维喂给旋转矩阵构造函数。
`_R()` 只吃 4 个分量 ⇒ 读到的姿态是**错的**，抓取方向从第一步就偏。
修正后 `p_grasp` 落在桌面高度 `z=0.018 m`（此前飘到 `z=0.29 m`）。

### 3.2 IK 必须用 base frame，不是 world frame
官方 `setup_planner()` 调 `planner.set_base_pose(np.hstack([base_pose.p, base_pose.q]))`，
`_transform_pose_for_planning()` 原样返回，`plan_qpos_to_pose(..., wrt_world=True)`。
**实测**：world frame 传入 ⇒ `IK Failed` **100%**（800/800）；base frame ⇒ 成功。
转换：`B = inv(robot.pose)`，`_p = (B @ [_t,1])[:3]`，`_qb = mat2quat(B[:3,:3] @ _R(q))`。

### 3.3 夹爪安装旋转必须显式建模（缺失项）
直接用 `charger` 的姿态当抓取姿态，与 reset 时的真实抓取姿态**差 180°**，
reset 处 IK 必然失败（`Cannot find valid solution`）。
从 warm 数据 586 次闭合事件标定出安装旋转：

```
REL_QUAT (wxyz) = [-0.000205, 0.991622, 0.000165, 0.129171]   # ≈ 绕 x 180° + 绕 z 15°
q_grasp = normalize(qmult(q_charger, REL_QUAT))
```

偏差中位数 **0.534°**，p90 0.916° —— 这就是"姿态看着对、IK 就是解不出来"的真凶。
相对抓取偏移 `REL = [-0.01828, 0.00016, 0.00557] m`。

### 3.4 `n_init_qpos=1` + `sol[0]` 会造成机械臂瞬移
单采样只取第一个解，会返回肘部翻转等**远分支**：位姿满足但工具瞬移 0.28 m
（实测 z 从 0.042 跳到 0.324）。
改为多采样 + 取**离当前关节最近的解** `min|s - qnow[:7]|` 后，`ik_fail` 从 943 降到 **0**。

### 3.5 IK 的 `threshold` 过松 = 原地不动
`threshold=8e-3`（8 mm）时，求解器发现当前构型已在容差内，直接返回当前关节
⇒ 机械臂"指令在动、实际不动"，插入永远推进不了。
改 `threshold=1e-3` 后插入才有推进：`|dp|` 0.0007 → 0.0013，`|a|` 0.03 → 0.06。

### 3.6 控制器只跟踪约 40% 的指令（尺度根因）
`pd_joint_delta_pos` 稳态跟踪增益实测 **0.396**（`a=0.5` → `dq1=0.01981`）。
demo 的 `act` 是**归一化增量指令**（scale 0.1），不是"实际效果"：

```
med|a*0.1 − Δq| = 3.771e-3  vs  med|Δq| = 2.509e-3   比值 2.499
```

即 `a*0.1 = 2.5·Δq`。这解释了历史所有 BC 部署"动作偏小、走不到位"。
关节增量段用 `GAINC=2.5` 补偿；**笛卡尔 dp 段不能放大**（放大 2.5 倍后 π 输出从 0.8 mm/步
涨到 12~24 mm/步，直接把插头从夹爪里撞出去）。

### 3.7 恒定 25 mm 的 x 残差
base frame 下小步定标：目标 x `0.0112`，实得 `-0.0138`，err **0.0251 m**（y/z 精确）。
这个量级同时解释了历史指标里 `min|ax|` 恒卡在 **15~16 mm** ——原来那不是"差一点"，
而是同一个系统性偏置。插入阶段必须用闭环几何反馈，不能靠开环位姿。

### 3.8 π_cart 插入段的致命方向错误
π_cart2/3 预测的是笛卡尔位移（标签 `dp*100`，单位 cm/step），离线 `|pred|` 中位数 2.6 mm 正常。
但闭环实测：**它让 tcp 在 z=0.022 平面上水平滑动，而插座在 z=0.100**，
y 方向还差 6 cm ⇒ **几何上永远对不上**。不是训练不足，是策略学到的映射与"插入"这一目标不一致。
⇒ 插入段改用**几何直线推进**（朝 `receptacle` 位姿，锁腕，4 mm/步），见 §4。

### 3.9 插入阶段夹爪必须强制闭合
让 π 驱动夹爪时它把 grip 放松到 `-0.14`（几乎张开），充电器会滑脱。
stage 4 固定 `grip=-1.0`。

---

## 4. 当前管线（`_probe_rule.py`，4 段状态机）

| stage | 动作 | 判据 |
|---|---|---|
| 0 | 从正上方接近（`pre = p_grasp + [0,0,0.08]`），步长 8 mm，`GAINC=2.5` | `\|pre\|<8 mm` |
| 1 | 位置 + 姿态**同步**插值到 `p_grasp / q_grasp`，步长 5 mm | `\|tcp-p_grasp\|<8 mm` 且 `ang<0.10` |
| 2 | 精修到 `p_grasp`，步长 4 mm | `\|tcp-p_grasp\|<5 mm` |
| 3 | 闭合夹爪，保持 8 步 | `hold>=8` |
| 4 | **几何直线插入**：朝 `o[32:35]`（receptacle）推进 4 mm/步，锁腕，`grip=-1.0` | env 判定 success |

关键点：stage 1 早期做成"原地旋转"（`tgt=tcp`）会产生**正反馈漂移**——
腕部一动 tcp 就位移，目标跟着走，62 步累积 25 cm（z 0.036→0.427）。
位置目标必须**冻结**在 `p_grasp`。

---

## 5. 历史评测汇总（N=100，除注明）

| 模型 / 策略 | SR | grasp | depth med | min\|ax\| |
|---|---|---|---|---|
| `plgmix3`（PPO） | **19%** | 97/100 | 0.0235 | 0.0154 |
| `plgbc_all5` | 8% | 92/100 | 0.0204 | 0.0153 |
| `plgbc_all6` | 7% | 96/100 | 0.0204 | 0.0159 |
| all6 + GAIN 4.0 | 0% | 94/100 | 0.0202 | 0.0039 |
| 混合 + 轴投影 | 0% | 96/100 | 0.0207 | 0.0009 |
| 解析抓取 + IK（本轮，N=4） | 0% | **4/4** | — | — |

参照：ManiSkill 官方榜 PlugCharger SR 0–4%（官方承认 ground-truth planner 在其基准上也失败）。
官方 solver 在本机 81%（`joint_vel_limits=2.0`、倾角 15°，243/300）。

**小样本严重高估，必须 N=100**：16 集报 12% → 实际 8%；20 集报 40% → 实际 19%。

---

## 6. 下一步

1. 验证 stage 4 几何直线插入的收敛（`PROBE_INS_STEP` 4 mm/步），先 N=4 看 `|tcp-receptacle|` 是否单调下降。
2. 通过后 `PROBE_N=100` 全量评测，目标 SR ≥80%。
3. 同路线迁移 StackPyramid（21 段几何门控已有 50% 基线，插入式解析段可复用）。
4. 更新 `docs/CL_10TASK_RESULTS.md`、`docs/PLUGCHARGER_DIAGNOSIS.md`。

---

## 7. 相关文件

- `plugcharger/_probe_rule.py` — 主管线（4 段状态机 + 解析抓取 + 几何插入）
- `plugcharger/_probe_ik2.py` — base-frame 小步定标（25 mm 残差证据）
- `plugcharger/_probe_rot.py` — `REL_QUAT` 安装旋转标定（586 次闭合事件）
- `plugcharger/_probe_ik.py` — IK 解灌回 env 的落点校验
- `plugcharger/_probe_fk.py` — 末端 link 一致性校验（排除刚体 offset 假设）
- 数据：`demos/plg_warm2.npz`（82840 帧 / 1000 集 / 586 次闭合）、`demos/plg_all6.npz`（129808 / 1061）
- 权重：`ckpts/plg_cart2.pt`（π，`mae_ins 0.089 mm`）、`ckpts/plg_cart3.pt`（all6 训练中）

## 8. 环境坑（复现须知）

- 探针必须 `sim_backend="cpu", render_backend="none"`，否则 `Failed to find a supported physical device "cuda:0"`。
- 后台训练用 `setsid`，否则 `rs.py` 会话结束会把训练一起杀掉（显存 32760 MiB → 4 MiB）。
- `transforms3d.quaternions` 无 `quat_slerp` / `quat2axang`：自实现 `_slerp`，角度用 `quat2axangle`。
- mplib `Planner` **无** `get_fk` / `p`；可用 `link_name_2_idx` / `move_group_link_id` / `plan_qpos_to_pose` / `user_link_names`。
