# Progress Report — ManiSkill3 Continual Learning

**Date:** 2026-08-23  
**Student:** Lijie Zu  
**Advisor Meeting:** 2026-07-20

---

## 1. Advisor Requirements (2026-07-20 Meeting)

1. **Focus on Experience Replay (ER)** — Do not test EWC (also poor in ContinualWorld). Getting ER working establishes the full CL framework.
2. **Upload code to GitHub** — Keep it updated; advisor will review ER implementation.
3. **Analyze single-task RL results** — Investigate why tasks like StackCube initially showed 0% success; render evaluation videos.
4. **Work methodology** — Take ownership; do not switch methods at every obstacle; communicate progress regularly.

---

## 2. Single-Task Baselines — Complete

All five ManiSkill3 tasks have trained or validated checkpoints:

| Task | Success Rate | Method | Notes |
|------|-------------|--------|-------|
| PickCube-v1 | **100%** | SB3 PPO, 5M steps | Converged cleanly |
| PushCube-v1 | **100%** | SB3 PPO, 3M steps | Converged cleanly |
| StackCube-v1 | **75%** | Official ckpt (Tanh fix) | Key bug: activation function mismatch (ReLU vs Tanh) vs official architecture; after aligning to all-Tanh hidden layers, 20-episode eval = 75% |
| PokeCube-v1 | **92%** | Official ckpt, `pd_joint_delta_pos` | Used as starting point for CL transfer experiments |
| PullCube-v1 | **100%** | Official ckpt, `pd_joint_delta_pos` | Validated directly |

**StackCube diagnosis:** The initial 0% success was caused by using ReLU activations instead of the official PPO architecture's Tanh activations. After switching all hidden layers to Tanh and loading the official pretrained weights, evaluation reached 75% over 20 episodes. This is documented in `cl/StackCube_PPO_失败分析报告.md`.

Checkpoints are located at `~/.maniskill/demos/<TaskName>-v1/rl/`.

---

## 3. CL Framework (continual_learning_v2)

The CL framework in `cl/` implements:

- **Sequential Fine-Tuning (SeqFT)** — Baseline: train on tasks sequentially without any anti-forgetting mechanism.
- **Experience Replay (ER)** — Primary method: maintain a replay buffer of transitions from previous tasks; mix replay data with current task data during PPO updates.
- **DISTR** — Diffusion-based trajectory replay (ported from prior work).

Core modules:

| File | Purpose |
|------|---------|
| `cl_methods.py` | SeqFT and ER method implementations (12.7 KB) |
| `cl_comparison.py` | Multi-method comparison runner (35.5 KB) |
| `env_wrapper.py` | Multi-task environment wrapper, observation unification, zero-padding (20.8 KB) |
| `bc_agent.py` | Behavior cloning agent for expert demonstration processing |
| `config.py` | Centralized configuration |
| `main.py` | Entry point for CL experiments |
| `evaluate.py` | Per-task and multi-task evaluation utilities |

Current framework supports the 5-task sequence with unified observation space (zero-padding to max dim = 54 for PokeCube).

---

## 4. Route A: Parameter Isolation Experiments (Exploratory)

Starting from the official PokeCube-v1 checkpoint (92% SR), we explored parameter isolation and adapter methods to transfer to PickCube-v1 while preserving PokeCube performance. These experiments serve as **ablation studies** demonstrating why ER is the preferred approach.

### 4.1 Observation Space Mismatch

PokeCube-v1 has 54-dimensional observations while PickCube-v1 has 42 dimensions. The standard approach zero-pads PickCube observations to 54 dimensions. A dedicated diagnosis (`diag_pickcube_54dim.py`) confirmed:

- Training PickCube from scratch with 54-dim zero-padded input for 1M steps: SR=0%, avg_R=22.72
- The reward was increasing (reaching behavior learned), but grasp behavior never emerged
- `approx_kl=0.33–0.40`, `clip_fraction=0.74+` indicated aggressive, unstable updates

### 4.2 Full Results

| Version | Strategy | PokeCube SR | PickCube SR | PickCube avg_R | Steps | Key Finding |
|---------|----------|-------------|-------------|----------------|-------|-------------|
| v5 | Full fine-tune | 92% | 0% | — | 5.5M | Policy collapse; catastrophic forgetting begins |
| v6 | Low LR full FT | 95% | 0% | 1.15 | 3M | Completely locked into poking motion |
| v7 | Fresh head, backbone unfrozen | 0% (forgot) | 0% | 24.5 | 1.5M | Features overwritten by PickCube; target_kl=0.02 caused early stop |
| v8 | Frozen backbone + task-specific heads | 95% | 0% | 8.64 | 3M | Frozen PokeCube features too rigid for grasping |
| v9 | Frozen backbone + LoRA rank=8 | 95% | 0% | 20.31 | 3M | LoRA adapters improve over pure freeze (R: 8.64→20.31) but cannot achieve grasp |
| v10 | LoRA rank=16, ent=0.02, lr=5e-4 | 90% | 0% | 19.81 | 3M | Doubling rank yields no improvement; bottleneck is not adapter capacity |
| v11 | Per-task input encoder + LoRA rank=16 | 75–90% | **best 5%** | 24.14 | 3M | Separate encoders solve zero-padding issue; first non-zero PickCube SR in the series, but grasp remains unstable |

### 4.3 Technical Details

**v8 (Frozen Backbone):** Froze all `policy_net` layers from the official PokeCube checkpoint. Only trained a fresh `action_net` head for PickCube. PokeCube performance perfectly preserved (95%), but PickCube avg_R dropped to 8.64 — the frozen feature space encodes poking-specific motion patterns that cannot support grasping.

**v9/v10 (LoRA Adapters):** Inspired by TAIL (ICLR 2024), injected low-rank adaptation matrices into policy network layers:
```
output = original(x) + (alpha/rank) * (x @ A @ B)
```
Original weights frozen; only LoRA parameters (A, B) and task-specific heads trained. PokeCube eval disables LoRA and swaps back the saved head (zero forgetting). LoRA improved PickCube reward (8.64→20.31) but rank 8 vs 16 made no difference, indicating the limitation is structural, not capacity-based.

**v11 (Per-Task Input Encoder):** Identified zero-padding as the fundamental bottleneck — the frozen first layer was trained on meaningful 54-dim PokeCube input, while PickCube's 42 meaningful dims + 12 zero dims create an input distribution mismatch. Solution:
- PokeCube: frozen `Linear(54→256)` with official weights
- PickCube: fresh trainable `Linear(42→256)` seeing only the 42 valid dims
- Shared hidden layers with LoRA(rank=16, alpha=32) + task-specific action heads
- Value network fully trainable

This produced the first PickCube success (best SR=5%) in the entire series, confirming the diagnosis. However, success was rare and unstable (only occurred around 800K steps, not reproduced at 1.8M or 2.7M).

### 4.4 Conclusions from Route A

1. **Zero-padding creates a significant input distribution mismatch** that affects transfer even with adapter methods. Per-task encoders help but add architectural complexity.
2. **Poking and grasping require fundamentally different motor patterns.** PokeCube's hidden layers encode a poking motion primitive (approach + push), while PickCube requires approach + close fingers + lift — a different temporal gesture sequence. Low-rank corrections cannot fully decouple these.
3. **Parameter isolation alone is insufficient** for RL tasks with conflicting motor patterns. The adapter can learn reaching (avg_R≈24) but cannot consistently learn grasp.
4. **These negative results motivate the ER approach:** rather than trying to isolate parameters, ER allows the shared network to learn both tasks by interleaving replay data, which is more natural for RL and avoids the representation conflict.

---

## 5. Next Steps

Per advisor direction, the immediate next step is **returning to the ER mainline**:

1. **Complete ER implementation** in `cl_methods.py` — ensure replay buffer properly stores and samples transitions from prior tasks during PPO training.
2. **Run ER on the task sequence** PickCube → PushCube → StackCube → PokeCube → PullCube (or a subset for initial validation).
3. **Compare ER vs SeqFT** using forward transfer (FWT), backward transfer (BWT), and average success rate.
4. **Render evaluation videos** for single-task baselines and CL results.
5. **Keep GitHub updated** with each ER iteration.

---

## 6. File Index

| Category | Directory | Key Files |
|----------|-----------|-----------|
| Single-task PPO | `baselines/` | `official_ppo.py`, `train_*_ppo.py`, `eval_*.py` |
| CL framework (ER) | `cl/` | `cl_methods.py`, `cl_comparison.py`, `env_wrapper.py`, `main.py` |
| Route A experiments | `experiments/route_a/` | `train_cl_route_a_v5.py` through `v11.py`, results JSON |
| Documentation | `docs/` | This report |
| Project root | — | `README.md` |
