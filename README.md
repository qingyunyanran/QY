# ManiSkill3 Continual Learning

Continual Learning experiments on [ManiSkill3](https://www.maniskill.ai/) manipulation tasks, focusing on **Experience Replay (ER)** as the primary method per advisor direction, with **parameter isolation / adapter methods** (Route A) as ablation studies.

## Task Suite

Five ManiSkill3 tasks using the Panda robot with `pd_joint_delta_pos` controller:

| Task | Obs Dim | Action Dim | Description |
|------|---------|------------|-------------|
| PickCube-v1 | 42 | 8 | Pick up a cube and move it to target |
| PushCube-v1 | 42 | 8 | Push cube to target position |
| StackCube-v1 | 42 | 8 | Stack one cube on top of another |
| PokeCube-v1 | 54 | 8 | Poke cube with tool (different obs space) |
| PullCube-v1 | 42 | 8 | Pull cube toward robot |

## Single-Task Baselines (PPO)

| Task | Success Rate | Training | Checkpoint |
|------|-------------|----------|------------|
| PickCube-v1 | **100%** | SB3 PPO, 5M steps | `~/.maniskill/demos/PickCube-v1/rl/` |
| PushCube-v1 | **100%** | SB3 PPO, 3M steps | `~/.maniskill/demos/PushCube-v1/rl/` |
| StackCube-v1 | **75%** | Official ckpt + Tanh fix (20 eval eps) | `~/.maniskill/demos/StackCube-v1/rl/` |
| PokeCube-v1 | **92%** | Official ckpt (pd_joint_delta_pos) | `~/.maniskill/demos/PokeCube-v1/rl/` |
| PullCube-v1 | **100%** | Official ckpt (pd_joint_delta_pos) | `~/.maniskill/demos/PullCube-v1/rl/` |

## Project Structure

```
maniskill-cl/
├── baselines/                    # Single-task PPO training & evaluation
│   ├── official_ppo.py           # Official PPO training script
│   ├── train_pickcube_ppo_v2.py  # PickCube SB3 PPO
│   ├── train_pushcube_ppo_v2.py  # PushCube SB3 PPO
│   ├── train_stackcube_ppo_v3.py # StackCube PPO
│   └── eval_*.py                 # Evaluation scripts
│
├── cl/                           # Continual Learning framework (ER focus)
│   ├── cl_methods.py             # SeqFT, ER, DISTR method implementations
│   ├── cl_comparison.py          # Multi-method comparison runner
│   ├── env_wrapper.py            # Multi-task env wrapper & observation unification
│   ├── bc_agent.py               # Behavior cloning agent for expert demos
│   ├── config.py                 # Configuration
│   ├── main.py                   # Entry point
│   ├── evaluate.py               # Evaluation utilities
│   ├── scripted_experts.py       # Scripted expert policies
│   ├── train_bc_experts.py       # BC expert training
│   ├── results/                  # CL experiment results
│   └── README.md                 # CL framework documentation
│
├── experiments/
│   └── route_a/                  # Parameter isolation / adapter experiments
│       ├── train_cl_route_a_v5.py    # Full fine-tune (catastrophic forgetting)
│       ├── train_cl_route_a_v7.py    # Fresh head without freezing
│       ├── train_cl_route_a_v8.py    # Frozen backbone + task-specific heads
│       ├── train_cl_route_a_v9.py    # LoRA rank=8 adapters
│       ├── train_cl_route_a_v10.py   # LoRA rank=16 adapters
│       ├── train_cl_route_a_v11.py   # Per-task input encoder + LoRA
│       ├── diag_pickcube_54dim.py    # Zero-padding diagnosis
│       └── results/                  # JSON result files
│
└── docs/
    └── PROGRESS_REPORT.md        # Detailed progress report
```

## Environment Setup

```powershell
# Conda environment (Python 3.10)
conda activate diffcl10

# Key dependencies
pip install mani_skill stable-baselines3 torch numpy gymnasium

# Download demonstrations (for BC / ER buffer initialization)
python -m mani_skill.utils.download_demo PickCube-v1
python -m mani_skill.utils.download_demo PushCube-v1
python -m mani_skill.utils.download_demo StackCube-v1
python -m mani_skill.utils.download_demo PokeCube-v1
python -m mani_skill.utils.download_demo PullCube-v1
```

**Note:** Training runs on CPU (AMD RX 7600, no CUDA). For faster training, use a GPU machine with the official ManiSkill3 PPO implementation.

## Route A: Parameter Isolation Experiments

Starting from the official PokeCube-v1 checkpoint (92% SR), we transfer to PickCube-v1 while preserving PokeCube performance. The observation dimension mismatch (PokeCube 54-dim vs PickCube 42-dim) is handled via zero-padding, except v11 which uses per-task input encoders.

| Version | Strategy | PokeCube SR | PickCube SR | PickCube avg_R | Key Finding |
|---------|----------|-------------|-------------|----------------|-------------|
| v5 | Full fine-tune | 92% | 0% | — | Policy collapse at 5.5M steps |
| v6 | Low LR full FT | 95% | 0% | 1.15 | Locked into poking behavior |
| v7 | Fresh head, no freeze | 0% (forgot) | 0% | 24.5 | Features overwritten, early stop |
| v8 | Frozen backbone + head | 95% | 0% | 8.64 | Features too rigid |
| v9 | LoRA rank=8 | 95% | 0% | 20.31 | LoRA helps but insufficient |
| v10 | LoRA rank=16 | 90% | 0% | 19.81 | Higher rank no improvement |
| v11 | Per-task encoder + LoRA | 75–90% | **best 5%** | 24.14 | First non-zero SR, but unstable |

### Key Conclusions

1. **Zero-padding observation mismatch** significantly hinders transfer — v11's per-task encoder was the first approach to achieve any PickCube success.
2. **Poking and grasping require fundamentally different movement patterns.** Low-rank adapters (LoRA) cannot decouple these from a frozen poking-centric feature space.
3. **Parameter isolation alone is insufficient** for RL tasks with conflicting motor patterns, motivating the Experience Replay (ER) approach as the primary CL method.

## Current Focus: Experience Replay (ER)

Per advisor direction (2026-07-20 meeting), the primary focus is implementing and evaluating **Experience Replay** on ManiSkill3:

- Maintain a replay buffer of transitions from previous tasks
- Mix replay samples with current task data during PPO updates
- Evaluate forward transfer and backward transfer across the 5-task sequence
- Compare against Sequential Fine-Tuning (SeqFT) baseline

## References

- [ManiSkill3](https://www.maniskill.ai/) — Benchmark for generalizable manipulation
- [TAIL (ICLR 2024)](https://arxiv.org/abs/2310.05905) — Task-Aware Incremental Learning
- [TAILS](https://arxiv.org/abs/2608.16345) — Task-Aware Incremental Learning for Robotics
