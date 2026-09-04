# ManiSkill3 Continual Learning

**English** | [中文](README_zh.md)

Continual Learning (CL) experiments on [ManiSkill3](https://www.maniskill.ai/) manipulation tasks.

Current stage (per advisor plan): **unified observation interface → single-task RL baselines → continual learning**. Stage 2 is now complete: all 4 tasks reach **100% success rate** under the unified 42-dim observation with the *unmodified* official PPO algorithm.

## Unified 42-dim Observation Interface

The ManiSkill3 Panda tasks ship with different raw state dimensions (Push/Pull 35, Pick 42, LiftPegUpright 32). Rather than learning a mapping layer (which cost ~30 pp of single-task SR in earlier experiments), observations are aligned **by physical meaning** into fixed semantic slots:

```
[ 0:18]  agent proprioception   Panda qpos(9) + qvel(9)        — identical across tasks
[18:25]  tcp_pose               end-effector xyz + quaternion (7)
[25:32]  object SLOT 1          primary manipulated object pose (7)
[32:39]  object SLOT 2          second object / tool (reserved; zeros)
[39:42]  goal_pos               goal xyz (3; zeros if task has no goal site)
```

Design decisions:
- **Relative vectors removed** (`tcp_to_obj_pos`, `obj_to_goal_pos`, `is_grasped`): they are linear differences / functions of absolute poses — zero new information. PushCube/PullCube contain no relative vectors natively and both train to 100% SR.
- **All absolute poses kept** (tcp / object / goal): the scene is localized exclusively by these.
- Slots with no corresponding object are zero-padded; slot semantics stay identical across tasks.

Implementation: `cl/unified_obs.py` (`build_unified_obs_batch`).

## Task Suite (4 tasks)

All tasks use the Panda robot, `pd_joint_delta_pos` controller, state observations.

| Task | Raw obs dim | Episode steps | Skill |
|------|------------|---------------|-------|
| PushCube-v1 | 35 | 50 | Push cube to target |
| PullCube-v1 | 35 | 50 | Pull cube toward robot |
| PickCube-v1 | 42 | 50 | Pick up cube and move to target |
| LiftPegUpright-v1 | 32 | 50 | Reorient a lying peg upright |

### Why 4 tasks (CPU compute boundary)

Training runs on **CPU (16 vectorized envs, AMD RX 7600, no CUDA)**. Two-stage / multi-body contact tasks do not converge under this budget:

| Excluded task | Steps tried | Best SR | Note |
|---------------|------------|---------|------|
| StackCube-v1 | 10M | 0% | two-cube precise alignment |
| PokeCube-v1 | 8.8M (GPU) | ~10% | tool-use, two-stage |
| PlaceSphere-v1 | 2M | 0% | pick-and-place into bin |
| PullCubeTool-v1 | 10M | 12.5% (1/8 eval, luck) | L-tool hooking + pulling |

Reference: GTP-FA (arXiv:2606.03385) reports these tasks need 50M steps × 2048 envs on 2× RTX 4090. The failure is a compute-budget issue, not an observation-interface issue — reward curves rise but the full contact chain is never learned reliably.

## Single-Task Baselines (official PPO, zero algorithm changes)

Script: `cl/ppo_official_unified.py` — a fork of the official `examples/baselines/ppo/ppo.py` with **only** the 42-dim obs conversion inserted; algorithm, hyperparameters and network architecture are untouched. CPU adaptation: 16 envs, 512 rollout steps (batch 8192).

| Task | Training steps | Best SR |
|------|---------------|---------|
| PushCube-v1 | 2M | **100%** |
| PullCube-v1 | 2M | **100%** |
| PickCube-v1 | 5.12M (converged within 10M run) | **100%** |
| LiftPegUpright-v1 | 2M | **100%** |

Checkpoints: `cl/ckpts/ppo_official_unified/{env}_seed1_{latest,final}.pt`, results in `{env}_seed1_results.json`.

### Reproduce

```powershell
conda activate diffcl10
cd cl
python ppo_official_unified.py --env-id PushCube-v1       --total-timesteps 2000000
python ppo_official_unified.py --env-id PullCube-v1       --total-timesteps 2000000
python ppo_official_unified.py --env-id PickCube-v1       --total-timesteps 10000000
python ppo_official_unified.py --env-id LiftPegUpright-v1 --total-timesteps 2000000
```

## Next Stage: Continual Learning

With single-task baselines established, the CL phase compares:
- **SeqFT** — sequential fine-tuning (catastrophic forgetting baseline)
- **ER** — Experience Replay: replay buffer of previous-task transitions mixed into PPO updates (primary method per advisor direction)
- Parameter isolation / adapter ablations (earlier Route A experiments in `experiments/route_a/`)

CL entry point: `cl/ppo_er_unified.py` (built on the same unified 42-dim interface).

## Project Structure

```
maniskill-cl/
├── cl/
│   ├── unified_obs.py            # 42-dim semantic-slot observation interface
│   ├── ppo_official_unified.py   # Official PPO + unified obs (single-task baselines)
│   ├── ppo_er_unified.py         # PPO + Experience Replay (CL phase)
│   ├── cl_methods.py             # SeqFT / ER / DISTR method implementations
│   ├── env_wrapper.py            # Multi-task env wrapper
│   ├── config.py
│   ├── main.py
│   ├── evaluate.py
│   └── ckpts/                    # trained checkpoints & results json
├── baselines/                    # earlier single-task PPO scripts
├── experiments/route_a/          # parameter isolation / LoRA ablation experiments
└── docs/PROGRESS_REPORT.md
```

## Environment

```powershell
conda activate diffcl10   # Python 3.10, ManiSkill3, torch (CPU), gymnasium
```

## References

- [ManiSkill3](https://www.maniskill.ai/) — Benchmark for generalizable manipulation
- [TAIL (ICLR 2024)](https://arxiv.org/abs/2310.05905) — Task-Aware Incremental Learning
- GTP-FA (arXiv:2606.03385) — GPU-scale PPO reference results on ManiSkill task suite
