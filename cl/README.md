# cl/ — Unified-Observation PPO & Continual Learning

This package implements the ManiSkill3 continual-learning pipeline on a **unified 42-dim observation interface**. All scripts use CPU vectorized environments (16 envs, `pd_joint_delta_pos`, state observations).

## Core files

| File | Role |
|------|------|
| `unified_obs.py` | **42-dim semantic-slot observation interface.** Maps each task's raw state dict into fixed slots by physical meaning: `[0:18]` Panda proprioception, `[18:25]` tcp_pose, `[25:32]` primary object, `[32:39]` second object/tool (reserved, zeros), `[39:42]` goal_pos. Relative vectors (`tcp_to_obj_pos`, `obj_to_goal_pos`, `is_grasped`) are dropped — they are linear functions of absolute poses and carry no new information. |
| `ppo_official_unified.py` | **Single-task trainer.** A fork of the official ManiSkill3 `examples/baselines/ppo/ppo.py` with *only* the 42-dim obs conversion inserted — algorithm, hyperparameters and network are untouched. Contains `MultiSingleEnvVector`, a CPU vectorization wrapper that stacks N `num_envs=1` environments (needed because several tasks do not support native batched sim on CPU). |
| `ppo_er_unified.py` | **Continual-learning trainer (next stage).** PPO + Experience Replay on the same unified interface: maintains a replay buffer of previous-task transitions and mixes them into PPO updates. |
| `cl_methods.py` | SeqFT / ER / DISTR method implementations. |
| `env_wrapper.py` | Multi-task env wrapper. |
| `config.py` | Configuration. |
| `evaluate.py` | Evaluation utilities. |
| `ckpts/ppo_official_unified/` | Single-task checkpoints: `{env}_seed1_{latest,final}.pt` and `{env}_seed1_results.json`. |

## Task suite

| Task | Raw obs dim | Slot 1 | Slot 2 | Goal |
|------|------------|--------|--------|------|
| PushCube-v1 | 35 | cube (`obj_pose`) | zeros | `goal_pos` |
| PullCube-v1 | 35 | cube (`obj_pose`) | zeros | `goal_pos` |
| PickCube-v1 | 42 | cube (`obj_pose`) | zeros | `goal_pos` |
| LiftPegUpright-v1 | 32 | peg (`obj_pose`) | zeros | zeros (success = peg upright, implicit in slot 1 orientation) |

Tasks requiring two-stage / multi-body contact (StackCube-v1, PokeCube-v1, PlaceSphere-v1, PullCubeTool-v1) do not converge under the CPU 16-env budget (up to 10M steps tried; GPU references use 50M steps × 2048 envs) and are excluded from the suite.

## Single-task results (official PPO, zero algorithm changes)

| Task | Steps | Best SR |
|------|-------|---------|
| PushCube-v1 | 2M | **100%** |
| PullCube-v1 | 2M | **100%** |
| PickCube-v1 | 5.12M (within 10M run) | **100%** |
| LiftPegUpright-v1 | 2M | **100%** |

## Usage

```powershell
conda activate diffcl10

# Single-task training
python ppo_official_unified.py --env-id PushCube-v1       --total-timesteps 2000000
python ppo_official_unified.py --env-id PullCube-v1       --total-timesteps 2000000
python ppo_official_unified.py --env-id PickCube-v1       --total-timesteps 10000000
python ppo_official_unified.py --env-id LiftPegUpright-v1 --total-timesteps 2000000
```

CPU settings: `num_envs=16`, `num_eval_envs=8`, `num_steps=512` (batch 8192, minibatch 256), `sim_backend="cpu"`, `obs_mode="state_dict"`, default `normalized_dense` reward. Throughput ≈ 120–190 sps depending on task.

## Design constraints (per advisor)

- Only the observation input/output interface may change (dim extraction, zero-padding, strict slot alignment). Algorithm and hyperparameters are not modified.
- Any added layer must come with a statement of how it is trained.
- Every version is pushed to GitHub.
