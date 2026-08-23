"""Config - Direct Continual Learning with Pre-trained Models"""

import os

# ============================================================
# Directory Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = PROJECT_ROOT

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
EXPERT_DATA_DIR = os.path.join(DATA_ROOT, "expert_data")
RESULT_DIR = os.path.join(DATA_ROOT, "results")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPERT_DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# Task Configurations
# ============================================================
TASK_CONFIGS = {
    "PushCube-v1": {
        "task_id": 0,
        "env_id": "PushCube-v1",
        "raw_obs_dim": 35,
        "expert_type": "pretrained",
        "expert_model_path": r"E:\munichi\debug_ppo_PushCube_v1_pd_joint_delta_pos.pt",
        "control_mode": "pd_joint_delta_pos",
    },
    "PickCube-v1": {
        "task_id": 1,
        "env_id": "PickCube-v1",
        "raw_obs_dim": 42,
        "expert_type": "pretrained",
        "expert_model_path": r"E:\munichi\debug_ppo_PickCube_v1_pd_joint_delta_pos.pt",
        "control_mode": "pd_joint_delta_pos",
    },
    "StackCube-v1": {
        "task_id": 2,
        "env_id": "StackCube-v1",
        "raw_obs_dim": 48,
        "expert_type": "pretrained",
        "expert_model_path": r"E:\munichi\stackcube_bc_v2_policy.pt",
        "control_mode": "pd_joint_delta_pos",
    },
    "PegInsertionSide-v1": {
        "task_id": 3,
        "env_id": "PegInsertionSide-v1",
        "raw_obs_dim": 43,
        "expert_type": "pretrained",
        "expert_model_path": r"E:\munichi\peg_bc_policy.pt",
        "control_mode": "pd_joint_delta_pos",
    },
}

# Ordered task sequence for continual learning
TASK_SEQUENCE = ["PushCube-v1", "PickCube-v1", "StackCube-v1", "PegInsertionSide-v1"]
TASK_NAMES = list(TASK_SEQUENCE)
NUM_TASKS = len(TASK_SEQUENCE)
TASK_NAME_TO_ID = {name: TASK_CONFIGS[name]["task_id"] for name in TASK_SEQUENCE}

# ============================================================
# Observation / Action Dimensions
# ============================================================
MAX_OBS_DIM = 48
TASK_ID_DIM = NUM_TASKS
OBS_DIM = MAX_OBS_DIM + TASK_ID_DIM  # 52
ACTION_DIM = 7  # For pd_joint_delta_pos control mode

# ============================================================
# Behavior Cloning (BC) Hyperparameters
# ============================================================
BC_CONFIG = {
    "lr": 3e-4,
    "epochs": 50,
    "batch_size": 256,
    "hidden_dim": 256,
    "noise_std": 0.01,
    "noise_decay": 0.99,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
}

# ============================================================
# Continual Learning Method Hyperparameters
# ============================================================
# SeqFT - Sequential Fine-tuning
SEQFT_CONFIG = {}

# ER - Experience Replay
ER_CONFIG = {
    "buffer_size": 5000,
    "batch_size": 256,
    "replay_freq": 0.5,  # 50% replay data, 50% current task
}

# DISTR - Distillation-based Strategy
DISTR_CONFIG = {
    "distill_alpha": 0.5,
    "distill_temperature": 2.0,
}

# ============================================================
# Evaluation
# ============================================================
EVAL_CONFIG = {
    "episodes_per_task": 10,
    "save_results": True,
}
