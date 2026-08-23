"""
main.py - Continual Learning Framework Main Entry Point

Orchestrates the full CL pipeline:
  1. Generate expert data (or load if exists)
  2. For each CL method (SeqFT, ER, DISTR):
     a. Train on tasks sequentially
     b. Evaluate all learned tasks after each training step
     c. Compute CL metrics (FWT, BWT, Forgetting, AUC)
  3. Compare methods and save results

Run:
  cd E:\\munichi\\continual_learning_v2
  $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" -u main.py --method all --bc-epochs 50 --eval-episodes 20

Specific method:
  & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" -u main.py --method seqft
  & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" -u main.py --method er
  & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" -u main.py --method distr

Options:
  --expert-model-dir DIR   Directory containing pre-trained expert model files
  --skip-expert-gen        Skip expert data generation (use existing .npz files)
  --bc-epochs N            BC training epochs (default: 50)
  --eval-episodes N        Evaluation episodes per task (default: 20)
  --seed N                 Random seed (default: 42)
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch
import random
import gymnasium as gym
import mani_skill.envs  # Register ManiSkill environments

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TASK_CONFIGS, TASK_SEQUENCE, TASK_NAMES, NUM_TASKS,
    EXPERT_DATA_DIR, MODEL_DIR, RESULT_DIR,
    BC_CONFIG, EVAL_CONFIG, EXPERT_CONFIG, DEVICE,
    USE_SCRIPTED_EXPERT,
)
from env_wrapper import ManiSkillWrapper, PolicyRunner
from bc_agent import BCAgent
from cl_methods import create_cl_method, SeqFT, ER, DISTR
from evaluate import (
    create_success_matrix, compute_all_metrics,
    print_results, print_comparison, save_results,
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continual Learning Framework for ManiSkill3"
    )
    parser.add_argument(
        "--method", type=str, default="all",
        choices=["seqft", "er", "distr", "all"],
        help="CL method to run (default: all)"
    )
    parser.add_argument(
        "--expert-model-dir", type=str, default="",
        help="Directory containing pre-trained expert model files"
    )
    parser.add_argument(
        "--skip-expert-gen", action="store_true",
        help="Skip expert data generation (use existing .npz files)"
    )
    parser.add_argument(
        "--bc-epochs", type=int, default=BC_CONFIG["epochs"],
        help=f"BC training epochs (default: {BC_CONFIG['epochs']})"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=EVAL_CONFIG["n_episodes"],
        help=f"Evaluation episodes per task (default: {EVAL_CONFIG['n_episodes']})"
    )
    parser.add_argument(
        "--seed", type=int, default=EVAL_CONFIG["seed"],
        help=f"Random seed (default: {EVAL_CONFIG['seed']})"
    )
    parser.add_argument(
        "--n-expert-episodes", type=int, default=EXPERT_CONFIG["n_episodes"],
        help=f"Number of expert episodes to collect (default: {EXPERT_CONFIG['n_episodes']})"
    )
    parser.add_argument(
        "--use-scripted-expert", action="store_true", default=USE_SCRIPTED_EXPERT,
        help="Use scripted expert policies instead of pre-trained models (default: True)"
    )
    parser.add_argument(
        "--no-scripted-expert", action="store_true",
        help="Disable scripted expert, use pre-trained model files instead"
    )
    return parser.parse_args()


# ============================================================
# Expert Data Generation
# ============================================================

def generate_expert_data(args):
    """Generate expert trajectories for all tasks using scripted expert policies
    (or pre-trained models if scripted expert is disabled).

    If expert .npz files already exist and --skip-expert-gen is set, skip.
    """
    os.makedirs(EXPERT_DATA_DIR, exist_ok=True)

    # Determine expert source
    use_scripted = args.use_scripted_expert and not args.no_scripted_expert
    if use_scripted:
        # Override all task configs to use scripted expert
        for task_name in TASK_SEQUENCE:
            TASK_CONFIGS[task_name]["expert_type"] = "scripted"
        print(f"\n[Expert] Using SCRIPTED EXPERT policies (no pre-trained model needed)")
    else:
        print(f"\n[Expert] Using PRE-TRAINED MODEL policies")

    expert_data_all = {}

    for task_name in TASK_SEQUENCE:
        task_config = TASK_CONFIGS[task_name]
        task_id = task_config["task_id"]
        save_path = os.path.join(EXPERT_DATA_DIR, f"task_{task_id}_expert.npz")

        # Check if already exists
        if args.skip_expert_gen and os.path.exists(save_path):
            print(f"\n[Expert] Loading existing data for {task_name} from {save_path}")
            expert_data_all[task_name] = PolicyRunner.load_expert_data(save_path)
            n = expert_data_all[task_name]["n_episodes"]
            sr = expert_data_all[task_name]["successes"].mean() * 100
            print(f"  Loaded {n} episodes, success rate: {sr:.1f}%")
            continue

        if use_scripted:
            # Scripted expert: no model file needed
            print(f"\n[Expert] Collecting scripted expert data for {task_name}...")
            print(f"  Type: scripted expert policy")

            runner = PolicyRunner(task_name, expert_model_path=None)
            runner.load_policy()

            env = ManiSkillWrapper(task_config["env_id"], task_id)

            episodes = runner.collect_episodes(
                env,
                n_episodes=args.n_expert_episodes,
                max_steps=EXPERT_CONFIG["max_steps"],
                seed=args.seed,
            )

            runner.save_expert_data(episodes, save_path)
            expert_data_all[task_name] = PolicyRunner.load_expert_data(save_path)
            env.close()
        else:
            # Pre-trained model path
            expert_model_path = os.path.join(args.expert_model_dir, f"{task_name}_best.zip") \
                if args.expert_model_dir else task_config["expert_model_path"]

            # Check model path
            if not expert_model_path or not os.path.exists(expert_model_path):
                print(f"\n[Expert] WARNING: No pre-trained model for {task_name} at "
                      f"{expert_model_path}")
                print(f"  Generating random expert data (for testing only)!")
                expert_data_all[task_name] = _generate_random_expert(
                    task_name, task_id, args.n_expert_episodes, save_path
                )
                continue

            # Load policy and collect data
            print(f"\n[Expert] Collecting expert data for {task_name}...")
            print(f"  Model: {expert_model_path}")
            print(f"  Type: {task_config['expert_type']}")

            runner = PolicyRunner(task_name, expert_model_path)
            runner.load_policy()

            env = ManiSkillWrapper(task_config["env_id"], task_id)

            episodes = runner.collect_episodes(
                env,
                n_episodes=args.n_expert_episodes,
                max_steps=EXPERT_CONFIG["max_steps"],
                seed=args.seed,
            )

            runner.save_expert_data(episodes, save_path)
            expert_data_all[task_name] = PolicyRunner.load_expert_data(save_path)
            env.close()

    return expert_data_all


def _generate_random_expert(task_name, task_id, n_episodes, save_path):
    """Generate random expert data for testing (no pre-trained model)."""
    print(f"  Generating RANDOM data for {task_name} (test mode)")

    episodes = []
    for ep in range(n_episodes):
        n_steps = random.randint(50, 200)
        obs = np.random.randn(n_steps, 52).astype(np.float32)
        actions = np.clip(np.random.randn(n_steps, 7).astype(np.float32), -1, 1)
        rewards = np.random.rand(n_steps).astype(np.float32)
        success = random.random() < 0.3  # 30% random success

        episodes.append({
            "observations": obs,
            "actions": actions,
            "rewards": rewards,
            "success": success,
            "return": rewards.sum(),
        })

    PolicyRunner.save_expert_data(episodes, save_path)
    return PolicyRunner.load_expert_data(save_path)


# ============================================================
# CL Training & Evaluation Loop
# ============================================================

def run_cl_method(method_name, expert_data_all, args):
    """Run a single CL method through the full task sequence.

    Args:
        method_name: "seqft", "er", or "distr"
        expert_data_all: dict of {task_name: expert_data_dict}
        args: command-line arguments

    Returns:
        success_matrix: (T, T) numpy array of success rates
        metrics: dict of CL metrics
    """
    print(f"\n{'#' * 70}")
    print(f"# Running CL Method: {method_name.upper()}")
    print(f"{'#' * 70}")

    set_seed(args.seed)

    # Create CL method
    cl_method = create_cl_method(method_name, device=DEVICE)

    # Create success matrix
    success_matrix = create_success_matrix(NUM_TASKS)

    # Task sequence training
    for step, task_name in enumerate(TASK_SEQUENCE):
        task_config = TASK_CONFIGS[task_name]
        task_id = task_config["task_id"]

        print(f"\n{'=' * 50}")
        print(f"  Task {step + 1}/{NUM_TASKS}: {task_name} (id={task_id})")
        print(f"{'=' * 50}")

        # Create env wrapper for current task
        train_env = ManiSkillWrapper(task_config["env_id"], task_id)

        # Get expert data
        expert_data = expert_data_all[task_name]

        # Train on current task
        print(f"\n  Training BC for {args.bc_epochs} epochs...")
        train_metrics = cl_method.train_task(
            task_id, expert_data, train_env,
            epochs=args.bc_epochs, verbose=True,
        )
        print(f"  Training done. Final loss: {train_metrics['final_loss']:.6f}")

        # Post-training hook (store replay data, train diffusion, etc.)
        cl_method.after_task(task_id, expert_data)

        # Save model
        model_path = os.path.join(MODEL_DIR, f"bc_{method_name}_task_{task_id}.pt")
        cl_method.save_model(model_path)

        # Evaluate on all tasks learned so far (tasks 0..step)
        print(f"\n  Evaluating on tasks 0..{step}...")
        for eval_step in range(step + 1):
            eval_task_name = TASK_SEQUENCE[eval_step]
            eval_task_config = TASK_CONFIGS[eval_task_name]
            eval_task_id = eval_task_config["task_id"]

            eval_env = ManiSkillWrapper(eval_task_config["env_id"], eval_task_id)
            eval_result = cl_method.evaluate(
                eval_env,
                n_episodes=args.eval_episodes,
                max_steps=EVAL_CONFIG["max_steps"],
                seed=args.seed,
                verbose=False,
            )
            success_matrix[step, eval_step] = eval_result["success_rate"]
            print(f"    Task {eval_task_name}: SR={eval_result['success_rate']:.2%} "
                  f"({eval_result['n_success']}/{eval_result['n_episodes']})")
            eval_env.close()

        train_env.close()

    # Compute metrics
    metrics = compute_all_metrics(success_matrix)
    print_results(success_matrix, metrics, TASK_NAMES, method_name)

    # Save results
    save_results(success_matrix, metrics, TASK_NAMES, method_name)

    return success_matrix, metrics


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("  Continual Learning Framework for ManiSkill3")
    print("  Reference: ContinualWorld evaluation protocol")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Method:           {args.method}")
    print(f"  BC epochs:        {args.bc_epochs}")
    print(f"  Eval episodes:    {args.eval_episodes}")
    print(f"  Seed:             {args.seed}")
    print(f"  Expert episodes:  {args.n_expert_episodes}")
    print(f"  Expert source:    {'SCRIPTED' if (args.use_scripted_expert and not args.no_scripted_expert) else 'PRE-TRAINED MODEL'}")
    print(f"  Expert model dir: {args.expert_model_dir or '(not set)'}")
    print(f"  Expert data dir:  {EXPERT_DATA_DIR}")
    print(f"  Model dir:        {MODEL_DIR}")
    print(f"  Result dir:       {RESULT_DIR}")
    print(f"  Device:           {DEVICE}")
    print(f"  Tasks:            {TASK_SEQUENCE}")
    print(f"  OBS_DIM:          52 (48 padded + 4 task_id)")
    print(f"  ACTION_DIM:       7")
    print()

    # Create directories
    os.makedirs(EXPERT_DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Set seed
    set_seed(args.seed)

    # ----------------------------------------------------------
    # Step 1: Generate/load expert data
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Step 1: Expert Data Generation")
    print("=" * 70)

    start_time = time.time()
    expert_data_all = generate_expert_data(args)
    expert_time = time.time() - start_time
    print(f"\n  Expert data generation took {expert_time:.1f}s")

    # Print summary
    for task_name in TASK_SEQUENCE:
        data = expert_data_all[task_name]
        n = data["n_episodes"]
        sr = data["successes"].mean() * 100
        total_transitions = sum(len(data["obs_list"][i]) for i in range(n))
        print(f"  {task_name}: {n} episodes, {total_transitions} transitions, "
              f"SR={sr:.1f}%")

    # ----------------------------------------------------------
    # Step 2: Run CL methods
    # ----------------------------------------------------------
    methods_to_run = []
    if args.method == "all":
        methods_to_run = ["seqft", "er", "distr"]
    else:
        methods_to_run = [args.method]

    all_results = {}
    all_matrices = {}

    for method_name in methods_to_run:
        method_start = time.time()
        success_matrix, metrics = run_cl_method(
            method_name, expert_data_all, args
        )
        method_time = time.time() - method_start

        all_results[method_name] = metrics
        all_matrices[method_name] = success_matrix

        print(f"\n  [{method_name.upper()}] completed in {method_time:.1f}s")

    # ----------------------------------------------------------
    # Step 3: Compare methods
    # ----------------------------------------------------------
    if len(all_results) > 1:
        print_comparison(all_results, TASK_NAMES)

    # Save comparison summary
    summary = {
        "args": {
            "method": args.method,
            "bc_epochs": args.bc_epochs,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
            "n_expert_episodes": args.n_expert_episodes,
        },
        "methods": {},
    }
    for method_name, metrics in all_results.items():
        summary["methods"][method_name] = {
            "metrics": {k: float(v) for k, v in metrics.items()},
            "success_matrix": all_matrices[method_name].tolist(),
        }

    summary_path = os.path.join(RESULT_DIR, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.1f}s")
    print("\nDone!")


if __name__ == "__main__":
    main()
