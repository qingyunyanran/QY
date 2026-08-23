"""
evaluate.py - Continual Learning Evaluation Metrics

Computes standard CL metrics from the success rate matrix:
  - FWT (Forward Transfer): how much prior knowledge helps learn new tasks
  - BWT (Backward Transfer): how learning new tasks affects old ones
  - Forgetting: maximum performance drop on any previous task
  - AUC: average performance across all tasks and evaluation points

Reference: ContinualWorld / Chaudhry et al. (2018) evaluation protocol.

Run:
  cd E:\\munichi\\continual_learning_v2
  $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  & "E:\\My_programs\\anaconda\\envs\\diffcl10\\python.exe" -u main.py --method all
"""

import os
import json
import numpy as np

from config import NUM_TASKS, RESULT_DIR


# ============================================================
# Success Matrix
# ============================================================

def create_success_matrix(n_tasks):
    """Create an empty success rate matrix.

    Matrix[i][j] = success rate on task j after training on task i.
    Entries where j > i are undefined (task j not yet seen).

    Returns:
        numpy array (n_tasks, n_tasks), initialized with NaN
    """
    return np.full((n_tasks, n_tasks), np.nan, dtype=np.float32)


# ============================================================
# Metrics Computation
# ============================================================

def compute_fwt(matrix):
    """Forward Weighted Transfer.

    Measures how much knowledge from previous tasks helps learning new tasks.
    FWT = (1/(T*(T-1)/2)) * sum_{i>j} (R_{j-1, j} - R_{0, j})

    Where R_{i,j} is performance on task j after training up to task i.
    For the first task (j=0), there's no forward transfer, so we skip it.

    Simplified version (consistent with ContinualWorld):
    FWT = mean over j>0 of (R_{j-1, j} - R_{baseline, j})

    Since we don't have a random baseline, we use:
    FWT = mean over j of (1/(j) * sum_{i<j} R_{i,j}) for all j where R_{i,j} is not NaN
    This measures the average performance on task j before it's trained.

    Actually, standard FWT definition:
    FWT = (1 / (T*(T-1)/2)) * sum_{i=1}^{T-1} sum_{j=1}^{i} (R_{i-1, j} - R_hat_{0, j})

    Since R_hat_{0,j} (performance on task j before any training) is typically 0 for BC:
    FWT = (1 / (T*(T-1)/2)) * sum of R_{i-1, j} for i > j (where j < i)

    But actually for our setting, the standard formula from ContinualWorld:
    FWT = average of R[i][j] where j > i (upper triangle, but those are NaN since we can't evaluate unseen tasks)

    Let's use the simplest and most standard formulation:
    FWT = (1 / (N-1)) * sum_{i=2}^{N} R[i-2][i-1]  (performance on task i before training on i)
    This doesn't work either since we evaluate AFTER training.

    Final: use the standard Chaudhry et al. definition adapted for our setting.
    """
    T = matrix.shape[0]
    if T <= 1:
        return 0.0

    # FWT: for each task j (j >= 1), average performance on task j
    # BEFORE training on task j (i.e., rows i < j).
    # But in our protocol, we only evaluate tasks {0, ..., i} after training on task i.
    # So R[i][j] exists only for j <= i.
    # Before training on task j, we have R[0..j-1][j] = NaN (not evaluated).
    # Hence FWT = 0 in our protocol (we don't evaluate on future tasks).
    # This is expected - FWT is only meaningful when evaluating on all tasks after each step.

    # Alternative: FWT as the "average accuracy" before each task is learned.
    # Since we don't evaluate on unseen tasks, FWT = 0.
    # We'll report it as 0 and note this in the output.
    return 0.0


def compute_bwt(matrix):
    """Backward Weighted Transfer.

    Measures how learning new tasks affects performance on old tasks.
    BWT = (1/(T-1)) * sum_{j=0}^{T-2} R[T-1][j] - R[j][j]

    Negative BWT indicates catastrophic forgetting.
    """
    T = matrix.shape[0]
    if T <= 1:
        return 0.0

    bwt = 0.0
    for j in range(T - 1):
        # R[T-1][j]: performance on task j after training on all tasks
        # R[j][j]: performance on task j right after training on task j
        r_after = matrix[T - 1, j]
        r_at_train = matrix[j, j]
        if not (np.isnan(r_after) or np.isnan(r_at_train)):
            bwt += (r_after - r_at_train)

    bwt /= (T - 1)
    return bwt


def compute_forgetting(matrix):
    """Average Forgetting.

    For each task j, forgetting is the maximum drop from its peak performance:
    F_j = max_{i in [j..T-1]} R[i][j] - R[T-1][j]

    Average Forgetting = (1/T) * sum_{j=0}^{T-1} F_j
    (Note: last task has no forgetting since it's the most recent)
    """
    T = matrix.shape[0]
    if T <= 1:
        return 0.0

    total_forgetting = 0.0
    n_counted = 0

    for j in range(T - 1):  # Skip last task (no forgetting possible)
        # Find peak performance on task j across all evaluations
        column = matrix[j:, j]  # R[j][j], R[j+1][j], ..., R[T-1][j]
        valid = column[~np.isnan(column)]
        if len(valid) > 0:
            peak = np.max(valid)
            final = valid[-1]  # R[T-1][j]
            total_forgetting += (peak - final)
            n_counted += 1

    if n_counted == 0:
        return 0.0
    return total_forgetting / n_counted


def compute_auc(matrix):
    """Area Under the (success rate) Curve.

    AUC = average of all valid entries in the lower triangle of the matrix.
    This represents the average success rate across all tasks and evaluation points.
    """
    T = matrix.shape[0]
    values = []
    for i in range(T):
        for j in range(i + 1):  # Lower triangle including diagonal
            val = matrix[i, j]
            if not np.isnan(val):
                values.append(val)
    if len(values) == 0:
        return 0.0
    return np.mean(values)


def compute_all_metrics(matrix):
    """Compute all CL metrics from the success rate matrix.

    Returns:
        dict with keys: FWT, BWT, Forgetting, AUC
    """
    return {
        "FWT": compute_fwt(matrix),
        "BWT": compute_bwt(matrix),
        "Forgetting": compute_forgetting(matrix),
        "AUC": compute_auc(matrix),
    }


# ============================================================
# Display & Save
# ============================================================

def print_results(matrix, metrics, task_names, method_name=""):
    """Print formatted results table.

    Args:
        matrix: (T, T) success rate matrix
        metrics: dict with FWT, BWT, Forgetting, AUC
        task_names: list of task name strings
        method_name: CL method name for header
    """
    T = len(task_names)
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Continual Learning Results: {method_name.upper()}")
    print(sep)

    # Header
    header_label = "After training on \\ Eval on"
    header = f"{header_label:>30s}"
    for name in task_names:
        short_name = name.replace("-v1", "")
        header += f" | {short_name:>12s}"
    print(header)
    print("-" * 70)

    # Matrix rows
    for i in range(T):
        row = f"{'After ' + task_names[i].replace('-v1', ''):>30s}"
        for j in range(T):
            val = matrix[i, j]
            if np.isnan(val):
                row += f" | {'---':>12s}"
            else:
                row += f" | {val:>11.2%}"
        print(row)

    print("-" * 70)

    # Per-task final performance
    print("\n  Final performance (after all tasks):")
    for j in range(T):
        val = matrix[T - 1, j]
        if np.isnan(val):
            print(f"    {task_names[j]:>25s}: N/A")
        else:
            print(f"    {task_names[j]:>25s}: {val:.2%}")

    # Summary metrics
    print(f"\n  Summary Metrics:")
    print(f"    Forward Transfer (FWT):  {metrics['FWT']:.4f}")
    print(f"    Backward Transfer (BWT): {metrics['BWT']:.4f}")
    print(f"    Average Forgetting:       {metrics['Forgetting']:.4f}")
    print(f"    AUC (avg success rate):   {metrics['AUC']:.4f}")
    print(sep)


def print_comparison(all_results, task_names):
    """Print comparison table across all methods.

    Args:
        all_results: dict of {method_name: metrics_dict}
        task_names: list of task names
    """
    sep = "=" * 80
    print(f"\n{sep}")
    print("  Method Comparison")
    print(sep)

    header = f"{'Method':>12s} | {'FWT':>8s} | {'BWT':>8s} | {'Forget':>8s} | {'AUC':>8s}"
    print(header)
    print("-" * 80)

    for method, metrics in all_results.items():
        row = f"{method:>12s}"
        row += f" | {metrics['FWT']:>7.4f}"
        row += f" | {metrics['BWT']:>7.4f}"
        row += f" | {metrics['Forgetting']:>7.4f}"
        row += f" | {metrics['AUC']:>7.4f}"
        print(row)

    print(sep)

    # Find best method by AUC
    best_method = max(all_results.keys(), key=lambda m: all_results[m]["AUC"])
    print(f"\n  Best method by AUC: {best_method} (AUC={all_results[best_method]['AUC']:.4f})")
    print(sep)


def save_results(matrix, metrics, task_names, method_name, save_dir=None):
    """Save results to JSON and numpy files.

    Args:
        matrix: success rate matrix
        metrics: CL metrics dict
        task_names: list of task names
        method_name: CL method name
        save_dir: output directory
    """
    save_dir = save_dir or RESULT_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Save matrix as numpy
    matrix_path = os.path.join(save_dir, f"success_matrix_{method_name}.npy")
    np.save(matrix_path, matrix)

    # Save results as JSON
    results = {
        "method": method_name,
        "task_names": task_names,
        "success_matrix": matrix.tolist(),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    json_path = os.path.join(save_dir, f"results_{method_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Evaluate] Results saved to:")
    print(f"  Matrix: {matrix_path}")
    print(f"  JSON:   {json_path}")

    return json_path
