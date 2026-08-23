"""
eval_official_checkpoints.py (v2 - 修复 state_dict key 匹配)
评估 ManiSkill3 官方 PPO checkpoint 的单任务 baseline。

修复: OfficialActor 属性名必须与 checkpoint key 完全一致:
  - actor_mean.* (不是 net.*)
  - actor_logstd  (不是 logstd)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

# PATH 修复
python_dir = Path(sys.executable).parent
os.environ["PATH"] = (
    str(python_dir) + ";"
    + str(python_dir / "Library" / "bin") + ";"
    + os.environ.get("PATH", "")
)

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill  # ★ 必须导入以注册 ManiSkill3 环境

# ═════════════════════════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════════════════════════
DEMOS_DIR = Path.home() / ".maniskill" / "demos"
CKPT_NAME = "ppo_pd_joint_delta_pos_ckpt.pt"
CONTROL_MODE = "pd_joint_delta_pos"
NUM_EPISODES = 50

TASKS = {
    "PushCube-v1":  {"obs_dim": 35, "act_dim": 8},
    "PullCube-v1":  {"obs_dim": 35, "act_dim": 8},
    "PickCube-v1":  {"obs_dim": 42, "act_dim": 8},
    "PokeCube-v1":  {"obs_dim": 54, "act_dim": 8},
    "StackCube-v1": {"obs_dim": 48, "act_dim": 8},
}

# ═════════════════════════════════════════════════════════════════════════════
# 模型定义 — key 名必须与 checkpoint 完全一致
# ═════════════════════════════════════════════════════════════════════════════
class OfficialActor(nn.Module):
    """
    匹配官方 checkpoint 架构:
      actor_mean: obs → 256(Tanh) → 256(Tanh) → 256(Tanh) → act
      actor_logstd: 可学习参数 [1, act_dim]
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # ★ 关键: 属性名必须是 actor_mean，不是 net
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh(),
            nn.Linear(256, act_dim),
        )
        # ★ 关键: 属性名必须是 actor_logstd，不是 logstd
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs, deterministic=True):
        mean = self.actor_mean(obs)
        std = self.actor_logstd.expand_as(mean).exp()
        if deterministic:
            return mean
        return mean + std * torch.randn_like(mean)


def evaluate_checkpoint(task_id, obs_dim, act_dim):
    """评估单个任务的官方 checkpoint"""
    ckpt_path = DEMOS_DIR / task_id / "rl" / CKPT_NAME

    if not ckpt_path.exists():
        return None, f"checkpoint 不存在 → {ckpt_path}"

    # ── 加载 checkpoint ──
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        return None, "checkpoint 格式错误 (非 dict)"

    # ── 构建 actor 并加载权重 ──
    actor = OfficialActor(obs_dim, act_dim)

    # 先检查 key 匹配情况
    actor_keys = set(actor.state_dict().keys())
    ckpt_keys = set(ckpt.keys())
    matched = actor_keys & ckpt_keys
    missing_in_ckpt = actor_keys - ckpt_keys
    extra_in_ckpt = ckpt_keys - actor_keys

    if not matched:
        return None, (
            f"key 完全不匹配!\n"
            f"    actor 需要: {sorted(actor_keys)}\n"
            f"    ckpt 提供: {sorted(ckpt_keys)}"
        )

    # 加载 (strict=False 忽略 critic 等无关 key)
    actor.load_state_dict(ckpt, strict=False)
    actor.eval()

    # 验证权重确实被加载了（不是随机初始化）
    first_weight = actor.actor_mean[0].weight.data
    weight_mean = first_weight.abs().mean().item()
    if weight_mean < 1e-6:
        return None, f"权重加载失败 (第一层权重全为0, mean={weight_mean:.2e})"

    info = (
        f"加载成功 (obs={obs_dim}, act={act_dim}) | "
        f"匹配 {len(matched)}/{len(actor_keys)} keys | "
        f"|W|_mean={weight_mean:.4f}"
    )

    # ── 环境评估 ──
    env = gym.make(task_id, control_mode=CONTROL_MODE)
    successes = 0
    total_reward = 0.0

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        # 去掉 batch 维
        if hasattr(obs, 'shape') and len(obs.shape) == 2:
            obs = obs[0]
        obs = np.array(obs, dtype=np.float32)

        ep_reward = 0.0
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim)
            with torch.no_grad():
                action = actor(obs_t, deterministic=True)
            action_np = action.squeeze(0).numpy().clip(-1.0, 1.0)
            obs, reward, terminated, truncated, info_step = env.step(action_np)
            if hasattr(obs, 'shape') and len(obs.shape) == 2:
                obs = obs[0]
            obs = np.array(obs, dtype=np.float32)
            ep_reward += float(reward)
            done = terminated or truncated

        # 检查成功 (ManiSkill3 的 info 中 success 可能是 tensor)
        success = info_step.get("success", False)
        if hasattr(success, 'item'):
            success = success.item()
        if hasattr(success, '__len__'):
            success = any(s.item() if hasattr(s, 'item') else bool(s) for s in success)
        if success:
            successes += 1
        total_reward += ep_reward

    env.close()
    sr = successes / NUM_EPISODES * 100
    avg_r = total_reward / NUM_EPISODES
    return sr, f"SR = {sr:.0f}% ({successes}/{NUM_EPISODES}), avg_R = {avg_r:.2f} | {info}"


# ═════════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("官方 PPO Checkpoint 单任务 Baseline 评估 (v2 - key 修复)")
    print("=" * 70)

    results = {}
    for task_id, cfg in TASKS.items():
        print(f"\n{'─' * 70}")
        print(f"[{task_id}]")

        ckpt_path = DEMOS_DIR / task_id / "rl" / CKPT_NAME
        if not ckpt_path.exists():
            print(f"  ✗ checkpoint 不存在 → {ckpt_path}")
            results[task_id] = {"sr": 0.0, "note": "NOT_FOUND"}
            continue

        sr, msg = evaluate_checkpoint(task_id, cfg["obs_dim"], cfg["act_dim"])
        if sr is None:
            print(f"  ✗ 评估失败: {msg}")
            results[task_id] = {"sr": 0.0, "note": f"ERROR: {msg[:60]}"}
        else:
            print(f"  ✓ {msg}")
            results[task_id] = {"sr": sr, "note": ""}

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    max_sr = max(r["sr"] for r in results.values()) if results else 1
    for task_id, r in results.items():
        bar_len = int(r["sr"] / max(max_sr, 1) * 20)
        bar = "█" * bar_len + " " * (20 - bar_len)
        note = f" [{r['note']}]" if r["note"] else ""
        print(f"  {task_id:<16} SR={r['sr']:5.1f}% |{bar}|{note}")
    print("=" * 70)

    # 缺失提示
    missing = [t for t, r in results.items() if r["note"] == "NOT_FOUND"]
    if missing:
        print(f"\n⚠ 以下任务缺少 checkpoint，请运行下载命令:")
        for t in missing:
            print(f"  python -m mani_skill.utils.download_demo {t}")
