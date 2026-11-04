# -*- coding: utf-8 -*-
"""Build the end-of-run markdown report for the 5-task CL sequence.

Two independent sources are combined on purpose:

  * the training JSON  (results/ppo_er_unified/ppo_er_unified_results_<tag>.json)
    -> the forgetting matrix: what the shared actor scored on every task it had
       already seen, measured every 25 iterations DURING the run (20 episodes,
       so each cell is +-11 pp at 1 sigma -- good enough to see a trend, not to
       quote as a headline).
  * the re-evaluation log (logs/<tag>_eval100.log)
    -> the headline numbers: a fresh process, fresh sims, 100 episodes/task.
       This is what the report quotes.

Usage
-----
python summarize_c5.py \
    --expert StackCube-v1=87,PullCube-v1=100,PickCube-v1=98,\
LiftPegUpright-v1=95,PushCube-v1=100 \
    --run "ER=<json>=<log>" --run "SeqFT=<json>=<log>" ...

A run whose log is missing is still reported (100-ep columns become "-"), so a
half-finished run never breaks the table.
"""
import argparse
import json
import os
import re

SEQ = ["StackCube-v1", "PullCube-v1", "PickCube-v1",
       "LiftPegUpright-v1", "PushCube-v1"]
SHORT = {"PushCube-v1": "Push", "PullCube-v1": "Pull", "PickCube-v1": "Pick",
         "LiftPegUpright-v1": "LiftPeg", "StackCube-v1": "Stack"}
SR_LINE = re.compile(r"^\s+(\S+)\s+SR=\s*([\d.]+)%\s+avg_R=\s*(-?[\d.]+)")


def short(t):
    return SHORT.get(t, t[:-3] if t.endswith("-v1") else t)


def parse_eval100(path):
    """{full_task_name: (sr, avg_r)} from an eval_ckpt_5tasks.py log."""
    if not path or not os.path.isfile(path):
        return None
    # eval_ckpt_5tasks.py prints the bare env stem (StackCube, LiftPegUpright),
    # so index on that -- SHORT[..] ("Stack", "Push") is the report's own label.
    by_short = {}
    for t in SEQ:
        by_short[t[:-3] if t.endswith("-v1") else t] = t
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = SR_LINE.match(line)
        if m and m.group(1) in by_short:
            out[by_short[m.group(1)]] = (float(m.group(2)), float(m.group(3)))
    return out or None


def forgetting_matrix(path):
    """[(label, steps, {task: sr})] -- the last eval recorded for each task."""
    d = json.load(open(path, encoding="utf-8"))
    log = []
    for v in d["per_task_results"].values():   # shared, cumulative eval list
        log = v.get("eval_log") or []
        break
    seen, order = {}, []
    for ev in log:
        t = ev.get("task")
        if t is None:
            continue
        if t not in seen:
            order.append(t)
        seen[t] = ev
    rows = [(short(t), seen[t]["global_step"],
             {k: v["success_rate"] for k, v in seen[t]["eval_results"].items()})
            for t in order]
    return d, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="",
                    help="task=SR 100-episode expert table, comma separated")
    ap.add_argument("--run", action="append", default=[],
                    help="label=results.json=eval100.log (log may be omitted)")
    ap.add_argument("--tasks", default=",".join(SEQ))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]

    expert = {}
    for item in a.expert.split(","):
        if "=" in item:
            k, v = item.split("=")
            expert[k.strip()] = float(v)

    runs = []
    for spec in a.run:
        parts = spec.split("=")
        label, js = parts[0], parts[1] if len(parts) > 1 else ""
        lg = parts[2] if len(parts) > 2 else ""
        runs.append({"label": label, "json": js, "log": lg,
                     "eval100": parse_eval100(lg)})

    L = []
    L.append("# 5-task continual learning -- end-of-run report\n")
    L.append("Sequence: " + " -> ".join(short(t) for t in tasks) +
             "  (2.5M steps/task, 48-dim unified obs, 128 envs x 64 steps)\n")

    # ---- headline table: 100 episodes per task -------------------------
    L.append("## Final success rate, 100 episodes per task\n")
    head = "| Task | expert | " + " | ".join(r["label"] for r in runs) + " |"
    L.append(head)
    L.append("|" + "---|" * (len(runs) + 2))
    for t in tasks:
        cells = []
        for r in runs:
            s = r["eval100"]
            cells.append("%.0f%%" % s[t][0] if s and t in s else "-")
        e = "%.0f%%" % expert[t] if t in expert else "-"
        L.append("| %s | %s | %s |" % (short(t), e, " | ".join(cells)))

    def mean_of(getter):
        vals = [getter(t) for t in tasks]
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    # mean-SR row is appended last so it reads as a summary of the table above
    L.append("| **mean SR** | %.0f%% | " % (mean_of(expert.get) or 0) + " | ".join(
        ("%.0f%%" % (mean_of(lambda t, r=r: r["eval100"][t][0]
                             if r["eval100"] and t in r["eval100"] else None) or 0))
        for r in runs) + " |")

    # ---- forgetting, 100-episode numbers -------------------------------
    L.append("\n## Forgetting vs the single-task experts (100 episodes)\n")
    L.append("| Task | expert | " + " | ".join(r["label"] for r in runs) + " |")
    L.append("|" + "---|" * (len(runs) + 2))
    for t in tasks:
        if t not in expert:
            continue
        cells = []
        for r in runs:
            s = r["eval100"]
            cells.append("%+.0f pp" % (s[t][0] - expert[t])
                         if s and t in s else "-")
        L.append("| %s | %.0f%% | %s |" % (short(t), expert[t], " | ".join(cells)))
    L.append("")

    # ---- training-time matrices (20 episodes, every 25 iters) ----------
    for r in runs:
        if not r["json"] or not os.path.isfile(r["json"]):
            continue
        try:
            d, rows = forgetting_matrix(r["json"])
        except Exception as exc:                       # noqa: BLE001
            L.append("\n(matrix unavailable for %s: %s)" % (r["label"], exc))
            continue
        L.append("\n## Training-time forgetting matrix -- %s\n" % r["label"])
        L.append("Rows: the last eval recorded while that task was trained "
                 "(20 episodes). Columns: every task seen so far.\n")
        L.append("| after | steps | " + " | ".join(short(t) for t in tasks) + " |")
        L.append("|" + "---|" * (len(tasks) + 2))
        for label, steps, res in rows:
            L.append("| %s | %d | " % (label, steps) + " | ".join(
                ("%.0f%%" % res[t]) if t in res else "-" for t in tasks) + " |")

    txt = "\n".join(L) + "\n"
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
        print("wrote", a.out)
    print(txt)


if __name__ == "__main__":
    main()
