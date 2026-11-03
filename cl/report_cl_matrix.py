import json, sys

TASKS = ["PushCube-v1", "PullCube-v1", "PickCube-v1", "LiftPegUpright-v1", "StackCube-v1"]
SHORT = {"PushCube-v1": "Push", "PullCube-v1": "Pull", "PickCube-v1": "Pick",
         "LiftPegUpright-v1": "LiftPeg", "StackCube-v1": "Stack"}


def shared_log(d):
    # the trainer stores the same (cumulative) eval list in every per_task entry
    for v in d["per_task_results"].values():
        return v["eval_log"]
    return []


def rows_by_task(log):
    out, cur = {}, None
    for ev in log:
        t = ev.get("task")
        if t is not None:
            out[t] = ev          # keep the LAST eval recorded while that task trained
    return out


for path in sys.argv[1:]:
    d = json.load(open(path, encoding="utf-8"))
    print("\n===== %s | replay_ratio=%s =====" % (
        d.get("run_tag", "?"), d["er_config"]["replay_ratio"]))
    print("%-9s %10s  %s" % ("after", "steps", "".join("%9s" % SHORT[t] for t in TASKS)))
    last = rows_by_task(shared_log(d))
    for t in TASKS:
        ev = last.get(t)
        if not ev:
            continue
        res = ev["eval_results"]
        cells = "".join("%9s" % (("%.0f%%" % res[x]["success_rate"]) if x in res else "-")
                        for x in TASKS)
        print("%-9s %10d  %s" % (SHORT[t], ev["global_step"], cells))
    fin = d["per_task_results"][list(d["per_task_results"])[-1]]["final_eval"]
    srs = [fin[t]["success_rate"] for t in TASKS if t in fin]
    print("FINAL  :", {SHORT[t]: "%.0f%%" % fin[t]["success_rate"] for t in TASKS if t in fin},
          "| mean=%.1f%%" % (sum(srs) / len(srs)))
