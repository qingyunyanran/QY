import json, sys

TASKS = ["PushCube-v1", "PullCube-v1", "PickCube-v1", "LiftPegUpright-v1", "StackCube-v1"]
SHORT = {"PushCube-v1": "Push", "PullCube-v1": "Pull", "PickCube-v1": "Pick",
         "LiftPegUpright-v1": "LiftPeg", "StackCube-v1": "Stack",
         # complex-skill sequence
         "StackPyramid-v1": "Pyramid", "PegInsertionSide-v1": "PegIns",
         "PlugCharger-v1": "Charger", "PokeCube-v1": "Poke",
         "PullCubeTool-v1": "PullTool", "PlaceSphere-v1": "Place"}


def short(t):
    return SHORT.get(t, t[:-3] if t.endswith("-v1") else t)


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
    # Task list comes from the run itself: the same reporter serves the 5 easy
    # tasks and the 5 complex ones without editing a constant here.
    tasks = d.get("task_sequence") or TASKS
    print("\n===== %s | replay_ratio=%s =====" % (
        d.get("run_tag", "?"), d["er_config"]["replay_ratio"]))
    print("%-9s %10s  %s" % ("after", "steps", "".join("%9s" % short(t) for t in tasks)))
    last = rows_by_task(shared_log(d))
    for t in tasks:
        ev = last.get(t)
        if not ev:
            continue
        res = ev["eval_results"]
        cells = "".join("%9s" % (("%.0f%%" % res[x]["success_rate"]) if x in res else "-")
                        for x in tasks)
        print("%-9s %10d  %s" % (short(t), ev["global_step"], cells))
    fin = d["per_task_results"][list(d["per_task_results"])[-1]]["final_eval"]
    srs = [fin[t]["success_rate"] for t in tasks if t in fin]
    print("FINAL  :", {short(t): "%.0f%%" % fin[t]["success_rate"] for t in tasks if t in fin},
          "| mean=%.1f%%" % (sum(srs) / len(srs)))
