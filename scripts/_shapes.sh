cd /root/cl
cat > _shapes.py <<'PYEOF'
import numpy as np, os, glob
cands = ["demos/sd_dagger_r5.npz", "demos/sd_dagger_r4.npz", "demos/sd_8t.npz",
         "demos/sd_train.npz", "demos/sd_init.npz", "demos/sd_all.npz"]
for p in cands:
    if not os.path.exists(p):
        print("%-28s MISSING" % p)
        continue
    d = np.load(p, allow_pickle=True)
    ks = list(d.keys())
    n = d["obs"].shape[0]
    cnt = None
    if "tid" in ks:
        u, c = np.unique(d["tid"], return_counts=True)
        cnt = dict(zip([int(x) for x in u], [int(x) for x in c]))
    print("%-28s n=%7d obs=%s tid=%s keys=%s" % (p, n, d["obs"].shape, cnt, ks))
print("--- all sd*.npz ---")
for p in sorted(glob.glob("demos/sd*.npz")):
    try:
        d = np.load(p, allow_pickle=True)
        print("%-34s n=%7d" % (p, d["obs"].shape[0]))
    except Exception as e:
        print("%-34s ERR %s" % (p, e))
PYEOF
/root/miniconda3/envs/ms3/bin/python _shapes.py 2>&1 | tail -40
