"""Calibrate the fixed rotation between the gripper and the charger at the
moment the fingers close, straight from the expert demos."""
import numpy as np
import transforms3d as t3d

d = np.load("/root/cl/demos/plg_warm2.npz")
obs, act = d["obs"], d["act"]
print("obs", obs.shape, "act", act.shape)

ch_p, ch_q = obs[:, 25:28], obs[:, 28:32]
tp_p, tp_q = obs[:, 18:21], obs[:, 21:25]
g = act[:, 3]

closing = (g[1:] < -0.5) & (g[:-1] >= -0.5)
idx = np.where(closing)[0]
print("close events", len(idx), " g unique", np.unique(np.round(g, 1))[:8])
if len(idx) > 800:
    idx = idx[:: max(1, len(idx) // 800)]

rels, dps = [], []
for i in idx:
    q1 = ch_q[i] / np.linalg.norm(ch_q[i])
    q2 = tp_q[i] / np.linalg.norm(tp_q[i])
    q1c = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    rels.append(t3d.quaternions.qmult(q1c, q2))
    dps.append(tp_p[i] - ch_p[i])
rels = np.asarray(rels)
for _ in range(3):
    m = rels.mean(0)
    m /= np.linalg.norm(m)
    rels[rels @ m < 0] *= -1.0
m = rels.mean(0)
m /= np.linalg.norm(m)
ang = 2 * np.arccos(np.clip(np.abs(rels @ m), 0, 1))
print("REL_QUAT (wxyz)", np.round(m, 6).tolist())
print("angle dev  med %.3f deg  p90 %.3f deg"
      % (np.degrees(np.median(ang)), np.degrees(np.percentile(ang, 90))))
print("axis-angle", np.round(t3d.quaternions.quat2axang(m), 4).tolist())
dps = np.asarray(dps)
print("REL_POS mm  mean", np.round(dps.mean(0) * 1000, 2).tolist(),
      "  med", np.round(np.median(dps, 0) * 1000, 2).tolist())

# how far is the gripper orientation from the raw charger orientation?
dev = []
for i in idx[:200]:
    q1 = ch_q[i] / np.linalg.norm(ch_q[i])
    q2 = tp_q[i] / np.linalg.norm(tp_q[i])
    dev.append(2 * np.arccos(min(1.0, abs(float(np.dot(q1, q2))))))
print("|q_tcp vs q_charger| med %.1f deg" % np.degrees(np.median(dev)))
