# -*- coding: utf-8 -*-
"""校验 BENCHMARK_8TASK.md：表格列数一致性 + 关键数字命中次数"""
import io, re, sys

P = "BENCHMARK_8TASK.md"
txt = io.open(P, encoding="utf-8").read()
lines = txt.split("\n")

# 1) 表格块列数校验
bad = []
block = []
def flush(blk, start):
    if len(blk) < 2:
        return
    counts = []
    for i, ln in enumerate(blk):
        s = ln.strip()
        cells = [c for c in s.strip("|").split("|")]
        counts.append(len(cells))
    if len(set(counts)) != 1:
        bad.append((start, counts))

start = None
for i, ln in enumerate(lines, 1):
    if ln.strip().startswith("|"):
        if start is None:
            start = i
        block.append(ln)
    else:
        if block:
            flush(block, start)
        block, start = [], None
if block:
    flush(block, start)

print("TABLE_BLOCKS_BAD=%d" % len(bad))
for s, c in bad:
    print("  line %d -> %s" % (s, c))

# 2) 关键数字
keys = ["92.6", "92.4", "93.5", "65.5", "78.4", "83.4", "86.4", "91.1",
        "182,328", "7,425", "22,625", "121,769", "48", "56", "280", "240",
        "pd_joint_delta_pos", "TASK_INFO", "0.49", "88.7", "92.0", "79.0", "76.6"]
for k in keys:
    print("KEY %-22s x%d" % (k, txt.count(k)))

# 3) 章节标题清单
print("\nSECTIONS")
for ln in lines:
    if re.match(r"^#{2,3}\s", ln):
        print("  " + ln.strip())

# 4) 编码回读
raw = io.open(P, "rb").read()
try:
    raw.decode("utf-8")
    print("\nUTF8_OK chars=%d lines=%d" % (len(txt), len(lines)))
except Exception as e:
    print("\nUTF8_FAIL %s" % e)
