#!/usr/bin/env python3
"""Paste table rows, get MAC (Macro-Average across all k)."""
import sys
import numpy as np

print("Paste table rows (Ctrl-D to finish):")
lines = sys.stdin.read().strip().splitlines()

f1s, precs, recalls, size_rs, pred_rs = [], [], [], [], []
for line in lines:
    parts = line.split("|")
    if len(parts) < 6:
        continue
    try:
        vals = []
        for p in parts:
            p = p.strip().rstrip("x%")
            vals.append(float(p))
        _, f1, prec, recall, size_r, pred_r = vals[:6]
        f1s.append(f1)
        precs.append(prec)
        recalls.append(recall)
        size_rs.append(size_r)
        pred_rs.append(pred_r)
    except (ValueError, IndexError):
        continue

if not f1s:
    print("No valid rows found.")
    sys.exit(1)

print(f"\n  {'MAC':>3} | {np.mean(f1s):>7.4f} | {np.mean(precs):>7.4f} | "
      f"{np.mean(recalls):>7.4f} | {np.mean(size_rs):>7.2f}x | {np.mean(pred_rs):>6.2f}%")
