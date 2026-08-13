"""HD200 通道削减效应：C_BA vs 通道数 × 目标数的交互作用。

数据源: BL05 offline_tdca_grid → decision_channel/capacity_aggregate.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

TASK_DIR = Path(__file__).resolve().parent
CSV = TASK_DIR / "results" / "offline_tdca_grid" / "decision_channel" / "capacity_aggregate.csv"

df = pd.read_csv(CSV)

print("=" * 100)
print("HD200: Channel Reduction Effect on C_BA (信道物理分辨率 vs 信源分辨率)")
print("=" * 100)
print("  Targets    Ch   Win    Acc%      C0    C_BA    eta%   loss_vs_66ch")
print("-" * 100)

for n_targets in [40, 120, 200]:
    c0 = np.log2(n_targets)
    ref_66 = {}
    for n_ch in [66, 32, 21, 9]:
        for w_ms in [200, 500]:
            row = df[(df["targets"] == n_targets) & (df["channels"] == n_ch) & (df["window_ms"] == w_ms)]
            if row.empty:
                continue
            cba = row["c_ba_mean"].values[0]
            acc = row["accuracy_mean"].values[0] * 100
            eta = cba / c0 * 100
            key = (n_targets, w_ms)
            if n_ch == 66:
                ref_66[key] = cba
            loss = (ref_66.get(key, cba) - cba) / ref_66.get(key, cba) * 100 if key in ref_66 else 0
            print(f"  {n_targets:>6d}  {n_ch:>4d}  {w_ms:>4d}  {acc:>6.1f}  {c0:>6.2f}  {cba:>6.3f}  {eta:>5.1f}%  {loss:>8.1f}%")
    print()

print()
print("=" * 100)
print("Full C_BA matrix: 200 targets (C0 = log2(200) = 7.64 bits)")
print("=" * 100)
print("    Ch    100ms   200ms   300ms   400ms   500ms")
print("-" * 60)
for n_ch in [66, 32, 21, 9]:
    vals = []
    for w_ms in [100, 200, 300, 400, 500]:
        row = df[(df["targets"] == 200) & (df["channels"] == n_ch) & (df["window_ms"] == w_ms)]
        if not row.empty:
            v = row["c_ba_mean"].values[0]
            vals.append(f"{v:.3f}")
        else:
            vals.append("  --  ")
    print(f"  {n_ch:>4d}  {vals[0]:>7s} {vals[1]:>7s} {vals[2]:>7s} {vals[3]:>7s} {vals[4]:>7s}")

print()
print("Accuracy matrix: 200 targets")
print("    Ch    100ms   200ms   300ms   400ms   500ms")
print("-" * 60)
for n_ch in [66, 32, 21, 9]:
    vals = []
    for w_ms in [100, 200, 300, 400, 500]:
        row = df[(df["targets"] == 200) & (df["channels"] == n_ch) & (df["window_ms"] == w_ms)]
        if not row.empty:
            v = row["accuracy_mean"].values[0] * 100
            vals.append(f"{v:.1f}%")
        else:
            vals.append("  --  ")
    print(f"  {n_ch:>4d}  {vals[0]:>7s} {vals[1]:>7s} {vals[2]:>7s} {vals[3]:>7s} {vals[4]:>7s}")

print()
print("=" * 100)
print("对比维度解读:")
print("  - Target 维度 (40->200): 信源字母表增大, C0 增大但 eta 下降 = 信源降分辨率")
print("  - Channel 维度 (66->9):  信道物理带宽压缩, C0 不变但 C_BA 下降 = 信道降分辨率")
print("  - 两者正交: 可分离出信源限制 vs 信道限制对容量的独立贡献")
print("=" * 100)
