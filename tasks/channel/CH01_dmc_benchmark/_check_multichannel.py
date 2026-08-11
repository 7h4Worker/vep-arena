"""Check completeness of Benchmark multichannel capacity results."""
import pandas as pd

CSV = "tasks/channel/CH01_dmc_benchmark/results/extended/multichannel/decision_channel/capacity_by_subject_method_channels_window.csv"
df = pd.read_csv(CSV)

print(f"Shape: {df.shape}")
print(f"Subjects: {sorted(df['subject'].unique())}")
print(f"Methods: {sorted(df['method'].unique())}")
print(f"Channel configs: {sorted(df['channel_config'].unique())}")
print(f"Channels (numeric): {sorted(df['channels'].unique())}")
print(f"Windows: min={df['window'].min()}, max={df['window'].max()}, n_unique={df['window'].nunique()}")
print(f"BA converged all: {df['ba_converged'].all()}")
print()

n_sub = df["subject"].nunique()
n_method = df["method"].nunique()
n_ch = df["channels"].nunique()
n_win = df["window"].nunique()
expected = n_sub * n_method * n_ch * n_win
print(f"Expected ({n_sub}sub x {n_method}method x {n_ch}ch x {n_win}win): {expected}")
print(f"Actual rows: {len(df)}")
print(f"Completeness: {len(df)/expected*100:.1f}%")
print()

# Quick summary per channel config
import numpy as np
print("=" * 90)
print("Benchmark Multichannel: C_BA summary (mean over subjects, best window per method)")
print("=" * 90)
print(f"  {'Method':<8s} {'Config':<12s} {'Ch':>4s} {'Acc@1s':>7s} {'CBA@1s':>7s} {'Acc@5s':>7s} {'CBA@5s':>7s}")
print("-" * 90)

for method in sorted(df["method"].unique()):
    for ch_cfg in sorted(df["channel_config"].unique()):
        mdf = df[(df["method"] == method) & (df["channel_config"] == ch_cfg)]
        n_ch_val = mdf["channels"].iloc[0]
        # @1s
        at1 = mdf[abs(mdf["window"] - 1.0) < 0.01]
        # @5s
        at5 = mdf[abs(mdf["window"] - 5.0) < 0.01]
        acc1 = at1["accuracy"].mean() * 100 if len(at1) > 0 else float("nan")
        cba1 = at1["c_ba"].mean() if len(at1) > 0 else float("nan")
        acc5 = at5["accuracy"].mean() * 100 if len(at5) > 0 else float("nan")
        cba5 = at5["c_ba"].mean() if len(at5) > 0 else float("nan")
        print(f"  {method:<8s} {ch_cfg:<12s} {n_ch_val:>4d} {acc1:>7.1f} {cba1:>7.3f} {acc5:>7.1f} {cba5:>7.3f}")
    print()
