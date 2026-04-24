run_all.py — Master script: binary + multiclass experiments.
"""
import sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocess import load_and_preprocess, load_early_prediction, load_binary
from train import train_and_evaluate, save_metrics
from visualize import generate_all
from sklearn.model_selection import train_test_split

ROOT    = Path(__file__).parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
t_start = time.time()

print("="*60)
print("  Experiment A: Binary  (Dropout vs Graduate)")
print("  Experiment B: Multi   (Dropout / Enrolled / Graduate)")
print("="*60)

# 1. Preprocess
print("\n[1/5] Preprocessing...")
data_binary = load_binary(verbose=True)
data_multi  = load_and_preprocess(verbose=True)
data_early  = load_early_prediction(verbose=False)

# 2. Train binary
print("\n[2/5] Training BINARY...")
results_binary = train_and_evaluate(data_binary)
df_binary = save_metrics(results_binary, tag="binary")

# 3. Train multiclass
print("\n[3/5] Training MULTICLASS...")
results_multi = train_and_evaluate(data_multi)
df_multi = save_metrics(results_multi, tag="multiclass")

# 4. Train early
print("\n[4/5] Training EARLY PREDICTION (multiclass, 1st sem only)...")
results_early = train_and_evaluate(data_early)
save_metrics(results_early, tag="early")

# 5. Visualize
print("\n[5/5] Generating figures...")
df_raw = pd.read_csv(ROOT / "data" / "dataset.csv")
_, df_test_raw = train_test_split(df_raw, test_size=0.2, random_state=42, stratify=df_raw["Target"])
generate_all(results_full=results_multi, results_early=results_early,
             y_test=data_multi["y_test"], feature_names=data_multi["feature_names"],
             df_test=df_test_raw.reset_index(drop=True))

# Fig 13: Binary vs Multiclass
model_names = list(results_binary.keys())
x = np.arange(len(model_names))
w = 0.35
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (metric, title) in zip(axes, [("f1","F1 (macro)"),("accuracy","Accuracy"),("auc_roc","AUC-ROC")]):
    vb = [results_binary[m]["bootstrap_ci"][metric]["mean"] for m in model_names]
    vm = [results_multi[m]["bootstrap_ci"][metric]["mean"]  for m in model_names]
    eb = [[results_binary[m]["bootstrap_ci"][metric]["mean"]-results_binary[m]["bootstrap_ci"][metric]["lower"] for m in model_names],
          [results_binary[m]["bootstrap_ci"][metric]["upper"]-results_binary[m]["bootstrap_ci"][metric]["mean"] for m in model_names]]
    em = [[results_multi[m]["bootstrap_ci"][metric]["mean"]-results_multi[m]["bootstrap_ci"][metric]["lower"] for m in model_names],
          [results_multi[m]["bootstrap_ci"][metric]["upper"]-results_multi[m]["bootstrap_ci"][metric]["mean"] for m in model_names]]
    b1 = ax.bar(x-w/2, vb, w, label="Binary",     color="#2E75B6", alpha=0.9, edgecolor="white")
    b2 = ax.bar(x+w/2, vm, w, label="Multiclass", color="#ED7D31", alpha=0.9, edgecolor="white")
    ax.errorbar(x-w/2, vb, yerr=eb, fmt="none", color="black", capsize=3, lw=1)
    ax.errorbar(x+w/2, vm, yerr=em, fmt="none", color="black", capsize=3, lw=1)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.007,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.5, 1.05); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
fig.suptitle("Binary vs Multiclass — Performance Comparison (95% Bootstrap CI)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig13_binary_vs_multiclass.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig13 binary_vs_multiclass")

# Summary
elapsed = time.time() - t_start
print("\n"+"="*60)
print("  BINARY"); print("="*60)
print(df_binary[["model","accuracy","f1","auc_roc"]].to_string(index=False))
print("\n"+"="*60)
print("  MULTICLASS"); print("="*60)
print(df_multi[["model","accuracy","f1","auc_roc"]].to_string(index=False))
print(f"\n  Runtime: {elapsed:.0f}s\n✅ Done!")
