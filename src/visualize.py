Generate all publication-quality figures for the paper.
Run AFTER train.py — reads results dict passed from run_all.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]
MODEL_COLORS = {
    "Logistic Regression": "#3498DB",
    "Decision Tree":       "#E67E22",
    "Random Forest":       "#2ECC71",
    "SVM":                 "#9B59B6",
    "Gradient Boosting":   "#1ABC9C",
    "Neural Network":      "#E74C3C",
}
PALETTE = {"Dropout": "#E74C3C", "Enrolled": "#F39C12", "Graduate": "#2ECC71"}


# ─────────────────────────────────────────────────────────────────────────────
def fig_model_comparison(results):
    """Bar chart: Accuracy / Precision / Recall / F1 / AUC per model."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels = ["Accuracy", "Precision", "Recall", "F1 (macro)", "AUC-ROC"]

    model_names = list(results.keys())
    x = np.arange(len(model_names))
    n_metrics = len(metrics_to_plot)
    width = 0.14

    fig, ax = plt.subplots(figsize=(14, 6))
    metric_colors = ["#3498DB", "#2ECC71", "#E74C3C", "#9B59B6", "#F39C12"]

    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, labels, metric_colors)):
        vals  = [results[m]["bootstrap_ci"][metric]["mean"]  for m in model_names]
        lows  = [results[m]["bootstrap_ci"][metric]["mean"] - results[m]["bootstrap_ci"][metric]["lower"] for m in model_names]
        highs = [results[m]["bootstrap_ci"][metric]["upper"] - results[m]["bootstrap_ci"][metric]["mean"] for m in model_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85, edgecolor="white")
        ax.errorbar(x + offset, vals, yerr=[lows, highs],
                    fmt="none", color="black", capsize=2, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison (with 95% Bootstrap CI)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig07 model_comparison")


# ─────────────────────────────────────────────────────────────────────────────
def fig_roc_curves(results, y_test):
    """ROC curves (OvR) for all models — matches proposal Fig 3."""
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, cls_idx, cls_name in zip(axes, [0, 1, 2], CLASS_NAMES):
        for name, r in results.items():
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], r["y_prob"][:, cls_idx])
            auc_val = r["bootstrap_ci"]["auc_roc"]["mean"]
            ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=1.8,
                    label=f"{name} (AUC={auc_val:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_title(f"ROC — {cls_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("ROC Curves — One-vs-Rest (Multiclass)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig08_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig08 roc_curves")


# ─────────────────────────────────────────────────────────────────────────────
def fig_confusion_matrices(results):
    """3×2 grid of confusion matrices."""
    model_names = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ax, name in zip(axes.flat, model_names):
        cm = np.array(results[name]["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_title(name, fontsize=11, fontweight="bold")

        for i in range(3):
            for j in range(3):
                val = cm[i, j]
                pct = cm_norm[i, j]
                color = "white" if pct > 0.5 else "black"
                ax.text(j, i, f"{val}\n({pct:.1%})",
                        ha="center", va="center", fontsize=9,
                        color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Confusion Matrices — All Models (Row-Normalized)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig09_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig09 confusion_matrices")


# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_importance(results, feature_names):
    """Feature importance from Random Forest & Gradient Boosting."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    top_n = 20

    for ax, name in zip(axes, ["Random Forest", "Gradient Boosting"]):
        model = results[name]["model"]
        importances = model.feature_importances_
        idx = np.argsort(importances)[-top_n:]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))

        ax.barh(range(top_n), importances[idx], color=colors, edgecolor="white")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
        ax.set_xlabel("Feature Importance", fontsize=10)
        ax.set_title(f"Top {top_n} Features — {name}", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig10 feature_importance")


# ─────────────────────────────────────────────────────────────────────────────
def fig_early_vs_full(results_full, results_early):
    """Compare full model vs early prediction (no 2nd sem data)."""
    model_names = list(results_full.keys())
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    full_f1  = [results_full[m]["bootstrap_ci"]["f1"]["mean"]  for m in model_names]
    early_f1 = [results_early[m]["bootstrap_ci"]["f1"]["mean"] for m in model_names]

    bars1 = ax.bar(x - width/2, full_f1,  width, label="Full model (both semesters)",
                   color="#2E75B6", edgecolor="white", alpha=0.9)
    bars2 = ax.bar(x + width/2, early_f1, width, label="Early prediction (1st sem only)",
                   color="#ED7D31", edgecolor="white", alpha=0.9)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("F1 Score (macro)", fontsize=11)
    ax.set_title("Full Model vs Early Prediction — F1 Score Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_early_vs_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig11 early_vs_full")


# ─────────────────────────────────────────────────────────────────────────────
def fig_fairness(results, y_test, df_test):
    """F1 score breakdown by Gender and Scholarship status — best model."""
    best_model = max(results, key=lambda m: results[m]["bootstrap_ci"]["f1"]["mean"])
    y_pred = results[best_model]["y_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    groups = [
        ("Gender", {0: "Female", 1: "Male"}),
        ("Scholarship holder", {0: "No Scholarship", 1: "Scholarship"}),
    ]

    from sklearn.metrics import f1_score

    for ax, (col, mapping) in zip(axes, groups):
        group_vals = df_test[col].values
        group_labels = []
        group_f1s = []
        for gval, glabel in mapping.items():
            mask = group_vals == gval
            if mask.sum() < 10:
                continue
            gf1 = f1_score(y_test[mask], y_pred[mask],
                           average="macro", zero_division=0)
            group_labels.append(glabel)
            group_f1s.append(gf1)

        colors = ["#3498DB", "#E74C3C"]
        bars = ax.bar(group_labels, group_f1s, color=colors[:len(group_labels)],
                      edgecolor="white", width=0.4)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold")

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 Score (macro)", fontsize=11)
        ax.set_title(f"Fairness — {col}", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle(f"Fairness Analysis — {best_model} (Best Model)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_fairness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig12 fairness")


# ─────────────────────────────────────────────────────────────────────────────
def generate_all(results_full, results_early, y_test, feature_names, df_test):
    print("\n── Generating figures ──────────────────────────────")
    fig_model_comparison(results_full)
    fig_roc_curves(results_full, y_test)
    fig_confusion_matrices(results_full)
    fig_feature_importance(results_full, feature_names)
    fig_early_vs_full(results_full, results_early)
    fig_fairness(results_full, y_test, df_test)
    print(f"\n✅ All figures saved → {FIG_DIR}")


def fig_binary_vs_multiclass(results_binary, results_multi):
    """Key comparison figure: binary vs multiclass F1 and AUC per model."""
    model_names = list(results_binary.keys())
    x = np.arange(len(model_names))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(axes, ["f1", "auc_roc"], ["F1 Score", "AUC-ROC"]):
        bin_vals   = [results_binary[m]["bootstrap_ci"][metric]["mean"] for m in model_names]
        multi_vals = [results_multi[m]["bootstrap_ci"][metric]["mean"]  for m in model_names]
        bin_err    = [[results_binary[m]["bootstrap_ci"][metric]["mean"] - results_binary[m]["bootstrap_ci"][metric]["lower"] for m in model_names],
                      [results_binary[m]["bootstrap_ci"][metric]["upper"] - results_binary[m]["bootstrap_ci"][metric]["mean"] for m in model_names]]
        multi_err  = [[results_multi[m]["bootstrap_ci"][metric]["mean"] - results_multi[m]["bootstrap_ci"][metric]["lower"] for m in model_names],
                      [results_multi[m]["bootstrap_ci"][metric]["upper"] - results_multi[m]["bootstrap_ci"][metric]["mean"] for m in model_names]]

        b1 = ax.bar(x - width/2, bin_vals,   width, label="Binary (Dropout vs Graduate)",
                    color="#2E75B6", alpha=0.9, edgecolor="white")
        b2 = ax.bar(x + width/2, multi_vals, width, label="Multiclass (3 classes)",
                    color="#ED7D31", alpha=0.9, edgecolor="white")
        ax.errorbar(x - width/2, bin_vals,   yerr=bin_err,   fmt="none", color="black", capsize=3, lw=0.9)
        ax.errorbar(x + width/2, multi_vals, yerr=multi_err, fmt="none", color="black", capsize=3, lw=0.9)

        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0.55, 1.05)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title}: Binary vs Multiclass", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle("Binary vs Multiclass Classification — Performance Comparison (95% Bootstrap CI)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig13_binary_vs_multiclass.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig13 binary_vs_multiclass")


def fig_binary_vs_multiclass(results_binary, results_multiclass):
    """Key comparison figure: binary vs multiclass for each model, F1 and AUC."""
    model_names = list(results_multiclass.keys())
    x = np.arange(len(model_names))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    combos = [
        ("f1",      "F1 Score"),
        ("auc_roc", "AUC-ROC"),
    ]

    configs = [
        ("Multiclass (3-class)", results_multiclass, "#2E75B6", -1.5),
        ("Binary (Dropout vs Graduate)", results_binary, "#ED7D31", -0.5),
    ]

    for ax, (metric_key, metric_label) in zip(axes, combos):
        for label, results, color, pos_mult in configs:
            vals  = [results[m]["bootstrap_ci"][metric_key]["mean"]  for m in model_names]
            lows  = [results[m]["bootstrap_ci"][metric_key]["mean"] - results[m]["bootstrap_ci"][metric_key]["lower"] for m in model_names]
            highs = [results[m]["bootstrap_ci"][metric_key]["upper"] - results[m]["bootstrap_ci"][metric_key]["mean"] for m in model_names]
            offset = pos_mult * width + width
            bars = ax.bar(x + offset, vals, width, label=label,
                         color=color, alpha=0.85, edgecolor="white")
            ax.errorbar(x + offset, vals, yerr=[lows, highs],
                       fmt="none", color="black", capsize=2, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0.5, 1.05)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"{metric_label}: Binary vs Multiclass", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle("Experiment Comparison: Binary (Dropout/Graduate) vs Multiclass (3-class)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig13_binary_vs_multiclass.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig13 binary_vs_multiclass")
