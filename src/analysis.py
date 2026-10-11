"""
Secondary analyses and all paper figures (vector PDF).

Needs the outputs of src/experiments.py. Run:  python src/analysis.py

  * calibration of the dropout probability (out-of-fold, first CV repeat)
  * subgroup analysis (gender, scholarship, age at enrolment) with bootstrap CIs
  * feature selection inside cross-validation (no test-set selection)
  * SHAP values for the XGBoost multiclass model (raw, unscaled features)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import (RepeatedStratifiedKFold,  # noqa: E402
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from data import CLASS_NAMES, load_dataset, setting_xy  # noqa: E402
from experiments import SEED, get_models  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (dataviz reference instance), fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
     "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SETTING_LABEL = {"binary": "Binary (D vs G)", "multiclass": "3-class, both semesters",
                 "sem1": "3-class, 1st semester only", "enrol": "3-class, enrolment only"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def load_oof(setting):
    z = np.load(RES / f"oof_{setting}.npz")
    y = z["y"]
    return y, {k.replace("_", " "): z[k] for k in z.files if k != "y"}


def boot_ci(fn, idx_groups, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.choice(idx_groups, size=len(idx_groups), replace=True)
        try:
            vals.append(fn(idx))
        except ValueError:
            continue
    return np.percentile(vals, [2.5, 97.5])


# ── Figure 1: model x setting dot plot ───────────────────────────────────────
def fig_overview(summary):
    models = (summary[summary.setting == "multiclass"]
              .sort_values("f1")["model"].tolist())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, metric, title in [(axes[0], "f1", "Macro F1"),
                              (axes[1], "auc", "Macro AUC (one-vs-rest)")]:
        for j, (s, lab) in enumerate(SETTING_LABEL.items()):
            d = summary[summary.setting == s].set_index("model").loc[models]
            ypos = np.arange(len(models)) + (j - 1.5) * 0.17
            ax.errorbar(d[metric], ypos, xerr=d[metric + "_sd"], fmt="o", ms=4,
                        color=C[j], ecolor=C[j], elinewidth=1, capsize=0,
                        label=lab)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(models)), models)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
               handletextpad=0.2, columnspacing=1.2)
    save(fig, "fig_overview")


# ── Figure 2: per-class F1 by information stage ──────────────────────────────
def fig_per_class(summary, model):
    stages = ["enrol", "sem1", "multiclass"]
    labels = ["Enrolment\nonly", "+ 1st\nsemester", "+ 2nd\nsemester"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    w = 0.26
    for j, cls in enumerate(CLASS_NAMES):
        vals = [summary[(summary.setting == s) & (summary.model == model)]
                [f"f1_{cls.lower()}"].item() for s in stages]
        sds = [summary[(summary.setting == s) & (summary.model == model)]
               [f"f1_{cls.lower()}_sd"].item() for s in stages]
        x = np.arange(3) + (j - 1) * (w + 0.02)
        ax.bar(x, vals, w, color=C[j], label=cls, yerr=sds,
               error_kw=dict(ecolor=INK2, lw=0.8))
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("F1 (class-wise)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.12))
    save(fig, "fig_per_class")


# ── Figure 3: confusion matrix (row-normalised, OOF) ─────────────────────────
def fig_confusion(model):
    y, oof = load_oof("multiclass")
    pred = oof[model].argmax(1)
    cm = confusion_matrix(y, pred)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n(n={cm[i, j]})", ha="center",
                    va="center", fontsize=7.5,
                    color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig_confusion")
    return cm


# ── Figure 4: calibration of P(dropout) ──────────────────────────────────────
def fig_calibration(models):
    y, oof = load_oof("multiclass")
    is_drop = (y == 0).astype(int)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=0.8)
    out = {}
    for j, m in enumerate(models):
        p = oof[m][:, 0]
        frac, mean = calibration_curve(is_drop, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "-o", ms=3.5, lw=1.6, color=C[j], label=m)
        bins = np.quantile(p, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
        ece = sum(abs(p[idx == b].mean() - is_drop[idx == b].mean()) * (idx == b).mean()
                  for b in range(10) if (idx == b).any())
        out[m] = {"ece": float(ece), "brier": float(np.mean((p - is_drop) ** 2))}
    ax.set_xlabel("Predicted P(dropout)")
    ax.set_ylabel("Observed dropout rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    save(fig, "fig_calibration")
    return out


# ── Subgroup analysis ────────────────────────────────────────────────────────
def subgroup_table(df, model):
    y, oof = load_oof("multiclass")
    prob = oof[model]
    pred = prob.argmax(1)
    age = df["Age at enrollment"].to_numpy()
    groups = {
        "Gender": {"Female": df["Gender"].to_numpy() == 0,
                   "Male": df["Gender"].to_numpy() == 1},
        "Scholarship": {"No": df["Scholarship holder"].to_numpy() == 0,
                        "Yes": df["Scholarship holder"].to_numpy() == 1},
        "Age at enrolment": {"<=20": age <= 20, "21-25": (age > 20) & (age <= 25),
                             ">25": age > 25},
    }
    rows = []
    for attr, gs in groups.items():
        for g, mask in gs.items():
            idx = np.where(mask)[0]
            dr = lambda i: ((pred[i] == 0) & (y[i] == 0)).sum() / (y[i] == 0).sum()  # noqa: E731
            fpr = lambda i: ((pred[i] == 0) & (y[i] != 0)).sum() / (y[i] != 0).sum()  # noqa: E731
            mf1 = lambda i: f1_score(y[i], pred[i], average="macro")  # noqa: E731
            rows.append({
                "attribute": attr, "group": g, "n": len(idx),
                "dropout_rate": (y[idx] == 0).mean(),
                "tpr": dr(idx), "tpr_ci": boot_ci(dr, idx).tolist(),
                "fpr": fpr(idx), "fpr_ci": boot_ci(fpr, idx).tolist(),
                "macro_f1": mf1(idx), "macro_f1_ci": boot_ci(mf1, idx).tolist(),
            })
    return pd.DataFrame(rows)


def fig_subgroups(tab):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    labels = [f"{a}: {g}".replace("<=", "\u2264 ").replace(">", "> ")
              .replace("21-25", "21\u201325") for a, g in zip(tab.attribute, tab.group)]
    ypos = np.arange(len(tab))[::-1]
    for ax, col, title, color in [(axes[0], "tpr", "Dropout recall (TPR)", C[0]),
                                  (axes[1], "fpr", "False dropout alerts (FPR)", C[1])]:
        lo = tab[col] - tab[col + "_ci"].map(lambda v: v[0])
        hi = tab[col + "_ci"].map(lambda v: v[1]) - tab[col]
        ax.errorbar(tab[col], ypos, xerr=[lo, hi], fmt="o", ms=4, color=color,
                    elinewidth=1)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, labels)
    save(fig, "fig_subgroups")


# ── Feature selection inside CV ──────────────────────────────────────────────
def feature_selection(df, model_name):
    X, y, feats = setting_xy(df, "multiclass")
    from sklearn.pipeline import Pipeline
    base = get_models(3, feats)[model_name]
    if isinstance(base, Pipeline):  # column indices would shift after selection
        model_name, base = "LightGBM", get_models(3, feats)["LightGBM"]
    print("  feature-selection model:", model_name)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rows = []
    for k in [5, 10, 15, 20, 25, 30, X.shape[1]]:
        pipe = make_pipeline(SelectKBest(f_classif, k=k), clone(base))
        s = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        rows.append({"k": k, "f1": s.mean(), "f1_sd": s.std(ddof=1)})
        print(f"  k={k}: F1={s.mean():.4f} ± {s.std(ddof=1):.4f}")
    sel = SelectKBest(f_classif, k="all").fit(X, y)
    fscores = pd.Series(sel.scores_, index=feats).sort_values(ascending=False)
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.errorbar(tab.k, tab.f1, yerr=tab.f1_sd, fmt="-o", ms=4, color=C[0],
                elinewidth=1, lw=1.6)
    ax.set_xlabel("Number of features kept (ANOVA F, chosen inside each fold)")
    ax.set_ylabel("Macro F1")
    save(fig, "fig_feature_selection")
    return tab, fscores


# ── SHAP ─────────────────────────────────────────────────────────────────────
def shap_analysis(df):
    import shap
    from xgboost import XGBClassifier
    X, y, feats = setting_xy(df, "multiclass")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=SEED)
    clf = get_models(3, feats)["XGBoost"]
    assert isinstance(clf, XGBClassifier)
    clf.fit(Xtr, ytr)
    f1 = f1_score(yte, clf.predict(Xte), average="macro")
    ex = shap.TreeExplainer(clf)
    sv = ex(pd.DataFrame(Xte, columns=feats))  # (n, features, classes), log-odds
    vals = sv.values
    imp = pd.DataFrame({c: np.abs(vals[:, :, i]).mean(0)
                        for i, c in enumerate(CLASS_NAMES)}, index=feats)
    imp["mean"] = imp.mean(1)
    imp = imp.sort_values("mean", ascending=False)

    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    left = np.zeros(len(top))
    for j, c in enumerate(CLASS_NAMES):
        ax.barh(top.index, top[c] / 3, left=left, color=C[j], label=c,
                edgecolor="white", linewidth=0.8, height=0.7)
        left += top[c].to_numpy() / 3
    ax.set_xlabel("Mean |SHAP| (log-odds), averaged over classes")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fig_shap_global")

    plt.figure()
    exp = shap.Explanation(values=vals[:, :, 0], data=Xte, feature_names=feats)
    shap.plots.beeswarm(exp, max_display=12, show=False, plot_size=(5.2, 3.6),
                        color_bar_label="Feature value")
    f = plt.gcf()
    f.axes[0].set_xlabel("SHAP value for the Dropout class (log-odds)", fontsize=9)
    for ax_ in f.axes:
        ax_.tick_params(labelsize=8)
        ax_.yaxis.label.set_size(9)
    for t in f.axes[0].get_yticklabels():
        t.set_fontsize(8)
    f.savefig(FIG / "fig_shap_dropout.pdf", bbox_inches="tight")
    plt.close(f)

    # direction check for text: correlation between feature value and SHAP (dropout)
    direction = {feats[j]: float(np.corrcoef(Xte[:, j], vals[:, j, 0])[0, 1])
                 if Xte[:, j].std() > 0 else 0.0 for j in range(len(feats))}
    return {"xgb_holdout_f1": float(f1), "importance": imp.round(4).to_dict(),
            "direction_dropout": direction}


def main():
    df = load_dataset()
    summary = pd.read_csv(RES / "cv_summary.csv")
    multi = summary[summary.setting == "multiclass"].sort_values("f1", ascending=False)
    best = multi.iloc[0]["model"]
    top3 = multi["model"].head(3).tolist()
    print("best multiclass model:", best)

    fig_overview(summary)
    fig_per_class(summary, best)
    cm = fig_confusion(best)
    cal = fig_calibration(top3 + ["Logistic Regression"]
                          if "Logistic Regression" not in top3 else top3)
    sub = subgroup_table(df, best)
    fig_subgroups(sub)
    print("feature selection (nested):")
    fs_tab, fscores = feature_selection(df, best)
    shp = shap_analysis(df)

    out = {"best_model": best, "confusion": cm.tolist(), "calibration": cal,
           "subgroups": sub.to_dict(orient="records"),
           "feature_selection": fs_tab.to_dict(orient="records"),
           "anova_f": fscores.round(1).to_dict(), "shap": shp}
    with open(RES / "analysis.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ["best_model", "confusion", "calibration"]},
                     indent=1))
    print(sub.round(3).to_string())


if __name__ == "__main__":
    main()
