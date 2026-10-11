"""
From scores to decisions.

1. Alert lists under a fixed capacity: if advisers can contact the top c% of
   students ranked by P(dropout), what share of the contacted students later
   drop out (precision), and what share of all dropouts is reached (recall)?
   Computed at three prediction points with out-of-fold XGBoost predictions.

2. Group-specific thresholds: does equalising dropout recall across groups
   (scholarship status, age at enrolment) cost many extra false alerts?
   Thresholds are chosen by cross-fitting: on four folds of students, applied
   to the fifth, so no student's own label sets the threshold used for them.

Run after experiments.py:  python src/decision.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from analysis import C, FIG, INK2, RES, load_oof, save  # noqa: E402,F401
from data import load_dataset  # noqa: E402
from experiments import SEED  # noqa: E402

MODEL = "XGBoost"
STAGES = [("enrol", "At enrolment"), ("sem1", "After 1st semester"),
          ("multiclass", "After 2nd semester")]
CAPS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def topk_metrics(p, is_drop, cap):
    n = len(p)
    k = int(round(cap * n))
    flagged = np.zeros(n, bool)
    flagged[np.argsort(-p, kind="stable")[:k]] = True
    tp = (flagged & is_drop).sum()
    return tp / k, tp / is_drop.sum()


def stage_scores(setting, fees=True):
    """Out-of-fold P(dropout) from XGBoost (first CV repeat)."""
    y, oof = load_oof(setting)
    if fees:
        return y, oof[MODEL][:, 0]
    return y, np.load(RES / f"oof_xgb_{setting}_nofee.npy")


def alert_lists():
    rng = np.random.default_rng(SEED)
    out = {}
    curves = {}
    grid = np.linspace(0.005, 1, 200)
    for setting, lab in STAGES:
        for fees in (True, False):
            y, p = stage_scores(setting, fees)
            d = (y == 0)
            rows = []
            for cap in CAPS:
                prec, rec = topk_metrics(p, d, cap)
                boots = []
                for _ in range(1000):
                    i = rng.integers(0, len(p), len(p))
                    boots.append(topk_metrics(p[i], d[i], cap))
                b = np.array(boots)
                rows.append({"capacity": cap, "precision": prec, "recall": rec,
                             "lift": prec / d.mean(),
                             "precision_ci": np.percentile(b[:, 0], [2.5, 97.5]).tolist(),
                             "recall_ci": np.percentile(b[:, 1], [2.5, 97.5]).tolist()})
            key = setting + ("" if fees else "_nofee")
            out[key] = rows
            curves[key] = [topk_metrics(p, d, c)[1] for c in grid]
    base = float((load_oof("multiclass")[0] == 0).mean())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for j, (s, lab) in enumerate(STAGES):
        axes[0].plot(grid * 100, curves[s], color=C[j], lw=1.8, label=lab)
        axes[0].plot(grid * 100, curves[s + "_nofee"], color=C[j], lw=1.2, ls=":")
        prec = [r["precision"] for r in out[s]]
        lo = [r["precision"] - r["precision_ci"][0] for r in out[s]]
        hi = [r["precision_ci"][1] - r["precision"] for r in out[s]]
        axes[1].errorbar([c * 100 for c in CAPS], prec, yerr=[lo, hi], fmt="-o",
                         ms=3.5, lw=1.6, color=C[j], elinewidth=0.9, label=lab)
        axes[1].plot([c * 100 for c in CAPS], [r["precision"] for r in out[s + "_nofee"]],
                     ls=":", lw=1.2, color=C[j])
    axes[0].plot(grid * 100, np.minimum(1, grid / base), color=INK2, lw=0.8)
    axes[0].text(50, 1.035, "perfect ranking", color=INK2, fontsize=7.5)
    axes[0].plot([0, 100], [0, 1], ls="--", color=INK2, lw=0.8)
    axes[0].text(62, 0.52, "random list", color=INK2, fontsize=7.5, rotation=24)
    axes[0].set_xlabel("Students contacted (% of cohort, highest risk first)")
    axes[0].set_ylabel("Share of all dropouts reached")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1.1)
    axes[1].axhline(base, ls="--", color=INK2, lw=0.8)
    axes[1].text(31, base + 0.015, f"base rate {base:.2f}", color=INK2, fontsize=7.5)
    axes[1].set_xlabel("Students contacted (% of cohort)")
    axes[1].set_ylabel("Share of contacted who drop out")
    axes[1].set_ylim(0, 1.02)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C[j], lw=1.8) for j in range(3)]
    h += [Line2D([], [], color=INK2, lw=1.8), Line2D([], [], color=INK2, lw=1.2, ls=":")]
    fig.legend(h, [lab for _, lab in STAGES] + ["all features", "without fee variables"],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.13),
               handlelength=1.6, columnspacing=1.0)
    save(fig, "fig_alert_lists")
    return out, base


def group_thresholds(df, attr, groups, cap=0.20, n_splits=5):
    """Single global threshold vs per-group thresholds that match the global
    recall inside each group. Cross-fitted over student folds."""
    y, oof = load_oof("multiclass")
    p = oof[MODEL][:, 0]
    d = (y == 0)
    g = np.full(len(y), -1)
    for gi, mask in enumerate(groups.values()):
        g[mask] = gi
    strat = d.astype(int) * 10 + g
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    flag_single = np.zeros(len(y), bool)
    flag_group = np.zeros(len(y), bool)
    for tr, te in skf.split(p, strat):
        t_single = np.quantile(p[tr], 1 - cap)
        target = (p[tr][d[tr]] >= t_single).mean()  # global recall on training part
        flag_single[te] = p[te] >= t_single
        for gi in range(len(groups)):
            m_tr = (g[tr] == gi) & d[tr]
            # threshold giving the same recall within this group's dropouts
            t_g = np.quantile(p[tr][m_tr], 1 - target)
            m_te = g[te] == gi
            flag_group[te[m_te]] = p[te[m_te]] >= t_g
    rows = []
    for name, flag in [("single", flag_single), ("per-group", flag_group)]:
        for gi, gname in enumerate(groups):
            m = g == gi
            rows.append({"policy": name, "group": gname, "n": int(m.sum()),
                         "recall": float((flag & d & m).sum() / (d & m).sum()),
                         "fpr": float((flag & ~d & m).sum() / (~d & m).sum()),
                         "alerts": int((flag & m).sum())})
        rows.append({"policy": name, "group": "All", "n": len(y),
                     "recall": float((flag & d).sum() / d.sum()),
                     "fpr": float((flag & ~d).sum() / (~d).sum()),
                     "alerts": int(flag.sum()),
                     "precision": float((flag & d).sum() / flag.sum())})
    return {"attribute": attr, "capacity": cap, "rows": rows}


def main():
    df = load_dataset()
    al, base = alert_lists()
    age = df["Age at enrollment"].to_numpy()
    thr = [
        group_thresholds(df, "Scholarship", {
            "No": df["Scholarship holder"].to_numpy() == 0,
            "Yes": df["Scholarship holder"].to_numpy() == 1}),
        group_thresholds(df, "Age at enrolment", {
            "<=20": age <= 20, "21-25": (age > 20) & (age <= 25), ">25": age > 25}),
    ]
    # who is at the top of the list? (first-semester model)
    y, p = stage_scores("sem1")
    order = np.argsort(-p, kind="stable")
    k = int(round(0.10 * len(p)))
    zero = df["Curricular units 1st sem (approved)"].to_numpy() == 0
    nofee = df["Tuition fees up to date"].to_numpy() == 0
    profile = {"top10_zero_approved": float(zero[order[:k]].mean()),
               "next10_zero_approved": float(zero[order[k:2 * k]].mean()),
               "all_zero_approved": float(zero.mean()),
               "top10_fees_not_up_to_date": float(nofee[order[:k]].mean())}
    json.dump({"alert_lists": al, "base_rate": base, "thresholds": thr,
               "top_profile": profile},
              open(RES / "decision.json", "w"), indent=1)
    for s, rows in al.items():
        if s.endswith("_nofee"):
            continue
        print(s, [(r["capacity"], round(r["precision"], 3), round(r["recall"], 3))
                  for r in rows])
    for t in thr:
        print(t["attribute"])
        for r in t["rows"]:
            print("  ", r)


if __name__ == "__main__":
    main()
