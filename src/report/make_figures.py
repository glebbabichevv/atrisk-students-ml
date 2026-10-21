"""
Figures that put the two universities side by side (paper/figures/fig_universities.pdf).

  (a) Dropout AUC at each prediction point, best model in repeated CV.
      UCI: three-class model, P(dropout) vs rest. UPV: binary model.
  (b) Change in AUC when the model meets students unlike its training data,
      logistic regression vs XGBoost, after the first semester.

Skipped with a message when the UPV results are missing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import UCI_RESULTS, UPV_RESULTS  # noqa: E402
from common.plotting import C, INK2, plt, save  # noqa: E402

STAGES = ["Enrolment", "1st semester", "End of year 1"]
UCI_STAGE = {"Enrolment": "enrol", "1st semester": "sem1", "End of year 1": "multiclass"}
UPV_STAGE = {"Enrolment": "enrol", "1st semester": "sem1", "End of year 1": "year"}
UPV_LMS = {"1st semester": "sem1_lms", "End of year 1": "year_lms"}
PAIR = ["Logistic Regression", "XGBoost"]


def uci_best(summary: pd.DataFrame, setting: str) -> float:
    return float(summary.loc[summary.setting == setting, "auc_dropout"].max())


def upv_best(upv: dict, stage: str) -> float:
    return max(v["auc"] for v in upv["cv"][stage]["summary"].values())


def shifts(uci_tr: dict, upv: dict) -> list[tuple[str, dict]]:
    """AUC under shift minus AUC in repeated CV, per model."""
    out = []
    pooled = {m: uci_tr["sem1"][m]["pooled"] for m in PAIR}
    out.append(("UCI\nunseen\nprogramme",
                {m: pooled[m]["lopo"]["auc_dropout"] - pooled[m]["within"]["auc_dropout"]
                 for m in PAIR}))
    g = upv["grouped_degree"]["sem1"]
    out.append(("UPV\nunseen\ndegree", {m: g[m]["auc"] - g[m]["cv_auc"] for m in PAIR}))
    cv = upv["cv"]["sem1"]["summary"]
    for split, lab in [("2018_to_2021", "UPV\n2018 →\n2021"),
                       ("2018_2021_to_2022", "UPV\n2018, 2021\n→ 2022")]:
        t = upv["temporal"][split]["sem1"]
        out.append((lab, {m: t[m]["auc"] - cv[m]["auc"] for m in PAIR}))
    return out


def main():
    path = UPV_RESULTS / "upv_results.json"
    if not path.exists():
        print("No UPV results (results/upv/upv_results.json): cross-university figure skipped.")
        return
    upv = json.loads(path.read_text())
    summary = pd.read_csv(UCI_RESULTS / "cv_summary.csv")
    uci_tr = json.loads((UCI_RESULTS / "transfer.json").read_text())

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})

    x = np.arange(len(STAGES))
    uci = [uci_best(summary, UCI_STAGE[s]) for s in STAGES]
    upv_v = [upv_best(upv, UPV_STAGE[s]) for s in STAGES]
    a.plot(x, uci, "-o", color=C[0], label="UCI, Portugal")
    a.plot(x, upv_v, "-o", color=C[1], label="UPV, Spain")
    lx = [STAGES.index(s) for s in UPV_LMS]
    a.plot(lx, [upv_best(upv, UPV_LMS[s]) for s in UPV_LMS], "o", mfc="white",
           color=C[1], label="UPV + LMS activity")
    for xi, v in zip(x, uci):
        a.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 6),
                   ha="center", fontsize=7.5, color=C[0])
    for xi, v in zip(x, upv_v):
        a.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, -12),
                   ha="center", fontsize=7.5, color=C[1])
    a.set_xticks(x, STAGES)
    a.set_ylabel("Dropout AUC (best model, CV)")
    a.set_ylim(0.65, 0.97)
    a.set_title("(a) Information over the first year", loc="left")
    a.legend(loc="lower right")

    sh = shifts(uci_tr, upv)
    xb = np.arange(len(sh))
    w = 0.36
    for i, (m, col) in enumerate(zip(PAIR, [C[2], C[3]])):
        vals = [d[m] for _, d in sh]
        b.bar(xb + (i - 0.5) * w, vals, w, color=col, label=m)
    b.axhline(0, color=INK2, lw=0.8)
    b.set_xticks(xb, [lab for lab, _ in sh], fontsize=7.5)
    b.set_ylabel("AUC change vs. CV, 1st semester")
    b.set_title("(b) Unseen programmes and later cohorts", loc="left")
    lo = min(min(d.values()) for _, d in sh)
    b.set_ylim(lo * 1.35, 0.005)          # room below the bars for the legend
    b.legend(loc="lower center", ncol=2, fontsize=7.5)
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_universities")
    print("fig_universities.pdf")
    for lab, d in sh:
        print(f"  {lab.replace(chr(10), ' '):32s} " +
              "  ".join(f"{m}: {v:+.3f}" for m, v in d.items()))


if __name__ == "__main__":
    main()
