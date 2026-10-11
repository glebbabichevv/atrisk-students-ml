"""
Write every number used in the paper to paper/generated/, so the text and the
results cannot drift apart again.

  paper/generated/numbers.tex   \\newcommand macros used in the prose
  paper/generated/tab_*.tex     tables

Run after experiments.py and analysis.py:  python src/make_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
GEN = ROOT / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)

S = pd.read_csv(RES / "cv_summary.csv")
T = pd.read_csv(RES / "cv_tests.csv")
A = json.load(open(RES / "analysis.json"))
MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM (RBF)",
          "Gradient Boosting", "XGBoost", "LightGBM", "MLP"]


def f3(x):
    return f"{x:.3f}"


def g(setting, model, col):
    return S[(S.setting == setting) & (S.model == model)][col].item()


def best(setting, col):
    d = S[S.setting == setting].sort_values(col, ascending=False)
    return d.iloc[0]["model"], d.iloc[0][col]


def macro_name(s):
    return "".join(ch for ch in s if ch.isalpha())


macros = {}


def m(name, value):
    macros[name] = value


# ── results tables ───────────────────────────────────────────────────────────
def results_table(setting, cols, caption, label, extra_note=""):
    d = S[S.setting == setting].set_index("model").loc[MODELS]
    best_vals = {c: d[c].max() for c in cols}
    tests = T[T.setting == setting]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{3.5pt}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Model & " + " & ".join(c[1] for c in cols_meta(cols)) + r" \\",
             r"\midrule"]
    for model in MODELS:
        cells = []
        for c in cols:
            v, sd = d.loc[model, c], d.loc[model, c + "_sd"]
            txt = (f"{v:.3f}" if c.startswith("f1_")
                   else f"{v:.3f} $\\pm$ {sd:.3f}")
            if np.isclose(v, best_vals[c]):
                txt = r"\textbf{" + txt + "}"
            elif c in ("f1", "auc"):
                row = tests[(tests.metric == c) & (tests.other == model)]
                if len(row) and row.p.item() >= 0.05:
                    txt += r"$^{\dagger}$"
            cells.append(txt)
        lines.append(model + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\footnotesize Mean $\pm$ SD over 25 test folds "
              r"(5-fold CV repeated 5 times)" + ("; class-wise F1 columns show the mean only" if any(c.startswith("f1_") for c in cols) else "") + r". Bold: best value. $^{\dagger}$: not "
              r"significantly different from the best model on that metric "
              r"(corrected resampled $t$-test, $p\ge 0.05$)." + extra_note,
              r"\end{table}"]
    (GEN / f"tab_{setting}.tex").write_text("\n".join(lines) + "\n")


META = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
        "f1": "Macro F1", "auc": "AUC", "f1_dropout": "F1 Dropout",
        "f1_enrolled": "F1 Enrolled", "f1_graduate": "F1 Graduate",
        "auc_dropout": "AUC Dropout", "brier_dropout": "Brier"}


def cols_meta(cols):
    return [(c, META[c]) for c in cols]


results_table("binary", ["accuracy", "f1", "auc"],
              "Binary setting (Dropout vs.\\ Graduate, $n=3{,}630$).", "tab:binary")
results_table("multiclass", ["accuracy", "f1", "auc", "f1_dropout", "f1_enrolled",
                             "f1_graduate"],
              "Three-class setting with first- and second-semester records "
              "($n=4{,}424$). AUC is the macro one-vs-rest average.", "tab:multi")


# ── stage table: best model per stage ────────────────────────────────────────
def stage_table():
    rows = []
    for s, lab in [("enrol", "Enrolment only"), ("sem1", "+ 1st semester"),
                   ("multiclass", "+ 2nd semester")]:
        bm, _ = best(s, "f1")
        rows.append((lab, bm, g(s, bm, "f1"), g(s, bm, "f1_sd"),
                     g(s, bm, "f1_dropout"), g(s, bm, "f1_enrolled"),
                     g(s, bm, "f1_graduate"), g(s, bm, "auc_dropout")))
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}", r"\footnotesize",
             r"\caption{Three-class prediction at three points in time. "
             r"Each row reports the model with the highest macro F1 at that stage.}",
             r"\label{tab:stages}", r"\begin{tabular}{llccccc}", r"\toprule",
             r"Information used & Best model & Macro F1 & F1 Dropout & F1 Enrolled "
             r"& F1 Graduate & AUC Dropout \\", r"\midrule"]
    for lab, bm, f, sd, fd, fe, fg, ad in rows:
        lines.append(f"{lab} & {bm} & {f:.3f} $\\pm$ {sd:.3f} & {fd:.3f} & "
                     f"{fe:.3f} & {fg:.3f} & {ad:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\footnotesize AUC Dropout: the predicted probability "
              r"of the Dropout class scored against Dropout vs.\ all other students.",
              r"\end{table}"]
    (GEN / "tab_stages.tex").write_text("\n".join(lines) + "\n")
    return rows


stage_rows = stage_table()


# ── subgroup table ───────────────────────────────────────────────────────────
def subgroup_table():
    sub = pd.DataFrame(A["subgroups"])
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{3.5pt}", r"\footnotesize",
             r"\caption{Subgroup results for the best three-class model "
             r"(out-of-fold predictions for all 4{,}424 students, first CV repeat). "
             r"Brackets: 95\% bootstrap intervals.}",
             r"\label{tab:subgroups}", r"\begin{tabular}{llrcccc}", r"\toprule",
             r"Attribute & Group & $n$ & Dropout rate & Dropout recall & "
             r"False alert rate & Macro F1 \\", r"\midrule"]
    prev = None
    for _, r in sub.iterrows():
        attr = r.attribute if r.attribute != prev else ""
        prev = r.attribute
        grp = (r.group.replace("<=", r"$\le$").replace(">", "$>$")
               .replace("21-25", "21--25"))
        ci = lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]"  # noqa: E731
        lines.append(f"{attr} & {grp} & {r.n:,} & {r.dropout_rate:.2f} & "
                     f"{r.tpr:.2f} {ci(r.tpr_ci)} & {r.fpr:.2f} {ci(r.fpr_ci)} & "
                     f"{r.macro_f1:.2f} {ci(r.macro_f1_ci)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\footnotesize Dropout recall: share of actual "
              r"dropouts predicted as Dropout. False alert rate: share of "
              r"non-dropouts predicted as Dropout.", r"\end{table}"]
    (GEN / "tab_subgroups.tex").write_text("\n".join(lines).replace(",", "{,}")
                                           .replace("{,} ", ", ") + "\n")


subgroup_table()

# ── macros ───────────────────────────────────────────────────────────────────
for s in ["binary", "multiclass", "sem1", "enrol"]:
    for metric in ["f1", "auc", "accuracy"]:
        bm, v = best(s, metric)
        key = macro_name(s) + {"f1": "F", "auc": "Auc", "accuracy": "Acc"}[metric]
        m(f"best{key}Model", bm)
        m(f"best{key}", f3(v))
for s in ["binary", "multiclass", "sem1", "enrol"]:
    for model in MODELS:
        for metric in ["f1", "auc", "accuracy", "auc_dropout", "f1_enrolled",
                       "f1_dropout", "f1_graduate", "brier_dropout"]:
            if metric + "" in S.columns and not pd.isna(g(s, model, metric)):
                m(f"{macro_name(s)}{macro_name(model)}{macro_name(metric)}",
                  f3(g(s, model, metric)))

# averages and gaps (mean over the 8 models)
for metric in ["f1", "accuracy", "auc"]:
    b = S[S.setting == "binary"][metric].mean()
    mc = S[S.setting == "multiclass"][metric].mean()
    m(f"avgBinary{macro_name(metric)}", f3(b))
    m(f"avgMulti{macro_name(metric)}", f3(mc))
    m(f"gap{macro_name(metric)}", f3(b - mc))
    m(f"gapRel{macro_name(metric)}", f"{100 * (b - mc) / b:.1f}")
# spread of macro F1 across models
for s in ["binary", "multiclass"]:
    d = S[S.setting == s]["f1"]
    m(f"spread{macro_name(s)}F", f3(d.max() - d.min()))
# number of models tied with the best (p >= .05)
for s in ["binary", "multiclass", "sem1", "enrol"]:
    for metric in ["f1", "auc"]:
        tt = T[(T.setting == s) & (T.metric == metric)]
        m(f"nTied{macro_name(s)}{macro_name(metric)}", str(int((tt.p >= 0.05).sum())))
# retained share of full-model F1
bm = A["best_model"]
m("bestModel", bm)
for s in ["sem1", "enrol"]:
    m(f"retain{macro_name(s)}", f"{100 * g(s, bm, 'f1') / g('multiclass', bm, 'f1'):.1f}")
# stages
for (lab, model, f, sd, fd, fe, fg, ad), key in zip(stage_rows, ["Enrol", "Semone", "Full"]):
    m(f"stage{key}Model", model)
    m(f"stage{key}F", f3(f))
    m(f"stage{key}Fenrolled", f3(fe))
    m(f"stage{key}AucDrop", f3(ad))
# calibration
for model, v in A["calibration"].items():
    m(f"ece{macro_name(model)}", f3(v["ece"]))
    m(f"brier{macro_name(model)}", f3(v["brier"]))
# confusion
cm = np.array(A["confusion"])
rown = cm / cm.sum(1, keepdims=True)
m("cmEnrolledCorrect", f"{100 * rown[1, 1]:.0f}")
m("cmEnrolledAsGrad", f"{100 * rown[1, 2]:.0f}")
m("cmEnrolledAsDrop", f"{100 * rown[1, 0]:.0f}")
m("cmDropCorrect", f"{100 * rown[0, 0]:.0f}")
m("cmGradCorrect", f"{100 * rown[2, 2]:.0f}")
# feature selection
fs = pd.DataFrame(A["feature_selection"])
full = fs.iloc[-1]
m("fsFull", f3(full.f1))
for _, r in fs.iterrows():
    m(f"fsK{int(r.k)}".replace("0", "zero").replace("1", "one").replace("2", "two")
      .replace("3", "three").replace("5", "five").replace("6", "six"), f3(r.f1))
k_ok = fs[fs.f1 >= full.f1 - full.f1_sd].k.min()
m("fsSmallestK", str(int(k_ok)))
# SHAP
imp = pd.DataFrame(A["shap"]["importance"])
m("shapXgbF", f3(A["shap"]["xgb_holdout_f1"]))
for i, (feat, row) in enumerate(imp.head(5).iterrows()):
    w = ["One", "Two", "Three", "Four", "Five"][i]
    m(f"shapTop{w}", feat)
    m(f"shapTop{w}Val", f3(row["mean"]))
# subgroups
sub = pd.DataFrame(A["subgroups"])
for _, r in sub.iterrows():
    key = macro_name(r.attribute) + macro_name(r.group.replace("<=", "le").replace(">", "gt")
                                                .replace("-", "to").replace("20", "twenty")
                                                .replace("21", "twentyone")
                                                .replace("25", "twentyfive"))
    m(f"sub{key}Tpr", f"{r.tpr:.2f}")
    m(f"sub{key}Fpr", f"{r.fpr:.2f}")
    m(f"sub{key}N", f"{r.n:,}".replace(",", "{,}"))
    m(f"sub{key}Rate", f"{r.dropout_rate:.2f}")

with open(GEN / "numbers.tex", "w") as fh:
    fh.write("% Auto-generated by src/make_tables.py. Do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(macros)} macros written")
