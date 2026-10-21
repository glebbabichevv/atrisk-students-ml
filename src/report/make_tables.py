"""
Write every number used in the paper to paper/generated/, so the text and the
results cannot drift apart again.

  paper/generated/numbers.tex   \\newcommand macros used in the prose
  paper/generated/tab_*.tex     tables

Last step of run_all.py. Reads results/uci/ (and results/upv/ when present).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.models import corrected_ttest  # noqa: E402
from common.paths import GENERATED as GEN, UCI_RESULTS as RES, UPV_RESULTS  # noqa: E402
from uci.data import load_dataset  # noqa: E402

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
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
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
results_table("sem1", ["accuracy", "f1", "auc", "f1_dropout", "f1_enrolled",
                       "f1_graduate"],
              "Three-class setting with first-semester records only (30 features).",
              "tab:sem1")
results_table("enrol", ["accuracy", "f1", "auc", "f1_dropout", "f1_enrolled",
                        "f1_graduate"],
              "Three-class setting with information available at enrolment (24 features).",
              "tab:enrol")
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
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
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
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
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
    fh.write("% Auto-generated by src/report/make_tables.py. Do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(macros)} macros written")


# ── decision analyses (decision.py, sensitivity.py, tuning.py) ───────────────
D = json.load(open(RES / "decision.json"))
SENS = json.load(open(RES / "sensitivity.json"))
STAGE_LABEL = {"enrol": "Enrolment", "sem1": "1st semester", "multiclass": "2nd semester"}
WORD = {"enrol": "Enrol", "sem1": "Semone", "multiclass": "Full"}


def cap_row(rows, cap):
    return next(r for r in rows if abs(r["capacity"] - cap) < 1e-9)


def alert_table():
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Alert lists of fixed size. Students are ranked by the "
             r"out-of-fold XGBoost probability of dropout; the top 10\% or 20\% of the "
             r"cohort are flagged. Precision: share of flagged students who drop out "
             r"(95\% bootstrap interval). Reach: share of all dropouts who are flagged.}",
             r"\label{tab:alerts}", r"\begin{tabular}{llcccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{Top 10\%} & \multicolumn{2}{c}{Top 20\%} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Prediction point & Features & Precision & Reach & Precision & Reach \\",
             r"\midrule"]
    for s in ["enrol", "sem1", "multiclass"]:
        for key, lab in [(s, "all"), (s + "_nofee", "without fee status")]:
            a, b = cap_row(D["alert_lists"][key], 0.10), cap_row(D["alert_lists"][key], 0.20)
            ci = lambda r: f"[{r['precision_ci'][0]:.2f}, {r['precision_ci'][1]:.2f}]"  # noqa: E731
            first = STAGE_LABEL[s] if lab == "all" else ""
            lines.append(f"{first} & {lab} & {a['precision']:.2f} {ci(a)} & {a['recall']:.2f} & "
                         f"{b['precision']:.2f} {ci(b)} & {b['recall']:.2f} \\\\")
    base = D["base_rate"]
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\par\smallskip\footnotesize Dropout base rate {base:.2f}. A random "
              r"list has precision equal to the base rate and reaches 10\% or 20\% of "
              r"dropouts.", r"\end{table}"]
    (GEN / "tab_alerts.tex").write_text("\n".join(lines) + "\n")


alert_table()
for key, rows in D["alert_lists"].items():
    s = key.replace("_nofee", "")
    suffix = "Nofee" if key.endswith("_nofee") else ""
    for cap, cw in [(0.10, "Ten"), (0.20, "Twenty")]:
        r = cap_row(rows, cap)
        m(f"alert{WORD[s]}{suffix}P{cw}", f"{r['precision']:.2f}")
        m(f"alert{WORD[s]}{suffix}R{cw}", f"{r['recall']:.2f}")
        m(f"alert{WORD[s]}{suffix}Lift{cw}", f"{r['lift']:.1f}")
        m(f"alert{WORD[s]}{suffix}PPct{cw}", f"{100 * r['precision']:.0f}\\%")
        m(f"alert{WORD[s]}{suffix}RPct{cw}", f"{100 * r['recall']:.0f}\\%")
m("baseRate", f"{D['base_rate']:.2f}")
# perfect ranking reach at 20%
m("perfectReachTwenty", f"{min(1, 0.20 / D['base_rate']):.2f}")
m("perfectReachTen", f"{100 * min(1, 0.10 / D['base_rate']):.0f}\\%")
for k_, v_ in D["top_profile"].items():
    m("profile" + "".join(w.capitalize() for w in k_.replace("10", "ten").split("_")),
      f"{100 * v_:.0f}")


def threshold_table():
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{One threshold for everyone vs.\ group-specific thresholds that "
             r"give each group the same dropout recall (XGBoost, both semesters). The single threshold flags 20\% "
             r"of students. Thresholds are chosen on four folds of students and applied "
             r"to the fifth (cross-fitting).}",
             r"\label{tab:thresholds}", r"\begin{tabular}{llrcccc}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Single threshold} & "
             r"\multicolumn{2}{c}{Group thresholds} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Attribute & Group & $n$ & Recall & False alerts & Recall & False alerts \\",
             r"\midrule"]
    for t in D["thresholds"]:
        single = {r["group"]: r for r in t["rows"] if r["policy"] == "single"}
        group = {r["group"]: r for r in t["rows"] if r["policy"] == "per-group"}
        first = True
        for gname in single:
            a, b = single[gname], group[gname]
            g = (gname.replace("<=", r"$\le$").replace(">", "$>$").replace("21-25", "21--25"))
            if gname == "All":
                g = r"\emph{All}"
            lines.append(f"{t['attribute'] if first else ''} & {g} & {a['n']:,} & "
                         f"{a['recall']:.2f} & {a['fpr']:.3f} & {b['recall']:.2f} & "
                         f"{b['fpr']:.3f} \\\\".replace(",", "{,}").replace("{,} ", ", "))
            first = False
        if t is not D["thresholds"][-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\footnotesize False alerts: share of students who do not "
              r"drop out but are flagged.", r"\end{table}"]
    (GEN / "tab_thresholds.tex").write_text("\n".join(lines) + "\n")


threshold_table()
for t in D["thresholds"]:
    a = macro_name(t["attribute"])
    for r in t["rows"]:
        gk = macro_name(r["group"].replace("<=", "le").replace(">", "gt").replace("-", "to")
                        .replace("21", "twentyone").replace("25", "twentyfive")
                        .replace("20", "twenty"))
        pk = "Single" if r["policy"] == "single" else "Group"
        m(f"thr{a}{gk}{pk}Recall", f"{r['recall']:.2f}")
        m(f"thr{a}{gk}{pk}Fpr", f"{r['fpr']:.3f}")
        m(f"thr{a}{gk}{pk}Alerts", f"{r['alerts']:,}".replace(",", "{,}"))
        if "precision" in r:
            m(f"thr{a}{gk}{pk}Precision", f"{r['precision']:.2f}")


def sensitivity_table():
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Sensitivity to the two fee-status variables (\emph{Debtor}, "
             r"\emph{Tuition fees up to date}), whose recording time is not documented. "
             r"Three-class setting, mean $\pm$ SD over 25 folds.}",
             r"\label{tab:sensitivity}", r"\begin{tabular}{llcccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{With fee status} & "
             r"\multicolumn{2}{c}{Without fee status} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Prediction point & Model & Macro F1 & AUC Dropout & Macro F1 & AUC Dropout \\",
             r"\midrule"]
    for s in ["enrol", "sem1", "multiclass"]:
        for k, model in enumerate(["XGBoost", "Logistic Regression"]):
            a = next(r for r in SENS if r["stage"] == s and r["model"] == model and r["fees"])
            b = next(r for r in SENS if r["stage"] == s and r["model"] == model and not r["fees"])
            f = lambda r, c: f"{r[c]:.3f} $\\pm$ {r[c + '_sd']:.3f}"  # noqa: E731
            lines.append(f"{STAGE_LABEL[s] if k == 0 else ''} & {model} & {f(a, 'f1')} & "
                         f"{f(a, 'auc_dropout')} & {f(b, 'f1')} & {f(b, 'auc_dropout')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "tab_sensitivity.tex").write_text("\n".join(lines) + "\n")


sensitivity_table()
for r in SENS:
    key = f"sens{WORD[r['stage']]}{macro_name(r['model'])}{'Fees' if r['fees'] else 'Nofee'}"
    m(key + "F", f3(r["f1"]))
    m(key + "Auc", f3(r["auc_dropout"]))

TUNE_PATH = RES / "tuning.json"
if TUNE_PATH.exists():
    TU = json.load(open(TUNE_PATH))
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{Fixed hyperparameters vs.\ nested tuning (randomised search, "
             r"inner 3-fold CV on macro F1) in the three-class setting. Mean $\pm$ SD "
             r"over 10 outer folds; $p$: corrected resampled $t$-test, tuned vs.\ fixed.}",
             r"\label{tab:tuning}", r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Macro F1} & \multicolumn{3}{c}{Macro AUC} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Model & Fixed & Tuned & $p$ & Fixed & Tuned & $p$ \\", r"\midrule"]
    for name, r in TU.items():
        f, a = r["f1"], r["auc"]
        lines.append(f"{name} & {f['fixed']:.3f} $\\pm$ {f['fixed_sd']:.3f} & "
                     f"{f['tuned']:.3f} $\\pm$ {f['tuned_sd']:.3f} & {f['p']:.2f} & "
                     f"{a['fixed']:.3f} & {a['tuned']:.3f} & {a['p']:.2f} \\\\")
        m(f"tune{macro_name(name)}Fixed", f3(f["fixed"]))
        m(f"tune{macro_name(name)}Tuned", f3(f["tuned"]))
        m(f"tune{macro_name(name)}P", f"{f['p']:.2f}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "tab_tuning.tex").write_text("\n".join(lines) + "\n")
    m("tuneMaxGain", f3(max(r["f1"]["diff"] for r in TU.values())))
    m("tuneMinP", f"{min(r['f1']['p'] for r in TU.values()):.2f}")

# ── transfer to unseen programmes (transfer.py) ─────────────────────────────
TR = json.load(open(RES / "transfer.json"))
lines = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Transfer to unseen degree programmes. \emph{Seen}: repeated 5-fold CV "
         r"over all students. \emph{Held out}: each of the 17 programmes predicted by a model "
         r"trained on the other 16. Both without the programme code. Pooled over all students; "
         r"brackets: 95\% cluster-bootstrap interval (by programme) for the change in AUC.}",
         r"\label{tab:transfer}", r"\begin{tabular}{llccccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{AUC Dropout} & & \multicolumn{2}{c}{Macro F1} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
         r"Prediction point & Model & Seen & Held out & Change & Seen & Held out \\",
         r"\midrule"]
for stage, lab in [("sem1", "1st semester"), ("multiclass", "2nd semester")]:
    for k, name in enumerate(["XGBoost", "Logistic Regression"]):
        P = TR[stage][name]["pooled"]
        ci = P["auc_diff_ci"]
        diff = P["lopo"]["auc_dropout"] - P["within"]["auc_dropout"]
        lines.append(f"{lab if k == 0 else ''} & {name} & {P['within']['auc_dropout']:.3f} & "
                     f"{P['lopo']['auc_dropout']:.3f} & ${diff:+.3f}$ [${ci[0]:+.3f}$, ${ci[1]:+.3f}$] & "
                     f"{P['within']['f1']:.3f} & {P['lopo']['f1']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(GEN / "tab_transfer.tex").write_text("\n".join(lines) + "\n")
for stage, sk in [("sem1", "Semone"), ("multiclass", "Full")]:
    for name in ["XGBoost", "Logistic Regression"]:
        P, S = TR[stage][name]["pooled"], TR[stage][name]["summary"]
        key = f"tr{sk}{macro_name(name)}"
        m(key + "InAuc", f3(P["within"]["auc_dropout"]))
        m(key + "OutAuc", f3(P["lopo"]["auc_dropout"]))
        m(key + "InF", f3(P["within"]["f1"]))
        m(key + "OutF", f3(P["lopo"]["f1"]))
        m(key + "InAucExcl", f3(P["within_excl171"]["auc_dropout"]))
        m(key + "OutAucExcl", f3(P["lopo_excl171"]["auc_dropout"]))
        m(key + "MedDrop", f"{S['median_auc_drop']:+.3f}".replace("+0.000", "0.000").replace("-0.000", "0.000"))
        m(key + "WorstDrop", f"{S['worst_auc_drop']:+.3f}")
        m(key + "NBigDrop", str(S["n_drop_gt_005"]))
        m(key + "CalGap", f"{S['calib_gap_mean_abs']:.2f}")
        m(key + "OutAucMin", f3(S["lopo_auc_min"]))
        m(key + "OutAucMax", f3(S["lopo_auc_max"]))
S = TR["sem1"]["XGBoost"]["summary"]
m("trNProg", str(S["n_programmes_reported"]))
m("trZeroAnim", f"{100 * S['zero_units_share_171']:.0f}")
m("trZeroGradAnim", f"{100 * S['zero_units_graduates_171']:.0f}")
anim = next(r for r in TR["sem1"]["XGBoost"]["per_programme"] if r["code"] == 171)
m("trAnimMeanP", f"{anim['lopo_mean_p']:.2f}")
m("trAnimRate", f"{anim['dropout_rate']:.2f}")
m("trAnimInAuc", f3(anim["within_auc"]))
m("trAnimOutAuc", f3(anim["lopo_auc"]))

# ── second university (UPV) and the replication table ───────────────────────
UPV_PATH = UPV_RESULTS / "upv_results.json"
CVS = pd.read_csv(RES / "cv_summary.csv")   # S is reused above for the transfer summary
UPV_STAGES = [("enrol", "Enrolment", "Enrol"), ("sem1", "+ 1st semester", "Semone"),
              ("sem1_lms", "+ LMS to January", "Semonelms"),
              ("year", "+ 2nd semester", "Year"), ("year_lms", "+ LMS to June", "Yearlms")]
UPV_WORD = {k: w for k, _, w in UPV_STAGES} | {"lms1_only": "Lmsonly"}
SHORT = {"Logistic Regression": "Lr", "XGBoost": "Xgb"}
SPLITS = [("2018_to_2021", "Ta"), ("2018_2021_to_2022", "Tb")]
YEAR_WORD = {"2018": "Eighteen", "2021": "Twentyone", "2022": "Twentytwo"}


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def n_tied(p_vs_best, alpha=0.05):
    """Models not significantly worse than the best (the best included)."""
    return 1 + sum(p >= alpha for p in p_vs_best.values())


def top20(rows):
    return next(r for r in rows if abs(r["capacity"] - 0.2) < 1e-9)


def thr_row(t, policy, group):
    return next(r for r in t["rows"] if r["policy"] == policy and r["group"] == group)


def best_auc(cv, stage):
    return cv[stage]["summary"][cv[stage]["best_auc"]]["auc"]


def upv_tables(U):
    cv = U["cv"]
    # macros: sample, CV at each stage, calibration, alert lists
    m("upvN", thousands(sum(U["n_by_year"].values())))
    m("upvRate", f"{U['dropout_rate']:.2f}")
    m("upvRatePct", f"{100 * U['dropout_rate']:.0f}")
    for yr, n in U["n_by_year"].items():
        m(f"upvN{YEAR_WORD[yr]}", thousands(n))
        m(f"upvRate{YEAR_WORD[yr]}", f"{U['dropout_rate_by_year'][yr]:.3f}")
    for stage, word in UPV_WORD.items():
        S_ = cv[stage]["summary"]
        bm = cv[stage]["best_auc"]
        m(f"upv{word}BestName", bm)
        m(f"upv{word}BestAuc", f3(S_[bm]["auc"]))
        m(f"upv{word}NFeat", str(cv[stage]["n_features"]))
        for name, sh in SHORT.items():
            m(f"upv{word}{sh}Auc", f3(S_[name]["auc"]))
        if stage == "lms1_only":
            continue
        m(f"upv{word}Tied", str(n_tied(cv[stage]["p_vs_best_auc"])))
        for name, sh in SHORT.items():
            m(f"upv{word}{sh}Ece", f3(U["calibration"][stage][name]["ece"]))
            for r in U["alerts"][stage][name]:
                c = int(round(100 * r["capacity"]))
                if c in (10, 20):
                    tag = {10: "Ten", 20: "Twenty"}[c]
                    m(f"upv{word}{sh}Top{tag}P", f"{r['precision']:.2f}")
                    m(f"upv{word}{sh}Top{tag}R", f"{r['recall']:.2f}")
                    m(f"upv{word}{sh}Top{tag}RPct", f"{100 * r['recall']:.0f}")
    m("upvLmsGainSemone", f"{best_auc(cv, 'sem1_lms') - best_auc(cv, 'sem1'):+.3f}")
    m("upvLmsGainYear", f"{best_auc(cv, 'year_lms') - best_auc(cv, 'year'):+.3f}")
    m("upvLmsOnlyGain", f"{best_auc(cv, 'lms1_only') - best_auc(cv, 'enrol'):+.3f}")
    m("upvTopTwentyMaxReach", f"{min(1, 0.20 / U['dropout_rate']):.2f}")

    # macros: temporal and unseen-degree validation
    for split, tag in SPLITS:
        for stage, word in UPV_WORD.items():
            if stage not in U["temporal"][split]:
                continue
            for name, sh in SHORT.items():
                r = U["temporal"][split][stage][name]
                m(f"upv{tag}{word}{sh}Auc", f3(r["auc"]))
                m(f"upv{tag}{word}{sh}Pred", f"{r['mean_pred']:.3f}")
                m(f"upv{tag}{word}{sh}Ece", f3(r["ece"]))
                m(f"upv{tag}{word}Obs", f"{r['observed']:.3f}")
    for stage, d in U["grouped_degree"].items():
        for name, sh in SHORT.items():
            m(f"upvGd{UPV_WORD[stage]}{sh}Auc", f3(d[name]["auc"]))

    # macros: subgroup recall, single vs per-group thresholds
    for stage, lst in U["thresholds"].items():
        for t in lst:
            for r in t["rows"]:
                pol = "Single" if r["policy"] == "single" else "Group"
                if r["group"] == "All":
                    m(f"upvThr{UPV_WORD[stage]}{macro_name(t['attribute'])}{pol}Prec",
                      f"{r['precision']:.2f}")
                    continue
                key = f"upvThr{UPV_WORD[stage]}{macro_name(r['group'])}{pol}"
                m(key + "Rec", f"{r['recall']:.2f}")
                m(key + "Fpr", f"{r['fpr']:.3f}")
                m(f"upvThr{macro_name(r['group'])}Rate", f"{r['dropout_rate']:.2f}")

    # table: the five prediction points
    lines = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Second university (UPV): dropout prediction for first-year entrants at five "
             r"points in the first year. AUC: mean over 5-fold CV repeated 5 times. \emph{Tied}: "
             r"models not significantly below the best (corrected $t$-test, $p \ge 0.05$), out of 8. "
             r"ECE and the top-20\% list use out-of-fold XGBoost predictions. "
             f"Dropout rate {U['dropout_rate']:.2f}, so a perfect top-20\\% list reaches "
             f"{min(1, 0.20 / U['dropout_rate']):.2f}.}}",
             r"\label{tab:upv_stages}", r"\begin{tabular}{lrlcccccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{Best model} & & & & \multicolumn{2}{c}{Top-20\% list} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){8-9}",
             r"Information used & Features & Model & AUC & LR AUC & Tied & ECE & Precision & Reach \\",
             r"\midrule"]
    for stage, lab, _ in UPV_STAGES:
        S_ = cv[stage]["summary"]
        bm = cv[stage]["best_auc"]
        top = top20(U["alerts"][stage]["XGBoost"])
        lines.append(f"{lab} & {cv[stage]['n_features']} & {bm} & {S_[bm]['auc']:.3f} & "
                     f"{S_['Logistic Regression']['auc']:.3f} & {n_tied(cv[stage]['p_vs_best_auc'])} & "
                     f"{U['calibration'][stage]['XGBoost']['ece']:.3f} & {top['precision']:.2f} & "
                     f"{top['recall']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "tab_upv_stages.tex").write_text("\n".join(lines) + "\n")

    # table: students unlike the training data
    lines = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{UPV: dropout AUC for logistic regression (LR) and XGBoost when the model "
             r"meets students unlike its training data. \emph{CV}: repeated 5-fold CV over all "
             r"cohorts. Later cohorts: trained on earlier entry years only. \emph{Unseen "
             r"degree}: 5-fold CV grouped by degree, degree indicators removed.}",
             r"\label{tab:upv_shift}", r"\begin{tabular}{lcccccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{CV} & \multicolumn{2}{c}{2018 $\to$ 2021} & "
             r"\multicolumn{2}{c}{2018, 2021 $\to$ 2022} & \multicolumn{2}{c}{Unseen degree} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Information used & LR & XGB & LR & XGB & LR & XGB & LR & XGB \\", r"\midrule"]
    for stage, lab, _ in UPV_STAGES:
        cells = [cv[stage]["summary"][n]["auc"] for n in SHORT]
        for split, _ in SPLITS:
            cells += [U["temporal"][split][stage][n]["auc"] for n in SHORT]
        gd = U["grouped_degree"].get(stage)
        txt = [f"{v:.3f}" for v in cells] + ([f"{gd[n]['auc']:.3f}" for n in SHORT] if gd else ["--", "--"])
        lines.append(f"{lab} & " + " & ".join(txt) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "tab_upv_shift.tex").write_text("\n".join(lines) + "\n")


def replication_table(U):
    """The main UCI findings next to the same quantities at UPV."""
    cv = U["cv"]
    best_uci = {s: CVS[CVS.setting == s]["auc_dropout"].max() for s in ("enrol", "sem1", "multiclass")}
    best_upv = {s: max(v["auc"] for v in cv[s]["summary"].values()) for s in ("enrol", "sem1", "year")}
    sh_uci = (best_uci["sem1"] - best_uci["enrol"]) / (best_uci["multiclass"] - best_uci["enrol"])
    sh_upv = (best_upv["sem1"] - best_upv["enrol"]) / (best_upv["year"] - best_upv["enrol"])
    m("repShareUci", f"{100 * sh_uci:.0f}")
    m("repShareUpv", f"{100 * sh_upv:.0f}")

    # UCI ties on the same metric as UPV: dropout-vs-rest AUC, from the saved fold scores
    folds = json.load(open(RES / "cv_fold_scores.json"))["sem1"]
    auc = {n: [f["auc_dropout"] for f in fs] for n, fs in folds.items()}
    top = max(auc, key=lambda n: np.mean(auc[n]))
    n_all = len(load_dataset())
    p_uci = {n: corrected_ttest(auc[top], a, n_all - n_all // 5, n_all // 5)[1]
             for n, a in auc.items() if n != top}
    tied_uci = n_tied(p_uci)
    tied_upv = n_tied(cv["sem1"]["p_vs_best_auc"])
    m("repTiedUci", str(tied_uci))
    m("repTiedUpv", str(tied_upv))

    uci_top = top20(D["alert_lists"]["multiclass"])
    upv_top = top20(U["alerts"]["year"]["XGBoost"])
    lift_uci = uci_top["precision"] / D["base_rate"]
    lift_upv = upv_top["precision"] / U["dropout_rate"]
    m("repLiftUci", f"{lift_uci:.1f}")
    m("repLiftUpv", f"{lift_upv:.1f}")

    sch = next(t for t in D["thresholds"] if t["attribute"] == "Scholarship")
    mode = next(t for t in U["thresholds"]["year"] if t["attribute"] == "Study mode")

    def rec(t, pol, gr):
        return thr_row(t, pol, gr)["recall"]

    tr = TR["sem1"]
    d_uci = {n: tr[n]["pooled"]["lopo"]["auc_dropout"] - tr[n]["pooled"]["within"]["auc_dropout"]
             for n in SHORT}
    gd = U["grouped_degree"]["sem1"]
    d_gd = {n: gd[n]["auc"] - gd[n]["cv_auc"] for n in SHORT}
    d_ta = {n: U["temporal"]["2018_to_2021"]["sem1"][n]["auc"] - cv["sem1"]["summary"][n]["auc"]
            for n in SHORT}
    d_tb = {n: U["temporal"]["2018_2021_to_2022"]["sem1"][n]["auc"] - cv["sem1"]["summary"][n]["auc"]
            for n in SHORT}
    for tag, d in [("Uci", d_uci), ("UpvGd", d_gd), ("UpvTa", d_ta), ("UpvTb", d_tb)]:
        for n, sh in SHORT.items():
            m(f"repShift{tag}{sh}", f"{d[n]:+.3f}")

    cal_uci = A["calibration"]["XGBoost"]["ece"]
    cal_upv = U["calibration"]["year"]["XGBoost"]["ece"]

    def lr_xgb(d):
        return f"{d['Logistic Regression']:+.3f} / {d['XGBoost']:+.3f}"

    rows = [
        (r"Models tied with the best, AUC after 1st semester (of 8)", f"{tied_uci}", f"{tied_upv}"),
        (r"Dropout AUC: enrolment / 1st semester / end of year",
         " / ".join(f"{best_uci[s]:.2f}" for s in ("enrol", "sem1", "multiclass")),
         " / ".join(f"{best_upv[s]:.2f}" for s in ("enrol", "sem1", "year"))),
        (r"Share of the first-year AUC gain present after 1st semester",
         f"{100 * sh_uci:.0f}\\%", f"{100 * sh_upv:.0f}\\%"),
        (r"Calibration error (ECE), end of year, XGBoost", f"{cal_uci:.3f}", f"{cal_upv:.3f}"),
        (r"Top-20\% list, end of year: precision (lift over base rate)",
         f"{uci_top['precision']:.2f} ({lift_uci:.1f}$\\times$)",
         f"{upv_top['precision']:.2f} ({lift_upv:.1f}$\\times$)"),
        (r"Top-20\% list, end of year: reach", f"{uci_top['recall']:.2f}", f"{upv_top['recall']:.2f}"),
        (r"Recall, low- vs.\ high-dropout group, one threshold",
         f"{rec(sch, 'single', 'Yes'):.2f} vs.\\ {rec(sch, 'single', 'No'):.2f}",
         f"{rec(mode, 'single', 'Full-time'):.2f} vs.\\ {rec(mode, 'single', 'Part-time'):.2f}"),
        (r"\quad same, with group-specific thresholds",
         f"{rec(sch, 'per-group', 'Yes'):.2f} vs.\\ {rec(sch, 'per-group', 'No'):.2f}",
         f"{rec(mode, 'per-group', 'Full-time'):.2f} vs.\\ {rec(mode, 'per-group', 'Part-time'):.2f}"),
        (r"AUC change, unseen programme, LR / XGBoost", lr_xgb(d_uci), lr_xgb(d_gd)),
        (r"AUC change, later cohort (2018 $\to$ 2021), LR / XGBoost", "--", lr_xgb(d_ta)),
        (r"AUC change, later cohort (2018, 2021 $\to$ 2022), LR / XGBoost", "--", lr_xgb(d_tb)),
    ]
    lines = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Main findings at the first university (UCI, Portugal) and the same quantities "
             r"at the second (UPV, Spain). UCI: three-class model, P(dropout) scored as dropout vs.\ "
             r"rest; low-dropout group: scholarship holders. UPV: binary model; low-dropout group: "
             r"full-time students. Lists and thresholds use out-of-fold XGBoost predictions. "
             r"Changes in AUC are relative to repeated CV on the same features. UCI has a single "
             r"cohort, so no later-cohort test is possible.}",
             r"\label{tab:replication}", r"\begin{tabular}{p{7.2cm}cc}", r"\toprule",
             r"Quantity & UCI & UPV \\", r"\midrule"]
    lines += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "tab_replication.tex").write_text("\n".join(lines) + "\n")


if UPV_PATH.exists():
    U = json.load(open(UPV_PATH))
    upv_tables(U)
    replication_table(U)
    print("UPV tables written")
else:
    print("No UPV results: UPV tables and macros skipped.")

with open(GEN / "numbers.tex", "w") as fh:
    fh.write("% Auto-generated by src/report/make_tables.py. Do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(macros)} macros written (with decision analyses)")
