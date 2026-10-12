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


# ── decision analyses (decision.py, sensitivity.py, tuning.py) ───────────────
D = json.load(open(RES / "decision.json"))
SENS = json.load(open(RES / "sensitivity.json"))
STAGE_LABEL = {"enrol": "Enrolment", "sem1": "1st semester", "multiclass": "2nd semester"}
WORD = {"enrol": "Enrol", "sem1": "Semone", "multiclass": "Full"}


def cap_row(rows, cap):
    return next(r for r in rows if abs(r["capacity"] - cap) < 1e-9)


def alert_table():
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
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
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
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
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
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
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
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
lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
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

with open(GEN / "numbers.tex", "w") as fh:
    fh.write("% Auto-generated by src/make_tables.py. Do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(macros)} macros written (with decision analyses)")
