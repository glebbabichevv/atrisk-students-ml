"""
Leave-one-programme-out (LOPO) validation: does the model transfer to a degree
programme it has never seen?

For each of the 17 programmes, a model is trained on the other 16 and tested on
the held-out one. The `Course` code is removed from the features, since a new
programme would have an unseen code. As the in-distribution reference, the same
model (also without `Course`) is evaluated with 5-fold CV repeated 5 times over
all students; the first repeat gives one out-of-fold prediction per student.

UCI step 5. Outputs: results/uci/transfer.json, paper/figures/fig_transfer.pdf
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.models import SEED, get_models  # noqa: E402
from common.paths import UCI_RESULTS as RES  # noqa: E402
from common.plotting import C, INK2, plt, save  # noqa: E402
from uci.data import load_dataset, setting_xy  # noqa: E402
MODELS = ["XGBoost", "Logistic Regression"]
STAGES = ["sem1", "multiclass"]
NAMES = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing", 9773: "Journalism and Communication",
    9853: "Basic Education", 9991: "Management (evening)",
}
MIN_N = 50  # programmes smaller than this are pooled but not reported one by one


def fit_predict(model, X, y, tr, te):
    m = clone(model).fit(X[tr], y[tr])
    return te, m.predict_proba(X[te])


def ece(p, d, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - d[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def scores(y, prob):
    d = y == 0
    out = {"f1": f1_score(y, prob.argmax(1), average="macro"),
           "n": int(len(y)), "dropout_rate": float(d.mean())}
    out["auc_dropout"] = float(roc_auc_score(d, prob[:, 0])) if 0 < d.sum() < len(d) else None
    return out


def paired_boot(y, pa, pb, groups, n=1000, seed=SEED):
    """Cluster bootstrap over programmes for the pooled Dropout-AUC difference."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    diffs = []
    for _ in range(n):
        idx = np.concatenate([idx_by[g] for g in rng.choice(ug, len(ug))])
        d = y[idx] == 0
        diffs.append(roc_auc_score(d, pa[idx, 0]) - roc_auc_score(d, pb[idx, 0]))
    return np.percentile(diffs, [2.5, 97.5]).tolist()


def run(df, stage, name):
    X, y, feats = setting_xy(df, stage)
    course = df["Course"].to_numpy()
    keep = [i for i, f in enumerate(feats) if f != "Course"]
    X = X[:, keep]
    feats = [feats[i] for i in keep]
    model = get_models(3, feats)[name]

    # in-distribution reference: repeated stratified CV, first repeat as OOF
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X, y))
    res = Parallel(n_jobs=-1)(delayed(fit_predict)(model, X, y, tr, te) for tr, te in cv)
    oof_in = np.zeros((len(y), 3))
    for k, (te, prob) in enumerate(res[:5]):
        oof_in[te] = prob

    # leave-one-programme-out
    progs = np.unique(course)
    splits = [(np.where(course != c)[0], np.where(course == c)[0]) for c in progs]
    res = Parallel(n_jobs=-1)(delayed(fit_predict)(model, X, y, tr, te) for tr, te in splits)
    oof_out = np.zeros((len(y), 3))
    for te, prob in res:
        oof_out[te] = prob

    d = y == 0
    pooled = {
        "within": {"f1": float(f1_score(y, oof_in.argmax(1), average="macro")),
                   "auc_dropout": float(roc_auc_score(d, oof_in[:, 0])),
                   "ece_dropout": ece(oof_in[:, 0], d)},
        "lopo": {"f1": float(f1_score(y, oof_out.argmax(1), average="macro")),
                 "auc_dropout": float(roc_auc_score(d, oof_out[:, 0])),
                 "ece_dropout": ece(oof_out[:, 0], d)},
    }
    pooled["auc_diff_ci"] = paired_boot(y, oof_out, oof_in, course)
    # Animation and Multimedia Design (171) records almost no curricular units;
    # report the pooled result without it as well (see text).
    k = course != 171
    pooled["within_excl171"] = {"auc_dropout": float(roc_auc_score(d[k], oof_in[k, 0])),
                                "f1": float(f1_score(y[k], oof_in[k].argmax(1), average="macro"))}
    pooled["lopo_excl171"] = {"auc_dropout": float(roc_auc_score(d[k], oof_out[k, 0])),
                              "f1": float(f1_score(y[k], oof_out[k].argmax(1), average="macro"))}
    pooled["auc_diff_ci_excl171"] = paired_boot(y[k], oof_out[k], oof_in[k], course[k])
    per = []
    for c in progs:
        m = course == c
        a, b = scores(y[m], oof_in[m]), scores(y[m], oof_out[m])
        per.append({"code": int(c), "name": NAMES.get(int(c), str(c)), "n": a["n"],
                    "dropout_rate": a["dropout_rate"],
                    "within_f1": a["f1"], "lopo_f1": b["f1"],
                    "within_auc": a["auc_dropout"], "lopo_auc": b["auc_dropout"],
                    "lopo_mean_p": float(oof_out[m, 0].mean())})
    big = [r for r in per if r["n"] >= MIN_N]
    diffs = np.array([r["lopo_auc"] - r["within_auc"] for r in big])
    zero = df["Curricular units 1st sem (enrolled)"].to_numpy() == 0
    summary = {"n_programmes_reported": len(big),
               "zero_units_share_171": float(zero[course == 171].mean()),
               "zero_units_share_other": float(zero[course != 171].mean()),
               "zero_units_graduates_171": float(zero[(course == 171) & (y == 2)].mean()),
               "median_auc_drop": float(np.median(diffs)),
               "worst_auc_drop": float(diffs.min()),
               "n_drop_gt_005": int((diffs < -0.05).sum()),
               "lopo_auc_min": float(min(r["lopo_auc"] for r in big)),
               "lopo_auc_max": float(max(r["lopo_auc"] for r in big)),
               "worst_programme": min(big, key=lambda r: r["lopo_auc"] - r["within_auc"])["name"],
               "lowest_auc_programme": min(big, key=lambda r: r["lopo_auc"])["name"]}
    # calibration shift: mean predicted vs observed dropout rate per programme
    gaps = np.array([r["lopo_mean_p"] - r["dropout_rate"] for r in big])
    summary["calib_gap_mean_abs"] = float(np.mean(np.abs(gaps)))
    summary["calib_gap_max_abs"] = float(np.max(np.abs(gaps)))
    summary["calib_gap_max_programme"] = big[int(np.argmax(np.abs(gaps)))]["name"]
    return {"pooled": pooled, "per_programme": per, "summary": summary}


def figure(out):
    r = out["sem1"]["XGBoost"]["per_programme"]
    r = sorted([x for x in r if x["n"] >= MIN_N], key=lambda x: x["lopo_auc"])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = np.arange(len(r))
    for i, x in enumerate(r):
        ax.plot([x["within_auc"], x["lopo_auc"]], [i, i], color=INK2, lw=0.8, zorder=1)
    ax.scatter([x["within_auc"] for x in r], y, s=26, color=C[0], label="Seen during training (CV)",
               zorder=2)
    ax.scatter([x["lopo_auc"] for x in r], y, s=26, color=C[1], marker="D",
               label="Held out (never seen)", zorder=3)
    ax.set_yticks(y, [f"{x['name']} (n={x['n']})" for x in r])
    ax.set_xlabel("AUC for Dropout vs. all other students, after the 1st semester (XGBoost)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fig_transfer")


def main():
    df = load_dataset()
    out = {}
    for stage in STAGES:
        out[stage] = {}
        for name in MODELS:
            out[stage][name] = run(df, stage, name)
            p, s = out[stage][name]["pooled"], out[stage][name]["summary"]
            print(f"{stage:10s} {name:20s} within AUC {p['within']['auc_dropout']:.3f} "
                  f"F1 {p['within']['f1']:.3f} | LOPO AUC {p['lopo']['auc_dropout']:.3f} "
                  f"F1 {p['lopo']['f1']:.3f} | diff CI {np.round(p['auc_diff_ci'], 3)} | "
                  f"median drop {s['median_auc_drop']:.3f}, worst {s['worst_auc_drop']:.3f} "
                  f"({s['worst_programme']}), calib gap {s['calib_gap_mean_abs']:.3f} | "
                  f"excl171: within {p['within_excl171']['auc_dropout']:.3f} "
                  f"LOPO {p['lopo_excl171']['auc_dropout']:.3f} CI {np.round(p['auc_diff_ci_excl171'], 3)}",
                  flush=True)
    json.dump(out, open(RES / "transfer.json", "w"), indent=1)
    figure(out)


if __name__ == "__main__":
    main()
