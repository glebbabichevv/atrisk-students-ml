"""
UPV step 1: external validation of the UCI findings on a second university.

Data and prediction points: see src/upv/data.py. The data have no graduation or
delayed label, so the task is binary: dropout vs. not.

  1. repeated 5x5 CV, 8 models, 5 prediction points; resit check (lms1_only)
  2. calibration and fixed-size alert lists (out-of-fold)
  3. temporal validation: 2018 -> 2021 and 2018 + 2021 -> 2022
  4. unseen programmes: 5-fold CV grouped by degree
  5. subgroup recall in a top-20% list, single vs per-group thresholds
  6. top features (XGBoost gain)

Outputs: results/upv/upv_results.json (about 10-40 minutes)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.models import MODEL_NAMES, SEED, corrected_ttest, get_models  # noqa: E402
from common.paths import UPV_RESULTS as RES  # noqa: E402
from common.decision import CAPS, group_thresholds, topk_metrics  # noqa: E402
from upv.data import MAIN_STAGES, design, load_upv  # noqa: E402

RES.mkdir(parents=True, exist_ok=True)


# ── evaluation ───────────────────────────────────────────────────────────────
def ece(p, y, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(y, p):
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "f1_dropout": f1_score(y, p >= 0.5)}


def fit(model, X, y, tr, te):
    m = clone(model).fit(X[tr], y[tr])
    return te, m.predict_proba(X[te])[:, 1]


def cv_stage(df, stage, models):
    X, y, feats = design(df, stage)
    strat = y * 10 + (df["year"].to_numpy() - 2018)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                          random_state=SEED).split(X, strat))
    mdl = get_models(2, feats)
    folds, oof = {}, {}
    for name in models:
        res = Parallel(n_jobs=-1)(delayed(fit)(mdl[name], X, y, tr, te) for tr, te in splits)
        folds[name] = [metrics(y[te], p) for te, p in res]
        o = np.zeros(len(y))
        for te, p in res[:5]:
            o[te] = p
        oof[name] = o
        f = pd.DataFrame(folds[name])
        print(f"  {stage:9s} {name:20s} AUC {f.auc.mean():.3f}±{f.auc.std():.3f}  "
              f"AP {f.ap.mean():.3f}  Brier {f.brier.mean():.3f}", flush=True)
    n_test = len(y) // 5
    best = max(models, key=lambda m: np.mean([s["auc"] for s in folds[m]]))
    tests = {}
    for name in models:
        if name != best:
            t, p = corrected_ttest([s["auc"] for s in folds[best]], [s["auc"] for s in folds[name]],
                                   len(y) - n_test, n_test)
            tests[name] = p
    summary = {n: {k: float(np.mean([s[k] for s in folds[n]])) for k in folds[n][0]}
               | {k + "_sd": float(np.std([s[k] for s in folds[n]], ddof=1)) for k in folds[n][0]}
               for n in models}
    return {"summary": summary, "best_auc": best, "p_vs_best_auc": tests, "n": int(len(y)),
            "n_features": len(feats)}, oof, y


PAIR = ["XGBoost", "Logistic Regression"]


def predict(df, stage, name, tr, te):
    X, y, feats = design(df, stage)
    m = clone(get_models(2, feats)[name]).fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1]


def main():
    df = load_upv()
    df.to_csv(RES / "upv_students.csv")
    yall = df["dropout"].to_numpy()
    yr = df["year"].to_numpy()
    print("students:", len(df), df.groupby("year")["dropout"].agg(["size", "mean"]).round(3).to_dict())
    out = {"n_by_year": df.groupby("year").size().to_dict(),
           "dropout_rate_by_year": df.groupby("year")["dropout"].mean().round(4).to_dict(),
           "dropout_rate": float(yall.mean())}

    # 1. repeated CV: all models at the five prediction points, then the resit check
    out["cv"], oofs = {}, {}
    for stage in MAIN_STAGES:
        out["cv"][stage], oofs[stage], _ = cv_stage(df, stage, MODEL_NAMES)
    out["cv"]["lms1_only"], oofs["lms1_only"], _ = cv_stage(df, "lms1_only", PAIR)

    # 2. calibration and fixed-size alert lists (out-of-fold, first CV repeat)
    out["alerts"], out["calibration"] = {}, {}
    for stage in MAIN_STAGES:
        out["alerts"][stage], out["calibration"][stage] = {}, {}
        for name in PAIR:
            p = oofs[stage][name]
            out["alerts"][stage][name] = [
                dict(zip(["capacity", "precision", "recall"], [c, *map(float, topk_metrics(p, yall == 1, c))]))
                for c in CAPS]
            out["calibration"][stage][name] = {"ece": ece(p, yall), "brier": float(np.mean((p - yall) ** 2))}

    # 3. temporal validation: train on earlier cohorts, test on a later one.
    #    2018 -> 2021 has complete labels; 2022 labels may still be incomplete
    #    under the two-year dropout rule, so the two splits are reported side by side.
    out["temporal"] = {}
    for split, tr_mask, te_mask in [("2018_to_2021", yr == 2018, yr == 2021),
                                    ("2018_2021_to_2022", yr < 2022, yr == 2022)]:
        tr, te = np.where(tr_mask)[0], np.where(te_mask)[0]
        out["temporal"][split] = {}
        for stage in MAIN_STAGES:
            out["temporal"][split][stage] = {}
            for name in PAIR:
                p = predict(df, stage, name, tr, te)
                yt = yall[te]
                r = metrics(yt, p) | {"ece": ece(p, yt), "mean_pred": float(p.mean()),
                                      "observed": float(yt.mean())}
                r["top20_precision"], r["top20_recall"] = map(float, topk_metrics(p, yt == 1, 0.20))
                out["temporal"][split][stage][name] = {k: float(v) for k, v in r.items()}
                print(f"  temporal {split:18s} {stage:9s} {name:20s} AUC {r['auc']:.3f}  "
                      f"mean pred {r['mean_pred']:.3f} vs observed {r['observed']:.3f}", flush=True)

    # 4. unseen programmes: 5-fold CV grouped by degree, degree dummies removed
    out["grouped_degree"] = {}
    groups = df["cat_tit_hash"].to_numpy()
    for stage in ["sem1", "sem1_lms", "year"]:
        X, y, feats = design(df.drop(columns=["cat_tit_hash"]), stage)
        out["grouped_degree"][stage] = {}
        for name in PAIR:
            mdl = get_models(2, feats)[name]
            res = Parallel(n_jobs=-1)(delayed(fit)(mdl, X, y, tr, te)
                                      for tr, te in GroupKFold(5).split(X, y, groups))
            o = np.zeros(len(y))
            for te, p in res:
                o[te] = p
            ref = out["cv"][stage]["summary"][name]["auc"]
            out["grouped_degree"][stage][name] = {"auc": float(roc_auc_score(y, o)), "ece": ece(o, y),
                                                  "cv_auc": ref}
            print(f"  grouped   {stage:9s} {name:20s} AUC {roc_auc_score(y, o):.3f} (CV {ref:.3f})", flush=True)

    # 5. subgroup recall in a top-20% list, single vs per-group thresholds (cross-fitted)
    part = df["part_time"].to_numpy() == 1
    nap = df["cat_tipo_ingreso"].to_numpy() == "NAP"
    out["thresholds"] = {}
    for stage in ["sem1_lms", "year"]:
        p = oofs[stage]["XGBoost"]
        out["thresholds"][stage] = [
            group_thresholds(p, yall == 1, {"Full-time": ~part, "Part-time": part}, "Study mode"),
            group_thresholds(p, yall == 1, {"Standard access": nap, "Other access routes": ~nap},
                             "Access route")]

    # 6. top features (XGBoost, first semester with LMS, gain importance)
    X, y, feats = design(df, "sem1_lms")
    xgb = clone(get_models(2, feats)["XGBoost"]).fit(X, y)
    imp = pd.Series(xgb.feature_importances_, index=feats).sort_values(ascending=False)
    out["top_features_sem1_lms"] = imp.head(15).round(4).to_dict()

    json.dump(out, open(RES / "upv_results.json", "w"), indent=1, default=float)
    for stage, lst in out["thresholds"].items():
        for t in lst:
            print(stage, t["attribute"], [(r["policy"], r["group"], round(r["recall"], 3))
                                          for r in t["rows"] if r["group"] != "All"])


if __name__ == "__main__":
    main()
