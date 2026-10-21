"""
UCI step 1: benchmark of 8 models in 4 settings.

Design
------
* Repeated stratified k-fold cross-validation (5 folds x 5 repeats = 25 fits per
  model and setting). Preprocessing (scaling) lives inside each model pipeline,
  so it is fitted on training folds only.
* Settings
    binary      Dropout vs Graduate (Enrolled rows removed)
    multiclass  Dropout / Enrolled / Graduate, all features
    sem1        multiclass, second-semester features removed
    enrol       multiclass, both semesters removed (information at enrolment only)
* Pairwise model comparison: corrected resampled t-test
  (Nadeau & Bengio, 2003), which accounts for overlap between training sets.
* Out-of-fold predictions from the first repeat are kept for confusion
  matrices, calibration and subgroup (fairness) analysis.

Outputs go to results/uci/. Run:  python run_all.py --only uci.benchmark
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.models import SEED, corrected_ttest, get_models  # noqa: E402
from common.paths import UCI_RESULTS as OUT  # noqa: E402
from uci.data import CLASS_NAMES, load_dataset, setting_xy  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
N_SPLITS, N_REPEATS = 5, 5
N_JOBS = -1


def score(y, pred, prob, n_classes):
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="macro", zero_division=0),
        "recall": recall_score(y, pred, average="macro", zero_division=0),
        "f1": f1_score(y, pred, average="macro", zero_division=0),
    }
    if n_classes == 2:
        out["auc"] = roc_auc_score(y, prob[:, 1])
    else:
        out["auc"] = roc_auc_score(y, prob, multi_class="ovr", average="macro")
        per = f1_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0)
        out.update({f"f1_{c.lower()}": v for c, v in zip(CLASS_NAMES, per)})
    # Operational target: is this student a dropout? (class 0 in every setting)
    is_drop = (y == 0).astype(int)
    out["auc_dropout"] = roc_auc_score(is_drop, prob[:, 0])
    out["recall_dropout"] = recall_score(is_drop, (pred == 0).astype(int))
    out["brier_dropout"] = brier_score_loss(is_drop, prob[:, 0])
    return out


def fit_fold(name, model, X, y, tr, te, n_classes):
    m = clone(model)
    m.fit(X[tr], y[tr])
    prob = m.predict_proba(X[te])
    pred = prob.argmax(1)
    return name, score(y[te], pred, prob, n_classes), te, prob


def run_setting(setting: str, df: pd.DataFrame):
    X, y, feats = setting_xy(df, setting)
    n_classes = len(np.unique(y))
    models = get_models(n_classes, feats)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=SEED)
    splits = list(cv.split(X, y))
    t0 = time.time()
    jobs = (delayed(fit_fold)(n, m, X, y, tr, te, n_classes)
            for n, m in models.items() for tr, te in splits)
    res = Parallel(n_jobs=N_JOBS)(jobs)
    print(f"[{setting}] {len(res)} fits in {time.time() - t0:.0f}s")

    folds = {n: [] for n in models}
    oof = {n: np.zeros((len(y), n_classes)) for n in models}
    counts = {n: 0 for n in models}
    for name, sc, te, prob in res:
        folds[name].append(sc)
        if counts[name] < N_SPLITS:          # first repeat only -> one OOF pass
            oof[name][te] = prob
        counts[name] += 1

    rows = []
    for name, lst in folds.items():
        f = pd.DataFrame(lst)
        row = {"setting": setting, "model": name}
        for c in f.columns:
            row[c] = f[c].mean()
            row[c + "_sd"] = f[c].std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("f1", ascending=False)

    # Pairwise tests against the best model (by macro F1 and by AUC)
    n_test = len(y) // N_SPLITS
    n_train = len(y) - n_test
    tests = []
    for metric in ["f1", "auc"]:
        best = summary.sort_values(metric, ascending=False).iloc[0]["model"]
        for name in models:
            if name == best:
                continue
            a = [s[metric] for s in folds[best]]
            b = [s[metric] for s in folds[name]]
            t, p = corrected_ttest(a, b, n_train, n_test)
            tests.append({"setting": setting, "metric": metric, "best": best,
                          "other": name, "diff": np.mean(a) - np.mean(b),
                          "t": t, "p": p})
    fold_scores = {n: [{k: float(v) for k, v in s.items()} for s in lst]
                   for n, lst in folds.items()}
    return summary, pd.DataFrame(tests), oof, y, feats, fold_scores


def main():
    df = load_dataset()
    all_sum, all_tests, store = [], [], {}
    for setting in ["binary", "multiclass", "sem1", "enrol"]:
        s, t, oof, y, feats, fs = run_setting(setting, df)
        all_sum.append(s)
        all_tests.append(t)
        np.savez_compressed(OUT / f"oof_{setting}.npz", y=y,
                            **{k.replace(" ", "_"): v for k, v in oof.items()})
        store[setting] = fs
    pd.concat(all_sum).to_csv(OUT / "cv_summary.csv", index=False)
    pd.concat(all_tests).to_csv(OUT / "cv_tests.csv", index=False)
    with open(OUT / "cv_fold_scores.json", "w") as fh:
        json.dump(store, fh)
    print(pd.concat(all_sum)[["setting", "model", "accuracy", "f1", "f1_sd",
                              "auc", "auc_dropout"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
