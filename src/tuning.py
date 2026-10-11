"""
Nested cross-validation: does hyperparameter tuning change the conclusions?

Outer loop: stratified 5-fold CV repeated twice (10 test folds), three-class
setting with all features. Inner loop: randomised search (3-fold CV, macro F1)
on the outer training part only. The tuned model is compared with the fixed
configuration from experiments.py on the same outer folds, using the corrected
resampled t-test.

Run:  python src/tuning.py      (about 30-40 minutes on 2 cores)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from data import load_dataset, setting_xy  # noqa: E402
from experiments import SEED, corrected_ttest, get_models  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

SPACES = {
    "Logistic Regression": ({"logisticregression__C": loguniform(1e-3, 1e2)}, 20),
    "Random Forest": ({"n_estimators": randint(200, 600),
                       "max_features": uniform(0.1, 0.6),
                       "min_samples_leaf": randint(1, 10),
                       "max_depth": [None, 8, 12, 20]}, 12),
    "XGBoost": ({"n_estimators": randint(100, 600),
                 "learning_rate": loguniform(0.01, 0.2),
                 "max_depth": randint(2, 8),
                 "min_child_weight": loguniform(0.5, 10),
                 "subsample": uniform(0.6, 0.4),
                 "colsample_bytree": uniform(0.4, 0.6),
                 "reg_lambda": loguniform(0.1, 10)}, 20),
    "LightGBM": ({"n_estimators": randint(100, 600),
                  "learning_rate": loguniform(0.01, 0.2),
                  "num_leaves": randint(8, 64),
                  "min_child_samples": randint(5, 60),
                  "subsample": uniform(0.6, 0.4),
                  "colsample_bytree": uniform(0.4, 0.6),
                  "reg_lambda": loguniform(0.01, 10)}, 20),
}


def main():
    df = load_dataset()
    X, y, feats = setting_xy(df, "multiclass")
    outer = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
                                         random_state=SEED + 1).split(X, y))
    out = {}
    for name, (space, n_iter) in SPACES.items():
        t0 = time.time()
        base = get_models(3, feats)[name]
        rows = []
        for k, (tr, te) in enumerate(outer):
            fixed = clone(base).fit(X[tr], y[tr])
            pf = fixed.predict_proba(X[te])
            search = RandomizedSearchCV(
                clone(base), space, n_iter=n_iter, scoring="f1_macro",
                cv=StratifiedKFold(3, shuffle=True, random_state=SEED + k),
                random_state=SEED + k, n_jobs=-1, refit=True)
            search.fit(X[tr], y[tr])
            pt = search.predict_proba(X[te])
            rows.append({
                "f1_fixed": f1_score(y[te], pf.argmax(1), average="macro"),
                "f1_tuned": f1_score(y[te], pt.argmax(1), average="macro"),
                "auc_fixed": roc_auc_score(y[te], pf, multi_class="ovr"),
                "auc_tuned": roc_auc_score(y[te], pt, multi_class="ovr"),
                "best_params": {p: (float(v) if isinstance(v, (int, float, np.floating,
                                                               np.integer)) else v)
                                for p, v in search.best_params_.items()},
            })
        n_test = len(y) // 5
        res = {}
        for metric in ["f1", "auc"]:
            a = [r[f"{metric}_tuned"] for r in rows]
            b = [r[f"{metric}_fixed"] for r in rows]
            t, p = corrected_ttest(a, b, len(y) - n_test, n_test)
            res[metric] = {"tuned": float(np.mean(a)), "tuned_sd": float(np.std(a, ddof=1)),
                           "fixed": float(np.mean(b)), "fixed_sd": float(np.std(b, ddof=1)),
                           "diff": float(np.mean(a) - np.mean(b)), "p": p}
        res["folds"] = rows
        out[name] = res
        print(f"{name}: F1 fixed {res['f1']['fixed']:.4f} tuned {res['f1']['tuned']:.4f} "
              f"(p={res['f1']['p']:.3f}); AUC fixed {res['auc']['fixed']:.4f} tuned "
              f"{res['auc']['tuned']:.4f} (p={res['auc']['p']:.3f})  [{time.time()-t0:.0f}s]",
              flush=True)
        with open(RES / "tuning.json", "w") as fh:
            json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
