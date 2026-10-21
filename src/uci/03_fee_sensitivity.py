"""
Sensitivity analysis: the two fee-status variables (Debtor, Tuition fees up to
date) are strong predictors, but the dataset documentation does not say when they
were recorded. If they were updated after a student stopped attending, they
would leak the outcome. We therefore repeat the three-class analysis at each
prediction point without them.

UCI step 3.
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
from uci.data import load_dataset, setting_xy  # noqa: E402
MODELS = ["XGBoost", "Logistic Regression"]


def fit(model, X, y, tr, te):
    m = clone(model).fit(X[tr], y[tr])
    return te, m.predict_proba(X[te])


def list_metrics(p, d, cap):
    k = int(round(cap * len(p)))
    top = np.argsort(-p, kind="stable")[:k]
    return d[top].mean(), d[top].sum() / d.sum()


def main():
    df = load_dataset()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    out = []
    for stage in ["enrol", "sem1", "multiclass"]:
        for variant in ["", "_nofee"]:
            X, y, feats = setting_xy(df, stage + variant)
            splits = list(cv.split(X, y))
            for name in MODELS:
                model = get_models(3, feats)[name]
                res = Parallel(n_jobs=-1)(delayed(fit)(model, X, y, tr, te)
                                          for tr, te in splits)
                f1s, aucs = [], []
                oof = np.zeros((len(y), 3))
                for i, (te, prob) in enumerate(res):
                    f1s.append(f1_score(y[te], prob.argmax(1), average="macro"))
                    aucs.append(roc_auc_score(y[te] == 0, prob[:, 0]))
                    if i < 5:
                        oof[te] = prob
                d = y == 0
                p10, _ = list_metrics(oof[:, 0], d, 0.10)
                _, r20 = list_metrics(oof[:, 0], d, 0.20)
                row = {"stage": stage, "fees": variant == "", "model": name,
                       "n_features": len(feats),
                       "f1": float(np.mean(f1s)), "f1_sd": float(np.std(f1s, ddof=1)),
                       "auc_dropout": float(np.mean(aucs)),
                       "auc_dropout_sd": float(np.std(aucs, ddof=1)),
                       "precision_at_10": float(p10), "recall_at_20": float(r20)}
                out.append(row)
                if name == "XGBoost":
                    np.save(RES / f"oof_xgb_{stage}{variant or '_fees'}.npy", oof[:, 0])
                print({k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in row.items()}, flush=True)
    json.dump(out, open(RES / "sensitivity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
