"""Turning risk scores into decisions. Shared by both datasets, so the UCI and
UPV numbers in the replication table are computed by the same code."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from common.models import SEED

CAPS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def topk_metrics(p, is_drop, cap):
    """Precision and reach of a list of the top `cap` share of students by risk."""
    n = len(p)
    k = int(round(cap * n))
    flagged = np.zeros(n, bool)
    flagged[np.argsort(-p, kind="stable")[:k]] = True
    tp = (flagged & is_drop).sum()
    return tp / k, tp / is_drop.sum()


def group_thresholds(p, is_drop, groups: dict, attr: str, cap=0.20, n_splits=5):
    """Single global threshold vs per-group thresholds that match the global
    recall inside each group. Cross-fitted: thresholds are set on four folds of
    students and applied to the fifth."""
    d = np.asarray(is_drop, bool)
    g = np.full(len(p), -1)
    for gi, mask in enumerate(groups.values()):
        g[mask] = gi
    strat = d.astype(int) * 10 + g
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    flag_single = np.zeros(len(p), bool)
    flag_group = np.zeros(len(p), bool)
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
                         "dropout_rate": float(d[m].mean()),
                         "recall": float((flag & d & m).sum() / (d & m).sum()),
                         "fpr": float((flag & ~d & m).sum() / (~d & m).sum()),
                         "alerts": int((flag & m).sum())})
        rows.append({"policy": name, "group": "All", "n": len(p),
                     "recall": float((flag & d).sum() / d.sum()),
                     "fpr": float((flag & ~d).sum() / (~d).sum()),
                     "alerts": int(flag.sum()),
                     "precision": float((flag & d).sum() / flag.sum())})
    return {"attribute": attr, "capacity": cap, "rows": rows}
