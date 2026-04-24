Model Training & Evaluation
6 models from the proposal: LR, DT, RF, SVM, GradientBoosting, MLP
Metrics: Accuracy, Precision, Recall, F1 (macro), AUC-ROC (OvR)
Bootstrap 95% CI for all metrics
"""
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
RANDOM_STATE = 42
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE,
            solver="lbfgs", C=1.0
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", solver="adam",
            max_iter=500, random_state=RANDOM_STATE,
            early_stopping=True, validation_fraction=0.1,
            learning_rate_init=0.001
        ),
    }


def compute_metrics(y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


def bootstrap_ci(y_true, y_pred, y_prob, n=N_BOOTSTRAP, alpha=0.05, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    n_expected = len(np.unique(y_true))
    boot_metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]}

    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < n_expected:
            continue
        try:
            m = compute_metrics(yt, yp, ypr)
        except Exception:
            continue
        for k, v in m.items():
            if not np.isnan(v):
                boot_metrics[k].append(v)

    ci = {}
    for k, vals in boot_metrics.items():
        if len(vals) == 0:
            ci[k] = {"mean": np.nan, "lower": np.nan, "upper": np.nan}
        else:
            arr = np.array(vals)
            ci[k] = {
                "mean": float(np.mean(arr)),
                "lower": float(np.percentile(arr, alpha / 2 * 100)),
                "upper": float(np.percentile(arr, (1 - alpha / 2) * 100)),
            }
    return ci


def train_and_evaluate(data, use_scaled=None):
    """
    use_scaled: list of model names that need scaled input.
                Defaults to LR, SVM, Neural Network.
    """
    if use_scaled is None:
        use_scaled = ["Logistic Regression", "SVM", "Neural Network"]

    X_train_raw = data["X_train"]
    X_test_raw  = data["X_test"]
    X_train_sc  = data["X_train_scaled"]
    X_test_sc   = data["X_test_scaled"]
    y_train     = data["y_train"]
    y_test      = data["y_test"]

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n▶ Training {name}...")
        t0 = time.time()

        X_tr = X_train_sc if name in use_scaled else X_train_raw
        X_te = X_test_sc  if name in use_scaled else X_test_raw

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)
        elapsed = time.time() - t0

        metrics = compute_metrics(y_test, y_pred, y_prob)
        print(f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
              f"AUC={metrics['auc_roc']:.4f}  [{elapsed:.1f}s]")

        print(f"  Computing bootstrap CI ({N_BOOTSTRAP} iterations)...")
        ci = bootstrap_ci(y_test, y_pred, y_prob)

        cm = confusion_matrix(y_test, y_pred)
        n_cls = len(np.unique(np.concatenate([y_train, y_test])))
        cls_names = data["class_names"] if "class_names" in data else CLASS_NAMES[:n_cls]
        n_cls = len(np.unique(y_test))
        report = classification_report(y_test, y_pred,
                                        target_names=CLASS_NAMES[:n_cls], output_dict=True)
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": metrics,
            "bootstrap_ci": ci,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "train_time_sec": elapsed,
        }

    return results


def save_metrics(results, tag="full"):
    rows = []
    for name, r in results.items():
        row = {"model": name}
        for metric, ci_vals in r["bootstrap_ci"].items():
            row[metric] = round(ci_vals["mean"], 4)
            row[f"{metric}_ci_low"] = round(ci_vals["lower"], 4)
            row[f"{metric}_ci_high"] = round(ci_vals["upper"], 4)
        row["train_time_sec"] = round(r["train_time_sec"], 2)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    path = METRICS_DIR / f"metrics_{tag}.csv"
    df.to_csv(path, index=False)
    print(f"\n✅ Metrics saved → {path}")

    # also save JSON with full CI info
    json_out = {}
    for name, r in results.items():
        json_out[name] = {
            "metrics": r["metrics"],
            "bootstrap_ci": r["bootstrap_ci"],
            "confusion_matrix": r["confusion_matrix"],
            "train_time_sec": r["train_time_sec"],
        }
    json_path = METRICS_DIR / f"metrics_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"✅ JSON saved → {json_path}")

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocess import load_and_preprocess

    data = load_and_preprocess()
    results = train_and_evaluate(data)
    df_metrics = save_metrics(results, tag="full")
    print("\n" + df_metrics[["model", "accuracy", "f1", "auc_roc"]].to_string(index=False))


def train_and_evaluate_binary(data):
    """Binary variant — uses binary metrics (F1, AUC standard)."""
    from sklearn.metrics import roc_auc_score as _auc

    X_train_raw = data["X_train"]
    X_test_raw  = data["X_test"]
    X_train_sc  = data["X_train_scaled"]
    X_test_sc   = data["X_test_scaled"]
    y_train     = data["y_train"]
    y_test      = data["y_test"]

    use_scaled = ["Logistic Regression", "SVM", "Neural Network"]

    # Binary-compatible models (same hyperparams)
    models = get_models()

    results = {}
    for name, model in models.items():
        print(f"\n▶ Training {name} [binary]...")
        t0 = time.time()

        X_tr = X_train_sc if name in use_scaled else X_train_raw
        X_te = X_test_sc  if name in use_scaled else X_test_raw

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]  # prob of class 1 (Graduate)
        elapsed = time.time() - t0

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = _auc(y_test, y_prob)
        print(f"  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  [{elapsed:.1f}s]")

        # Bootstrap CI
        print(f"  Computing bootstrap CI ({N_BOOTSTRAP} iterations)...")
        rng = np.random.default_rng(RANDOM_STATE)
        boot = {k: [] for k in ["accuracy","precision","recall","f1","auc_roc"]}
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, len(y_test), len(y_test))
            yt, yp, ypr = y_test[idx], y_pred[idx], y_prob[idx]
            if len(np.unique(yt)) < 2:
                continue
            boot["accuracy"].append(accuracy_score(yt, yp))
            boot["precision"].append(precision_score(yt, yp, zero_division=0))
            boot["recall"].append(recall_score(yt, yp, zero_division=0))
            boot["f1"].append(f1_score(yt, yp, zero_division=0))
            boot["auc_roc"].append(_auc(yt, ypr))

        ci = {}
        for k, vals in boot.items():
            arr = np.array(vals)
            ci[k] = {"mean": float(np.mean(arr)),
                     "lower": float(np.percentile(arr, 2.5)),
                     "upper": float(np.percentile(arr, 97.5))}

        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob_binary": y_prob,
            "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc},
            "bootstrap_ci": ci,
            "confusion_matrix": cm.tolist(),
            "train_time_sec": elapsed,
        }

    return results


def train_and_evaluate_binary(data):
    """Binary variant — uses binary metrics instead of macro OvR."""
    from sklearn.metrics import roc_auc_score as ras

    X_train_raw = data["X_train"]
    X_test_raw  = data["X_test"]
    X_train_sc  = data["X_train_scaled"]
    X_test_sc   = data["X_test_scaled"]
    y_train     = data["y_train"]
    y_test      = data["y_test"]
    use_scaled  = ["Logistic Regression", "SVM", "Neural Network"]

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n▶ Training {name} [binary]...")
        t0 = time.time()
        X_tr = X_train_sc if name in use_scaled else X_train_raw
        X_te = X_test_sc  if name in use_scaled else X_test_raw

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        elapsed = time.time() - t0

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = ras(y_test, y_prob)
        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}
        print(f"  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  [{elapsed:.1f}s]")

        # bootstrap CI
        print(f"  Bootstrap CI...")
        rng = np.random.default_rng(RANDOM_STATE)
        y_prob_2d = np.column_stack([1 - y_prob, y_prob])
        ci = bootstrap_ci(y_test, y_pred, y_prob_2d)

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                        target_names=["Dropout", "Graduate"],
                                        output_dict=True)
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob_2d,
            "metrics": metrics,
            "bootstrap_ci": ci,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "train_time_sec": elapsed,
        }

    return results
