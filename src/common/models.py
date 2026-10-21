"""Shared model zoo, seed and the corrected resampled t-test (used for both datasets)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM (RBF)",
               "Gradient Boosting", "XGBoost", "LightGBM", "MLP"]

# Integer-coded nominal variables of the UCI data. One-hot encoded for the models
# that need scaling (LR, SVM, MLP). The UPV design matrix is one-hot already, so
# none of its columns match this list.
NOMINAL = ["Marital status", "Application mode", "Course", "Previous qualification",
           "Nationality", "Mother's qualification", "Father's qualification",
           "Mother's occupation", "Father's occupation"]


def linear_preprocessor(feats):
    """One-hot encode nominal codes, standardise the rest (fitted per fold)."""
    nom = [i for i, f in enumerate(feats) if f in NOMINAL]
    num = [i for i, f in enumerate(feats) if f not in NOMINAL]
    return ColumnTransformer([
        ("nom", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=20, sparse_output=False), nom),
        ("num", StandardScaler(), num)])


def get_models(n_classes: int, feats: list[str]) -> dict:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    scaled = lambda m: make_pipeline(linear_preprocessor(feats), m)  # noqa: E731
    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=2000, C=1.0)),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5,
                                                random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                random_state=SEED, n_jobs=1),
        "SVM (RBF)": scaled(SVC(C=10, gamma="scale", probability=True,
                                random_state=SEED)),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200,
                                                        learning_rate=0.05,
                                                        max_depth=5,
                                                        random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=SEED, n_jobs=1, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   num_leaves=31, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, random_state=SEED,
                                   n_jobs=1, verbose=-1),
        "MLP": scaled(MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                                    early_stopping=True, max_iter=500,
                                    random_state=SEED)),
    }


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold scores."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)
