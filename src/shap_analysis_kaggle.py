# ============================================================
# Sub-RQ 2: SHAP Explainability Analysis
# Run on Kaggle with GPU or CPU accelerator
#
# Dataset: UCI Predict Students' Dropout and Academic Success
# https://www.kaggle.com/datasets/thedevastator/
#          higher-education-predictors-of-student-retention
#
# Note: SHAP TreeExplainer does not support sklearn's
# GradientBoostingClassifier for multiclass problems.
# We use XGBoost (identical hyperparameters) as a proxy.
# XGBoost achieves comparable macro F1 (0.711 vs 0.705).
# ============================================================

!pip install xgboost shap -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

SEED = 42

# ── 1. Load data ──────────────────────────────────────────────
# Update path to match your Kaggle dataset location.
# Option A (official UCI dataset on Kaggle):
#   path = "/kaggle/input/higher-education-predictors-of-student-retention/dataset.csv"
# Option B (manually uploaded):
path = "/kaggle/input/datasets/glebbabichev/education/Dropout_Academic Success - Sheet1.csv"

df = pd.read_csv(path)

le = LabelEncoder()
df["Target"] = le.fit_transform(df["Target"])
class_names = list(le.classes_)  # ['Dropout', 'Enrolled', 'Graduate']
print("Classes:", class_names)
print("Dataset shape:", df.shape)

X = df.drop("Target", axis=1)
y = df["Target"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

# ── 2. Train XGBoost ──────────────────────────────────────────
clf = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    objective="multi:softprob", num_class=3,
    random_state=SEED, eval_metric="mlogloss"
)
clf.fit(X_train_s, y_train)
acc = (clf.predict(X_test_s) == y_test.values).mean()
print(f"XGBoost trained. Test accuracy: {acc:.4f}")

# ── 3. SHAP values ────────────────────────────────────────────
explainer = shap.TreeExplainer(clf)
shap_values_obj = explainer(X_test_s)
# Shape: (samples, features, classes)
shap_values_list = [shap_values_obj.values[:, :, i] for i in range(3)]
print(f"SHAP computed. Shape per class: {shap_values_list[0].shape}")

# ── 4. fig16: Beeswarm — one figure per class ─────────────────
for i, cls in enumerate(class_names):
    fig, ax = plt.subplots(figsize=(8, 9))
    plt.sca(ax)
    class_exp = shap.Explanation(
        values=shap_values_list[i],
        data=X_test_s.values,
        feature_names=feature_names
    )
    shap.plots.beeswarm(class_exp, max_display=15, show=False)
    ax.set_title(f"Class: {cls}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = f"fig16_shap_beeswarm_{cls.lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

# ── 5. fig17: Global importance bar ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
shap_abs_mean = np.mean([np.abs(sv) for sv in shap_values_list], axis=0).mean(axis=0)
feat_imp = pd.Series(shap_abs_mean, index=feature_names).sort_values(ascending=True).tail(15)
colors = ["#1F4E79" if val > feat_imp.median() else "#9DC3E6" for val in feat_imp]
bars = ax.barh(feat_imp.index, feat_imp.values, color=colors)
ax.set_xlabel("Mean |SHAP Value| (Impact on model output)", fontsize=11)
ax.set_title("Global SHAP Feature Importance (Top 15)\nAveraged Across All Classes",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, feat_imp.values):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig17_shap_global_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig17_shap_global_importance.png")

# ── 6. fig18: Waterfall — one figure per class ────────────────
y_pred = clf.predict(X_test_s)
for i, cls_name in enumerate(class_names):
    correct = np.where((y_test.values == i) & (y_pred == i))[0]
    sample_idx = correct[0] if len(correct) > 0 else np.where(y_test.values == i)[0][0]
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.sca(ax)
    individual_exp = shap.Explanation(
        values=shap_values_list[i][sample_idx],
        base_values=explainer.expected_value[i],
        data=X_test_s.iloc[sample_idx].values,
        feature_names=feature_names
    )
    shap.plots.waterfall(individual_exp, max_display=12, show=False)
    ax.set_title(f"True & Predicted: {cls_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"fig18_shap_waterfall_{cls_name.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")

# ── 7. Top 10 features per class (console) ───────────────────
print("\n=== Top 10 SHAP Features per Class ===")
for i, cls in enumerate(class_names):
    mean_shap = np.abs(shap_values_list[i]).mean(axis=0)
    top_idx = np.argsort(mean_shap)[::-1][:10]
    print(f"\n{cls}:")
    for j in top_idx:
        print(f"  {feature_names[j]}: {mean_shap[j]:.4f}")

print("\nDone! Output files in /kaggle/working/:")
print("  fig16_shap_beeswarm_dropout/enrolled/graduate.png")
print("  fig17_shap_global_importance.png")
print("  fig18_shap_waterfall_dropout/enrolled/graduate.png")
