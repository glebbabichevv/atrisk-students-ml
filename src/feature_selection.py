Sub-RQ 3: How does feature selection affect model performance?
Methods: RFE (Recursive Feature Elimination), SelectKBest (ANOVA F-score), Variance Threshold
Best model: Gradient Boosting (multiclass setting)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_selection import RFE, SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/student_atrisk/data/dataset.csv", sep=",")
target_map = {"Graduate": 2, "Dropout": 0, "Enrolled": 1}
df["Target"] = df["Target"].map(target_map)

X = df.drop("Target", axis=1)
y = df["Target"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

def evaluate(X_tr, X_te, y_tr, y_te):
    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5, random_state=SEED
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)
    f1  = f1_score(y_te, y_pred, average="macro")
    auc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
    acc = accuracy_score(y_te, y_pred)
    return {"f1": round(f1, 4), "auc": round(auc, 4), "acc": round(acc, 4), "n_features": X_tr.shape[1]}

results = {}

# ── Baseline: all 36 features ──────────────────────────────────────────────────
results["Baseline (36)"] = evaluate(X_train_s, X_test_s, y_train, y_test)
print(f"Baseline (36 features): F1={results['Baseline (36)']['f1']}, AUC={results['Baseline (36)']['auc']}")

# ── Variance Threshold ─────────────────────────────────────────────────────────
vt = VarianceThreshold(threshold=0.01)
X_tr_vt = vt.fit_transform(X_train_s)
X_te_vt = vt.transform(X_test_s)
results[f"Variance Threshold ({X_tr_vt.shape[1]})"] = evaluate(X_tr_vt, X_te_vt, y_train, y_test)
print(f"Variance Threshold ({X_tr_vt.shape[1]} features): F1={results[f'Variance Threshold ({X_tr_vt.shape[1]})']['f1']}")

# ── SelectKBest (ANOVA F-score) ────────────────────────────────────────────────
for k in [10, 15, 20, 25, 30]:
    skb = SelectKBest(f_classif, k=k)
    X_tr_k = skb.fit_transform(X_train_s, y_train)
    X_te_k = skb.transform(X_test_s)
    label = f"SelectKBest (k={k})"
    results[label] = evaluate(X_tr_k, X_te_k, y_train, y_test)
    print(f"SelectKBest k={k}: F1={results[label]['f1']}, AUC={results[label]['auc']}")

    # Save top features for k=20
    if k == 20:
        scores = skb.scores_
        selected_idx = skb.get_support(indices=True)
        top_features_skb = [(feature_names[i], round(scores[i], 2)) for i in selected_idx]
        top_features_skb.sort(key=lambda x: -x[1])

# ── RFE with Random Forest ─────────────────────────────────────────────────────
rf_base = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
for n in [10, 15, 20, 25]:
    rfe = RFE(estimator=rf_base, n_features_to_select=n, step=2)
    X_tr_rfe = rfe.fit_transform(X_train_s, y_train)
    X_te_rfe = rfe.transform(X_test_s)
    label = f"RFE (n={n})"
    results[label] = evaluate(X_tr_rfe, X_te_rfe, y_train, y_test)
    print(f"RFE n={n}: F1={results[label]['f1']}, AUC={results[label]['auc']}")

    if n == 20:
        selected_rfe = [feature_names[i] for i, s in enumerate(rfe.support_) if s]

# ── Save results ───────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results).T.reset_index()
df_res.columns = ["method", "f1", "auc", "acc", "n_features"]
df_res.to_csv("/home/claude/student_atrisk/results/metrics/metrics_feature_selection.csv", index=False)
print("\nResults saved.")
print(df_res.to_string(index=False))

# ── Figure: Feature Selection Impact ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Sub-RQ 3: Impact of Feature Selection on Model Performance\n(Gradient Boosting, Multiclass Setting)",
             fontsize=13, fontweight="bold", y=1.02)

# Left: F1 by method
colors = []
labels_plot = []
f1_vals = []
auc_vals = []

for k, v in results.items():
    labels_plot.append(k)
    f1_vals.append(v["f1"])
    auc_vals.append(v["auc"])
    if "Baseline" in k:
        colors.append("#1F4E79")
    elif "Variance" in k:
        colors.append("#C55A11")
    elif "SelectKBest" in k:
        colors.append("#2E75B6")
    else:
        colors.append("#70AD47")

x = np.arange(len(labels_plot))
bars = axes[0].bar(x, f1_vals, color=colors, width=0.65, edgecolor="white", linewidth=0.5)
axes[0].axhline(y=results["Baseline (36)"]["f1"], color="#1F4E79", linestyle="--", linewidth=1.5, alpha=0.7)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels_plot, rotation=35, ha="right", fontsize=9)
axes[0].set_ylabel("Macro F1-Score", fontsize=11)
axes[0].set_title("F1-Score by Feature Selection Method", fontsize=11, fontweight="bold")
axes[0].set_ylim(0.60, 0.75)
axes[0].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, f1_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

# Right: F1 vs number of features (trend)
n_feats_skb = [results[f"SelectKBest (k={k})"]["n_features"] for k in [10,15,20,25,30]]
f1_skb       = [results[f"SelectKBest (k={k})"]["f1"]        for k in [10,15,20,25,30]]
n_feats_rfe  = [results[f"RFE (n={n})"]["n_features"]        for n in [10,15,20,25]]
f1_rfe       = [results[f"RFE (n={n})"]["f1"]                for n in [10,15,20,25]]

axes[1].plot(n_feats_skb, f1_skb, "o-", color="#2E75B6", linewidth=2, markersize=7, label="SelectKBest (ANOVA F)")
axes[1].plot(n_feats_rfe, f1_rfe, "s--", color="#70AD47", linewidth=2, markersize=7, label="RFE (Random Forest)")
axes[1].axhline(y=results["Baseline (36)"]["f1"], color="#1F4E79", linestyle=":", linewidth=2, label="Baseline (36 features)")
axes[1].set_xlabel("Number of Selected Features", fontsize=11)
axes[1].set_ylabel("Macro F1-Score", fontsize=11)
axes[1].set_title("F1-Score vs. Number of Features", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0.60, 0.75)

plt.tight_layout()
plt.savefig("/home/claude/student_atrisk/results/figures/fig14_feature_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure saved: fig14_feature_selection.png")

# ── Figure: Top features comparison ───────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 6))
top_feats = top_features_skb[:15]
feat_labels = [f[0] for f in top_feats]
feat_scores = [f[1] for f in top_feats]
bar_colors = ["#1F4E79" if i < 5 else "#2E75B6" if i < 10 else "#9DC3E6" for i in range(len(top_feats))]
bars = ax.barh(range(len(top_feats)), feat_scores, color=bar_colors, edgecolor="white")
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(feat_labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("ANOVA F-Score", fontsize=11)
ax.set_title("Top 15 Features by SelectKBest (ANOVA F-score)\nMulticlass Setting", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, feat_scores):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("/home/claude/student_atrisk/results/figures/fig15_selectkbest_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure saved: fig15_selectkbest_features.png")
print("\nTop 20 features (SelectKBest):")
for f, s in top_features_skb:
    print(f"  {f}: {s}")
print("\nTop 20 features (RFE):")
for f in selected_rfe:
    print(f"  {f}")
