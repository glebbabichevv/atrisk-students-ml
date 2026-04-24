# Early Detection of At-Risk Students Using Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: UCI](https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-green.svg)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
[![Paper: arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2503.XXXXX)

Reproducible code for the action research paper:
> **"Early Detection of At-Risk Students Using Machine Learning and Artificial Intelligence Techniques"**
> Gleb Babichev, Tamos Education, Almaty, Kazakhstan, 2026.

---

## Dataset

**UCI Predict Students' Dropout and Academic Success**
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- Citation: Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data*, 7(11), 146. https://doi.org/10.3390/data7110146
- License: CC BY 4.0
- 4,424 students · 36 features · 3 classes (Dropout / Enrolled / Graduate)

**Download the dataset before running:**
```bash
# Option A: UCI direct
wget https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip
unzip *.zip && mv *.csv data/dataset.csv

# Option B: Kaggle CLI
kaggle datasets download thedevastator/higher-education-predictors-of-student-retention
unzip *.zip && mv *.csv data/dataset.csv
```

> **Note:** The dataset file is NOT included in this repository (CC BY 4.0 license requires attribution; download from the source above).

---

## Results Summary

### Binary Classification (Dropout vs. Graduate, n=3,630)
| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.915 | 0.909 | **0.952** |
| Random Forest | 0.912 | **0.905** | 0.955 |
| Gradient Boosting | 0.904 | 0.896 | 0.955 |
| Neural Network | 0.903 | 0.894 | 0.952 |
| SVM | 0.900 | 0.893 | 0.944 |
| Decision Tree | 0.883 | 0.875 | 0.898 |

### Multiclass (3 classes, n=4,424)
| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| **Gradient Boosting** | **0.771** | **0.705** | **0.890** |
| Random Forest | 0.771 | 0.694 | 0.889 |
| Logistic Regression | 0.768 | 0.681 | 0.877 |
| Neural Network | 0.750 | 0.670 | 0.860 |
| SVM | 0.719 | 0.643 | 0.845 |
| Decision Tree | 0.711 | 0.647 | 0.801 |

All metrics: bootstrap means (B=1,000), 95% CI reported in paper.

### Key Findings
- **Outcome uncertainty bias**: Including Enrolled students drops macro F1 by ~20pp
- **Early prediction**: First-semester-only models retain 87–92% of full-model F1
- **Feature selection**: 15–20 features retain 98–99% of baseline performance
- **Top SHAP predictor**: Curricular units 2nd sem (approved), mean |SHAP| = 0.647

---

## Repository Structure

```
├── src/
│   ├── preprocess.py            # Data loading, cleaning, train/test split
│   ├── train.py                 # Binary + multiclass + early prediction
│   ├── train_binary_patch.py    # Standalone binary experiment
│   ├── visualize.py             # Figures fig07–fig13
│   ├── feature_selection.py     # Sub-RQ 3: RFE + SelectKBest (fig14–fig15)
│   └── shap_analysis_kaggle.py  # Sub-RQ 2: SHAP via XGBoost — run on Kaggle
├── figures/                     # Generated PNG figures (fig07–fig18)
├── results/
│   └── metrics/                 # CSV/JSON metrics for all experiments
├── paper/
│   ├── main.tex                 # LaTeX source
│   └── references.bib           # 41 BibTeX references
├── data/                        # Empty — download dataset here (see above)
├── run_all.py                   # Master script: runs all local experiments
├── requirements.txt
├── KAGGLE_SHAP_GUIDE.md         # Step-by-step Kaggle setup for SHAP
└── LICENSE
```

---

## Quickstart

```bash
git clone https://github.com/glebbabichevv/atrisk-students-ml.git
cd atrisk-students-ml
pip install -r requirements.txt

# 1. Download dataset (see above) → data/dataset.csv
mkdir -p data

# 2. Run all local experiments (binary, multiclass, early, feature selection)
python run_all.py

# Figures → figures/   Metrics → results/metrics/
```

### SHAP Analysis (Kaggle only)
SHAP TreeExplainer requires XGBoost for multiclass. See [`KAGGLE_SHAP_GUIDE.md`](KAGGLE_SHAP_GUIDE.md) for step-by-step setup. Script: `src/shap_analysis_kaggle.py`.

---

## Requirements

```
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
scipy>=1.9
# For SHAP on Kaggle only:
xgboost>=1.7
shap>=0.41
```

---

## Citation

If you use this code, please cite the paper and the dataset:

```bibtex
@article{[author]2026atrisk,
  title   = {Early Detection of At-Risk Students Using Machine Learning
             and Artificial Intelligence Techniques},
  author  = {Gleb Babichev},
  journal = {arXiv preprint arXiv:2503.XXXXX},
  year    = {2026}
}

@article{realinho2022predicting,
  title   = {Predicting student dropout and academic success},
  author  = {Realinho, Valentim and Machado, Jorge and
             Baptista, Lu{\'i}s and Martins, M{\'o}nica V.},
  journal = {Data},
  volume  = {7},
  number  = {11},
  pages   = {146},
  year    = {2022},
  doi     = {10.3390/data7110146}
}
```

---

## License

Code: MIT License — see [LICENSE](LICENSE).  
Dataset: CC BY 4.0 — must cite Realinho et al. (2022).
