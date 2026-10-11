# Predicting Dropout and Delayed Completion in Higher Education

Code and paper for *Predicting Dropout and Delayed Completion in Higher Education: Outcome Definition, Prediction Time and Evaluation on a Public Benchmark* (Gleb Babichev, 2026).

Every number, table and figure in `paper/` is produced by the scripts in `src/`. Nothing in the paper is typed by hand.

## Data

UCI *Predict Students' Dropout and Academic Success* (Realinho et al., 2022), DOI [10.24432/C5MC89](https://doi.org/10.24432/C5MC89), CC BY 4.0. 4,424 students, 36 features, three outcomes recorded at the end of the normal course duration: Dropout, Enrolled (still enrolled, i.e. delayed) and Graduate.

Download the CSV from the UCI page and save it as `data/dataset.csv`. The semicolon-separated UCI file and the comma-separated Kaggle copy both work.

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py                 # ~15 min on 2 cores
cd paper && latexmk -pdf main.tex
```

| Script | What it does |
|---|---|
| `src/data.py` | loading, column clean-up, the four settings (binary, three-class, first semester, enrolment) |
| `src/experiments.py` | 8 models x 4 settings, 5-fold CV repeated 5 times, corrected resampled t-tests |
| `src/analysis.py` | calibration, subgroup errors with bootstrap CIs, feature selection inside CV, SHAP, all figures |
| `src/make_tables.py` | writes `paper/generated/` (tables and `\newcommand` number macros) |

## Main results

| Setting | Best macro F1 | Notes |
|---|---|---|
| Binary (Dropout vs Graduate) | 0.905 | 7 of 8 models statistically tied |
| Three-class, both semesters | 0.712 (XGBoost) | F1: Dropout 0.794, Enrolled 0.483, Graduate 0.859; Dropout AUC 0.926 |
| Three-class, first semester | 0.673 | Dropout AUC 0.905 |
| Three-class, enrolment only | 0.551 | Dropout AUC 0.817 |

## Licence

Code: MIT. Data: CC BY 4.0 (Realinho et al.).
