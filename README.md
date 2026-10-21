# Predicting Dropout and Delayed Completion in Higher Education

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE) [![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-green.svg)](data/README.md) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

Code and paper for *Predicting Dropout and Delayed Completion in Higher Education: Outcome Definition, Prediction Time and a Replication at a Second University* (Gleb Babichev, 2026).

The main study uses the public UCI dataset from a Portuguese polytechnic. Its findings are then checked on a second university, Universitat Politècnica de València (Spain), with a different dropout definition, three entry cohorts and learning-platform activity.

Every number, table and figure in `paper/` is produced by `run_all.py`. Nothing in the paper is typed by hand.

## Data

- **UCI, Portugal** (main study): *Predict Students' Dropout and Academic Success* (Realinho et al., 2022), DOI [10.24432/C5MC89](https://doi.org/10.24432/C5MC89), CC BY 4.0. 4,424 students, 36 features, three outcomes recorded at the end of the normal course duration: Dropout, Enrolled (still enrolled, i.e. delayed) and Graduate. Included as `data/dataset.csv`.
- **UPV, Spain** (external validation): Igualde-Sáez et al. (2025), DOI [10.5281/zenodo.17239943](https://doi.org/10.5281/zenodo.17239943), CC BY 4.0. 9,359 first-year entrants from 2018, 2021 and 2022, with LMS and Wi-Fi activity. Download separately, see `data/README.md`.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --list          # the steps
python run_all.py                 # every step
python run_all.py --paper         # same, then build paper/main.pdf
```

Other options: `--only upv` (a group or step names), `--skip uci.tuning` (the slowest step; its table is then taken from the last run), `--from uci.alerts` (resume after a failure). Each run saves logs to `results/logs/` and a record of the commit, library versions, data checksums and step durations to `results/run_manifest.json`.

## Code

```
run_all.py                         entry point
src/common/   paths.py             all file locations
              models.py            the 8 models, seed, corrected resampled t-test
              decision.py          alert lists and group thresholds (same code for both datasets)
              plotting.py          figure style
src/uci/      data.py              loading, the four settings, out-of-fold predictions
              01_benchmark.py      8 models x 4 settings, 5-fold CV repeated 5 times
              02_calibration_subgroups_shap.py
                                   calibration, subgroup errors, feature selection inside CV, SHAP
              03_fee_sensitivity.py  three-class analysis without the two fee-status variables
              04_alerts_and_thresholds.py
                                   fixed-size alert lists, group-specific thresholds (cross-fitted)
              05_programme_transfer.py  leave-one-programme-out validation (17 programmes)
              06_nested_tuning.py  nested CV: fixed vs tuned hyperparameters
src/upv/      data.py              first-year entrants, five prediction points
              01_external_validation.py
                                   CV, calibration, alert lists, temporal and unseen-degree
                                   validation, subgroup thresholds, resit check
src/report/   make_figures.py      figure comparing the two universities
              make_tables.py       paper/generated/: tables and \newcommand number macros
results/uci/, results/upv/         outputs of the steps
```

## Main results

| UCI (Portugal) | Best macro F1 | Notes |
|---|---|---|
| Binary (Dropout vs Graduate) | 0.905 | 7 of 8 models statistically tied |
| Three-class, both semesters | 0.712 (XGBoost) | F1: Dropout 0.795, Enrolled 0.482, Graduate 0.860; Dropout AUC 0.926 |
| Three-class, first semester | 0.673 | Dropout AUC 0.905 |
| Three-class, enrolment only | 0.551 | Dropout AUC 0.817 |

After one semester, a list of the 20% highest-risk students reaches 57% of eventual dropouts with 92% precision. Group-specific thresholds raise dropout recall for scholarship holders from 0.38 to 0.57 at almost no cost in precision. Nested tuning does not change any result significantly.

At UPV (Spain, 9,359 first-year students, dropout vs. all others), the dropout AUC is 0.74 at enrolment, 0.82 after the first semester and 0.89 at the end of the year, with 5 of 8 models tied. The recall gap between low- and high-dropout groups and its correction by group thresholds replicate. Models trained on earlier cohorts lose about 0.06 AUC on later ones, and neither logistic regression nor XGBoost transfers consistently better. `paper/generated/tab_replication.tex` puts both universities side by side.

## Citation

See `CITATION.cff` (GitHub shows a "Cite this repository" button).

## Licence

Code: MIT. Data: CC BY 4.0 (Realinho et al.).
