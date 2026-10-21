"""
UPV data: Universitat Politecnica de Valencia (Spain), first-year entrants.

Igualde-Saez et al. (2025), "University Student Dropout: A Longitudinal Dataset of
Demographic, Socioeconomic, and Academic Indicators", Data 10(10):162,
Zenodo DOI 10.5281/zenodo.17239943, CC BY 4.0. The three CSV files
(dataset_2018_hash.csv, dataset_2021_hash.csv, dataset_2022_hash.csv) go in data/upv/.

Rows in the raw files are student x course x year. We keep first-year entrants
(anyo_ingreso == academic year and curso_mas_bajo == 1), one row per student.
Target: the institution's dropout flag (abandono == "A"), defined in the
descriptor as no enrolment for two consecutive years with credits pending.

Prediction points (columns are selected by prefix, see STAGE_PREFIX):
  enrol     information at enrolment (access route, admission grades, programme,
            enrolled credits, parents' education, part-time status, ...)
  sem1      + credits passed in the first semester
  sem1_lms  + LMS and Wi-Fi activity from September to January
  year      + second-semester and annual results, mean course grade
  year_lms  + LMS and Wi-Fi activity from September to June
  lms1_only enrol + LMS and Wi-Fi to January, without first-semester credits (check)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.paths import UPV_DATA as RAW

YEARS = [2018, 2021, 2022]
LMS_KINDS = ["pft_events", "pft_days_logged", "pft_visits", "pft_assignment_submissions",
             "pft_test_submissions", "pft_total_minutes", "resource_events", "n_resource_days"]
CATS = ["tipo_ingreso", "estudios_p_hash", "estudios_m_hash", "campus_hash", "tit_hash"]


def month_cols(cols, kind, year, months):
    out = []
    for m in months:
        y = year if m >= 9 else year + 1
        c = f"{kind}_{y}_{m}"
        if c in cols:
            out.append(c)
    return out


def build_year(year: int) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"dataset_{year}_hash.csv", sep=";", decimal=",", low_memory=False)
    g = df.groupby("dni_hash", sort=False)
    st = g.first().copy()  # copy: de-fragment before adding columns
    st["n_courses"] = g.size()
    st["mean_course_grade"] = g["nota_asig_hash"].mean()
    new = st[(st["anyo_ingreso"] == year) & (st["curso_mas_bajo"] == 1)].copy()
    ids = new.index
    out = pd.DataFrame(index=ids)
    out["year"] = year
    out["dropout"] = (new["abandono_hash"] == "A").astype(int)
    # enrolment
    out["part_time"] = (new["dedicacion"] == "TP").astype(int)
    out["displaced"] = (new["desplazado_hash"] == "A").astype(int)
    for c in ["nota10_hash", "nota14_hash", "preferencia_seleccion", "cred_mat_total",
              "cred_mat_sem_a", "cred_mat_sem_b", "cred_mat_anu", "n_courses"]:
        out["enr_" + c] = new[c]
    for c in CATS:
        out["cat_" + c] = new[c].fillna("missing").astype(str)
    # first semester results
    out["s1_cred_passed"] = new["cred_sup_sem_a"]
    out["s1_pass_rate"] = new["cred_sup_sem_a"] / new["cred_mat_sem_a"].replace(0, np.nan)
    # full-year results
    out["yr_cred_passed_b"] = new["cred_sup_sem_b"]
    out["yr_cred_passed_anu"] = new["cred_sup_anu"]
    out["yr_cred_passed"] = new["cred_sup_total"]
    out["yr_pass_rate"] = new["cred_sup_total"] / new["cred_mat_total"].replace(0, np.nan)
    out["yr_mean_grade"] = new["mean_course_grade"]
    # LMS: per-course counts are summed per student; Wi-Fi is per student
    sub = df[df["dni_hash"].isin(ids)]
    for tag, months in [("lms1", [9, 10, 11, 12, 1]), ("lmsy", [9, 10, 11, 12, 1, 2, 3, 4, 5, 6])]:
        for kind in LMS_KINDS:
            cols = month_cols(df.columns, kind, year, months)
            out[f"{tag}_{kind}"] = sub.groupby("dni_hash")[cols].sum(min_count=1).sum(axis=1, min_count=1).reindex(ids).fillna(0) if cols else 0
        wcols = month_cols(df.columns, "n_wifi_days", year, months)
        w = new[wcols].sum(axis=1, min_count=1) if wcols else pd.Series(np.nan, index=ids)
        out[f"{tag}_wifi_days"] = w.fillna(0)
        out[f"{tag}_wifi_missing"] = w.isna().astype(int)
    # early-term LMS trend (December vs October events)
    oc, dc = f"pft_events_{year}_10", f"pft_events_{year}_12"
    if oc in df and dc in df:
        o = sub.groupby("dni_hash")[oc].sum().reindex(ids).fillna(0)
        d = sub.groupby("dni_hash")[dc].sum().reindex(ids).fillna(0)
        out["lms1_trend"] = (d - o) / (o + d + 1)
    return out


def load_upv() -> pd.DataFrame:
    return pd.concat([build_year(y) for y in YEARS])


STAGE_PREFIX = {
    "enrol": ["part_time", "displaced", "enr_", "cat_"],
    "sem1": ["part_time", "displaced", "enr_", "cat_", "s1_"],
    "sem1_lms": ["part_time", "displaced", "enr_", "cat_", "s1_", "lms1_"],
    "year": ["part_time", "displaced", "enr_", "cat_", "s1_", "yr_"],
    "year_lms": ["part_time", "displaced", "enr_", "cat_", "s1_", "yr_", "lmsy_"],
    # Check stage: first-semester signal without first-semester credits. The credit
    # counts may include January resits; LMS and Wi-Fi activity up to January cannot.
    "lms1_only": ["part_time", "displaced", "enr_", "cat_", "lms1_"],
}
MAIN_STAGES = ["enrol", "sem1", "sem1_lms", "year", "year_lms"]


def design(df: pd.DataFrame, stage: str, min_count: int = 30):
    cols = [c for c in df.columns if any(c.startswith(p) for p in STAGE_PREFIX[stage])]
    X = df[cols].copy()
    cat = [c for c in cols if c.startswith("cat_")]
    num = [c for c in cols if c not in cat]
    for c in num:
        miss = X[c].isna()
        if miss.any():
            X[c + "_missing"] = miss.astype(int)
            X[c] = X[c].fillna(X[c].median())
    for c in cat:
        vc = X[c].value_counts()
        X[c] = X[c].where(X[c].map(vc) >= min_count, "other")
    X = pd.get_dummies(X, columns=cat, dtype=float)
    return X.to_numpy(dtype=float), df["dropout"].to_numpy(), list(X.columns)
