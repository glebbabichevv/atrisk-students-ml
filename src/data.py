"""Dataset loading and experimental settings.

Dataset: UCI "Predict Students' Dropout and Academic Success" (Realinho et al., 2022),
DOI 10.24432/C5MC89, CC BY 4.0. Place the CSV at data/dataset.csv. Both the UCI
release (semicolon-separated) and the common Kaggle copy (comma-separated) work.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dataset.csv"

CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]
CLASS_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}

# The UCI and Kaggle releases spell a few columns differently.
RENAME = {
    "Nacionality": "Nationality",
    "Daytime/evening attendance\t": "Daytime/evening attendance",
    "Mother's qualification": "Mother's qualification",
}


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    with open(path, encoding="utf-8-sig") as fh:
        head = fh.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k.strip(): v for k, v in RENAME.items()})
    if "Target" not in df.columns:
        raise ValueError("Expected a 'Target' column")
    df["Target"] = df["Target"].astype(str).str.strip()
    assert set(df["Target"]) == set(CLASS_NAMES), set(df["Target"])
    return df


def feature_columns(df: pd.DataFrame, setting: str) -> list[str]:
    cols = [c for c in df.columns if c != "Target"]
    if setting in ("sem1", "enrol"):
        cols = [c for c in cols if "2nd sem" not in c]
    if setting == "enrol":
        cols = [c for c in cols if "1st sem" not in c]
    return cols


def setting_xy(df: pd.DataFrame, setting: str):
    """Return X (float array), y (int array) and feature names.

    Labels: Dropout=0 in every setting. Binary: Dropout=0, Graduate=1.
    """
    d = df
    if setting == "binary":
        d = df[df["Target"] != "Enrolled"]
        y = (d["Target"] == "Graduate").astype(int).to_numpy()
    else:
        y = d["Target"].map(CLASS_MAP).to_numpy()
    cols = feature_columns(d, setting)
    X = d[cols].to_numpy(dtype=float)
    return X, y, cols


if __name__ == "__main__":
    df = load_dataset()
    print(df.shape)
    print(df["Target"].value_counts())
    for s in ["binary", "multiclass", "sem1", "enrol"]:
        X, y, c = setting_xy(df, s)
        print(s, X.shape, np.bincount(y))
