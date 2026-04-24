Data Preprocessing Pipeline
Outputs: X_train, X_test, y_train, y_test as numpy arrays + feature names
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "dataset.csv"

CLASS_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]


def load_and_preprocess(test_size=0.2, random_state=42, verbose=True):
    df = pd.read_csv(DATA_PATH)

    # Encode target
    df["label"] = df["Target"].map(CLASS_MAP)

    X = df.drop(columns=["Target", "label"])
    y = df["label"].values
    feature_names = X.columns.tolist()

    if verbose:
        print(f"Features: {X.shape[1]}")
        print(f"Class distribution: { {k: (y==v).sum() for k,v in CLASS_MAP.items()} }")

    # Train/test split — stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    # Standard scaling (needed for LR, SVM, MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if verbose:
        print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "class_names": CLASS_NAMES,
    }


def load_early_prediction(test_size=0.2, random_state=42, verbose=True):
    """Early prediction variant: drop all 2nd semester features."""
    df = pd.read_csv(DATA_PATH)
    df["label"] = df["Target"].map(CLASS_MAP)

    sem2_cols = [c for c in df.columns if "2nd sem" in c]
    X = df.drop(columns=["Target", "label"] + sem2_cols)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size,
        random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train, "X_test": X_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "feature_names": X.columns.tolist(),
        "scaler": scaler, "class_names": CLASS_NAMES,
        "dropped_cols": sem2_cols,
    }


if __name__ == "__main__":
    data = load_and_preprocess()
    print("Preprocessing OK")


def load_binary(test_size=0.2, random_state=42, verbose=True):
    """Binary classification: Dropout (0) vs Graduate (1), drop Enrolled."""
    df = pd.read_csv(DATA_PATH)
    df = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
    df["label"] = df["Target"].map({"Dropout": 0, "Graduate": 1})

    X = df.drop(columns=["Target", "label"])
    y = df["label"].values

    if verbose:
        print(f"Binary dataset: {len(df)} rows (Dropout={(y==0).sum()}, Graduate={(y==1).sum()})")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return {
        "X_train": X_train, "X_test": X_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "feature_names": X.columns.tolist(),
        "scaler": scaler, "class_names": ["Dropout", "Graduate"],
    }

def load_binary(test_size=0.2, random_state=42, verbose=True):
    """Binary experiment: remove Enrolled, Dropout=0, Graduate=1."""
    df = pd.read_csv(DATA_PATH)
    df = df[df["Target"] != "Enrolled"].copy()
    df["label"] = df["Target"].map({"Dropout": 0, "Graduate": 1})

    X = df.drop(columns=["Target", "label"])
    y = df["label"].values

    if verbose:
        print(f"Binary — samples: {len(y)}  Dropout={(y==0).sum()}  Graduate={(y==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return {
        "X_train": X_train, "X_test": X_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "feature_names": X.columns.tolist(),
        "scaler": scaler,
        "class_names": ["Dropout", "Graduate"],
    }


if __name__ == "__main__":
    data = load_and_preprocess()
    print("Preprocessing OK")
