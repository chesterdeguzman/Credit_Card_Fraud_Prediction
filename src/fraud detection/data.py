from pathlib import Path
import pandas as pd

TARGET = "is_fraud"
ID_COLUMN = "transaction_id"


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {TARGET, ID_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0 and 1.")
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID_COLUMN])
    y = df[TARGET].astype(int)
    return X, y
