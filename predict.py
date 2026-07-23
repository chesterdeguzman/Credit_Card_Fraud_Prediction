from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from fraud_detection.model import predict_with_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Score transactions with a trained fraud model.")
    parser.add_argument("--model", default="models/fraud_model.joblib")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = joblib.load(args.model)
    model = bundle["model"]
    threshold = args.threshold if args.threshold is not None else bundle["threshold"]
    df = pd.read_csv(args.input)
    ids = df["transaction_id"] if "transaction_id" in df.columns else pd.Series(range(len(df)))
    X = df.drop(columns=["is_fraud", "transaction_id"], errors="ignore")
    missing = set(bundle["feature_columns"]).difference(X.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    X = X[bundle["feature_columns"]]
    pred, prob = predict_with_threshold(model, X, threshold)
    out = pd.DataFrame({
        "transaction_id": ids,
        "predicted_is_fraud": pred,
        "fraud_probability": prob,
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
