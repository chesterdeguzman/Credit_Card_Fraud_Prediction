from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.data import load_dataset, split_features_target
from fraud_detection.model import build_pipeline, predict_with_threshold
from fraud_detection.evaluate import save_evaluation_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Train a credit-card fraud classifier.")
    parser.add_argument("--input", default="data/raw/credit_card_fraud_2026.csv")
    parser.add_argument("--model-output", default="models/fraud_model.joblib")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataset(args.input)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    y_pred, y_prob = predict_with_threshold(model, X_test, args.threshold)

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "threshold": args.threshold, "feature_columns": X.columns.tolist()},
        model_path,
    )

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = save_evaluation_artifacts(y_test, y_pred, y_prob, reports_dir)

    predictions = pd.DataFrame({
        "actual_is_fraud": y_test.to_numpy(),
        "predicted_is_fraud": y_pred,
        "fraud_probability": y_prob,
    })
    predictions.to_csv(reports_dir / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
