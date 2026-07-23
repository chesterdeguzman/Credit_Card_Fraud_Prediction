from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.data import load_dataset, split_features_target
from fraud_detection.model import build_pipeline, predict_with_threshold

DATA = Path(__file__).parents[1] / "data" / "sample" / "credit_card_fraud_sample.csv"


def test_dataset_loads():
    df = load_dataset(DATA)
    assert len(df) == 500
    assert "is_fraud" in df.columns


def test_pipeline_trains_and_scores():
    df = load_dataset(DATA)
    X, y = split_features_target(df)
    model = build_pipeline(X)
    model.fit(X, y)
    pred, prob = predict_with_threshold(model, X, threshold=0.5)
    assert len(pred) == len(df)
    assert ((prob >= 0) & (prob <= 1)).all()
