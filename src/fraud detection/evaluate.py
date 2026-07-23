from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    RocCurveDisplay,
)


def evaluate_predictions(y_true, y_pred, y_prob) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "fraud_precision": float(report["1"]["precision"]),
        "fraud_recall": float(report["1"]["recall"]),
        "fraud_f1": float(report["1"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_evaluation_artifacts(y_true, y_pred, y_prob, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Legitimate", "Fraud"])
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=160)
    plt.close()

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
        output_dir / "precision_recall_curve.csv", index=False
    )
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_curve.png", dpi=160)
    plt.close()
    return metrics
