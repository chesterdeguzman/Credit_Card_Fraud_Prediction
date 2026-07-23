# Credit Card Fraud Prediction

A GitHub-ready predictive analytics project for identifying potentially fraudulent credit-card transactions from the supplied `credit_card_fraud_2026.csv` dataset.

## Project summary

The dataset contains **20,000 transactions**, **25 candidate predictors**, and one binary target: `is_fraud`. Fraud is highly imbalanced: **339 fraud records (1.695%)** and **19,661 legitimate records**.

This repository trains a leakage-conscious, class-balanced logistic-regression baseline and evaluates it with fraud-appropriate metrics. `transaction_id` is retained only for output traceability and is excluded from training.

## Baseline test performance

Using a stratified 80/20 split and a 0.50 decision threshold:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.935 |
| Average precision | 0.440 |
| Fraud recall | 0.809 |
| Fraud precision | 0.107 |
| Fraud F1 | 0.188 |
| Accuracy | 0.881 |

Confusion matrix: `[[3471, 461], [13, 55]]`.

The high recall comes with many false-positive alerts. In an operational system, the threshold should be selected using investigation capacity and the relative cost of missed fraud versus false alerts.

## Repository structure

```text
.
├── data/
│   ├── raw/credit_card_fraud_2026.csv
│   └── sample/credit_card_fraud_sample.csv
├── notebooks/01_exploration_and_modeling.ipynb
├── src/fraud_detection/
│   ├── data.py
│   ├── evaluate.py
│   └── model.py
├── tests/test_pipeline.py
├── train.py
├── predict.py
├── MODEL_CARD.md
├── requirements.txt
└── .github/workflows/tests.yml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Train and evaluate

```bash
PYTHONPATH=src python train.py \
  --input data/raw/credit_card_fraud_2026.csv \
  --model-output models/fraud_model.joblib \
  --reports-dir reports \
  --threshold 0.50
```

The command creates:

- `models/fraud_model.joblib`
- `reports/metrics.json`
- `reports/test_predictions.csv`
- confusion-matrix, ROC, and precision-recall plots

## Score new transactions

```bash
PYTHONPATH=src python predict.py \
  --model models/fraud_model.joblib \
  --input data/sample/credit_card_fraud_sample.csv \
  --output reports/scored_transactions.csv
```

## Why accuracy is not enough

A classifier that labels every transaction legitimate would exceed 98% accuracy on this dataset while detecting no fraud. This project therefore emphasizes fraud recall, precision, average precision, ROC-AUC, and the confusion matrix.

## Responsible use

This is an educational predictive-analytics baseline, not a production fraud engine. Real deployments need time-based validation, model monitoring, probability calibration, data-governance controls, explainability, human review, and careful threshold selection.
