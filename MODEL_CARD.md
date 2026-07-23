# Model Card: Credit Card Fraud Prediction Baseline

## Intended use

This repository provides an educational baseline for binary fraud-risk classification. It is designed for portfolio work, experimentation, and reproducible model development.

## Model

- Balanced logistic regression
- Median imputation and standardization for numeric features
- Most-frequent imputation and one-hot encoding for categorical and Boolean features
- Probability threshold configurable at training or scoring time

## Important limitations

- The dataset appears synthetic and may not reproduce real-world fraud behavior.
- Fraud is rare, so accuracy is not a sufficient metric.
- Predictions should not be used as the sole basis for declining transactions or taking adverse action.
- Production deployment requires temporal validation, drift monitoring, calibration, cost-sensitive thresholding, human review, privacy controls, and fairness testing.
- Transaction ID is excluded from model features.

## Evaluation focus

Use ROC-AUC, average precision, fraud-class precision, fraud-class recall, confusion matrix, and expected operational cost. Select a threshold based on the relative cost of missed fraud and false alerts.
