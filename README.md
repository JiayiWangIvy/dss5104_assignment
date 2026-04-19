# DSS5104 Assignment 2: Fraud and Anomaly Detection
## Team Members

| Name        | Student Id       |
|-------------|------------------|
| Jiayi Wang  | A0327444A        |
| Jiasheng Zhang | A0333303W     |
| Ye Xia         | A0327242L     |
| Jiahe Han     | A0333982W      |

This repository contains the code for DSS5104 Assignment 2, focusing on fraud and anomaly detection with both classical machine learning and deep learning methods.

The main notebook is:

```text
ass2_fraudDetection.ipynb
```

It includes data loading, exploratory data analysis, feature engineering, temporal splitting, model comparison, threshold analysis, error analysis, and a secondary dataset benchmark.

## Project Structure

```text
.
├── ass2_complete.ipynb
└── README.md
```

## Datasets

### Primary Dataset: IEEE-CIS Fraud Detection

The primary dataset is the IEEE-CIS Fraud Detection dataset from Kaggle.

Source: https://www.kaggle.com/competitions/ieee-fraud-detection

Required files:

```text
ieee-fraud-detection/train_transaction.csv
ieee-fraud-detection/train_identity.csv
```

The official Kaggle test files can be downloaded from the same source, but they do not contain the `isFraud` label. Therefore, model evaluation is performed by splitting the labelled training data chronologically.

### Secondary Dataset: Credit Card Fraud

The secondary dataset is the Kaggle Credit Card Fraud dataset.

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Current expected path:

```text
creditcard.csv
```

This dataset is used as a modeling-only benchmark because its features are anonymised PCA components. Domain-specific feature engineering is not applied to this dataset.

## Why Chronological Splitting Is Used

Fraud detection is a time-dependent problem. In a real deployment setting, a model is trained on past transactions and used to predict future transactions. Random splitting can leak future behavioral information into the training set and produce overly optimistic results.

For this reason, the notebook sorts transactions by time and uses an 80/20 chronological split:

- first 80% of labelled records for training
- final 20% of labelled records for testing

For IEEE-CIS, the split uses `TransactionDT`.

For Credit Card Fraud, the split uses `Time`.

## Feature Engineering

Feature engineering is performed on the IEEE-CIS dataset because it contains raw transaction, card, address, email, and device variables.

The engineered features include:

- time-based features such as transaction hour, transaction day, and night-transaction indicator
- amount features such as `log1p(TransactionAmt)`, cents pattern, and rounded amount indicator
- email-domain features such as provider, suffix, and purchaser/recipient match flags
- pseudo-user and card-level aggregate statistics
- transaction amount ratios relative to historical user/card behavior
- time since last transaction for the same pseudo-user or card

Aggregate features are learned from the training split and then applied to the test split to reduce temporal leakage.

## Models

The notebook compares several types of methods:

- classical anomaly detection models such as Isolation Forest and LOF
- gradient boosting models with imbalance handling
- autoencoder-based anomaly detection
- VAE-based anomaly detection
- supervised neural network models with imbalance-aware training
- secondary-dataset baselines on Credit Card Fraud

The exact model sections are contained in `ass2_complete.ipynb`.

## Evaluation

Accuracy is not used as the main metric because fraud is rare.

The main metrics are:

- AUPRC
- ROC AUC
- precision
- recall
- F1 score
- precision-recall curves

The notebook also includes cost-sensitive threshold analysis using asymmetric costs:

- false negative cost: 500
- false positive cost: 2

This is used to compare the cost-optimal threshold with the F1-optimal threshold.

## How to Run

1. Create and activate a Python environment.

2. Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm torch
```

If `lightgbm` is unavailable, the notebook contains fallback code using scikit-learn models for some sections.

3. Make sure the primary dataset is placed under:

```text
ieee-fraud-detection/
```

4. Make sure the secondary Credit Card Fraud dataset is available as:

```text
creditcard.csv
```

5. Open and run:

```text
ass2_complete.ipynb
```

Run the notebook from top to bottom so that later modeling sections can reuse variables created in earlier preprocessing and feature-engineering sections.

## Reproducibility Notes

The notebook uses a fixed random seed where applicable.

Some deep learning results may vary slightly depending on hardware, PyTorch version, and GPU/CPU execution.

The Kaggle official IEEE-CIS test files are not used for metric evaluation because their labels are hidden. All reported metrics should be computed from the labelled training data using the chronological split.

## Report Guidance

The final report should focus on:

- why chronological splitting is necessary
- how feature engineering changes IEEE-CIS model performance
- whether deep learning models outperform classical methods
- how supervised and semi-supervised regimes compare
- why AUPRC is more informative than accuracy
- how threshold choice changes under asymmetric costs
- whether findings generalise to the secondary Credit Card Fraud dataset
