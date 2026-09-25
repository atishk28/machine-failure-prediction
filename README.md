# Predictive Maintenance: Classifying Machine Failure from Sensor Data

Classifying whether a machine will fail using sensor telemetry, with classic tabular ML (no deep learning).

## Overview

Rotational speed, torque, and tool wear are exactly the kind of sensor telemetry monitored on electric motors and drives in industrial condition-monitoring systems. This project builds a binary classifier to flag failures early, using real sensor readings from a simulated CNC milling machine.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** - 10,000 records with sensor readings (air/process temperature, rotational speed, torque, tool wear) and failure labels, including 5 distinct failure modes.

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- Also available on [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)

## Approach

1. Load and inspect data - only ~3.4% of machines fail, making this an **imbalanced classification** problem
2. Clean up features:
   - Drop `UDI`, `Product ID` (identifiers, not predictive)
   - Drop `TWF`, `HDF`, `PWF`, `OSF`, `RNF` — these are failure-subtype flags that directly determine the target, so keeping them would leak the answer
   - One-hot encode `Type` (product quality grade: L/M/H)
3. Stratified train/test split to preserve the failure ratio in both sets
4. Train and compare two models, both with `class_weight='balanced'` to handle the imbalance:
   - Logistic Regression (baseline)
   - Random Forest Classifier
5. Evaluate with accuracy, precision, recall, F1; visualize confusion matrix and feature importance

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.825 | 0.142 | 0.824 | 0.242 |
| Random Forest | 0.972 | 0.563 | 0.721 | **0.632** |

Random Forest substantially outperforms Logistic Regression on precision and F1 by capturing non-linear interactions between torque, speed, and tool wear. Logistic Regression catches slightly more failures (higher recall) but with far more false alarms — a real trade-off maintenance teams face between missed failures and unnecessary inspections. Torque, rotational speed, and tool wear dominate feature importance, consistent with the physical failure modes in the data.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook machine_failure_classification.ipynb
```

The dataset (`ai4i2020.csv`) is included in this repo - no external download needed.

## Possible Extensions

- Tune the decision threshold to trade off precision vs. recall based on business cost
- Engineer a `power = torque × speed` feature, a physically motivated predictor used in the original paper
- Try a gradient boosting model (XGBoost/LightGBM) for comparison
