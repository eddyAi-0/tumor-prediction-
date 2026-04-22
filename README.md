# Tumor Cell Prediction

Machine learning pipeline for binary classification of breast cancer cells (malignant vs. benign) using the scikit-learn Breast Cancer Wisconsin dataset.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9954** | 0.9957 ± 0.005 |
| SVM | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 | 0.9957 ± 0.005 |
| Gradient Boosting | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9907 | 0.9918 ± 0.005 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9939 | 0.9896 ± 0.008 |

Best model: **Logistic Regression** (ROC-AUC = 0.9954, 5-fold stratified CV)

## Dataset

- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569 (212 malignant · 357 benign)
- **Features:** 30 numerical (mean, SE, worst of 10 cell nucleus measurements)
- **Task:** Binary classification — 0 Malignant · 1 Benign

## Project Structure

```
tumor-prediction/
├── notebooks/
│   └── 01_exploratory_analysis.ipynb   # EDA: distributions, heatmap, PCA, scree plot
├── src/
│   ├── data/
│   │   └── preprocessing.py            # Dataset loading and StandardScaler pipeline
│   ├── models/
│   │   └── trainer.py                  # Training, evaluation, cross-validation
│   └── visualization/
│       └── plots.py                    # Reusable plotting functions
├── outputs/
│   ├── models/                         # Saved model (.pkl), scaler, results.csv
│   └── figures/                        # Confusion matrix, ROC curves, comparison chart
├── train.py                            # End-to-end training script
└── requirements.txt
```

## Setup

```bash
git clone <repo-url>
cd tumor-prediction
pip install -r requirements.txt
```

## Usage

```bash
# Full training pipeline
python train.py

# Exploratory analysis
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

`train.py` trains all four models, prints a comparison table, saves the best model to `outputs/models/best_model.pkl` and generates figures in `outputs/figures/`.

## Loading the saved model

```python
import joblib

model = joblib.load("outputs/models/best_model.pkl")
scaler = joblib.load("outputs/models/scaler.pkl")

X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
probabilities = model.predict_proba(X_new_scaled)[:, 1]
```

## Stack

- Python 3.10+
- scikit-learn · pandas · numpy · matplotlib · seaborn · joblib
