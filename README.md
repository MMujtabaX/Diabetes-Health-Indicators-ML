# 🩺 Diabetes Health Indicators — ML Prediction

A machine learning project that tackles diabetes prediction as **three separate problems** on 100,000 patient records:

1. **Binary classification:** does the patient have diagnosed diabetes?
2. **Multiclass classification:** which stage (No Diabetes, Pre-Diabetes, Type 1, Type 2, Gestational)?
3. **Regression:** what is the patient's diabetes risk score?

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📌 Overview

The dataset has 31 features covering demographics (age, gender, ethnicity, income, education), lifestyle (smoking, alcohol, physical activity, diet, sleep, screen time), medical history (family history, hypertension, cardiovascular disease), and clinical measurements (BMI, blood pressure, cholesterol, fasting and postprandial glucose, insulin, HbA1c).

## 🔄 Workflow

| Step | Stage | Details |
|------|-------|---------|
| 1 | Data inspection | 100K rows × 31 columns, no missing values |
| 2 | Preprocessing | Label encoding for the stage target, one-hot encoding for 6 categorical features |
| 3 | EDA | Target distributions, correlation heatmaps, and box and scatter plots for top predictors |
| 4 | Leakage prevention | For each task, the other two targets are removed from the features (e.g. `diabetes_stage` correlates 0.96 with `diagnosed_diabetes`) |
| 5 | Baseline models | Logistic Regression, Decision Tree, KNN and Linear Regression |
| 6 | Hyperparameter tuning | `GridSearchCV` with 5-fold cross-validation |
| 7 | Export | Final tuned models saved with `joblib` |

## 🔍 Key EDA Insights

- **HbA1c** is the strongest predictor of both diagnosis (r = 0.68) and stage (r = 0.71), followed by postprandial and fasting glucose.
- **Family history** is the strongest driver of the risk score (r = 0.73), followed by age.
- **Physical activity** is negatively related to the risk score.
- Stage classes are **heavily imbalanced**: Type 2 has 59,774 cases, while Type 1 has only 122 and Gestational only 278.

## 📊 Results

Evaluated on a held-out 20% test set of 20,000 samples.

### Binary classification: diagnosed diabetes

| Model | Accuracy | F1 (Diabetic) |
|-------|----------|---------------|
| Logistic Regression | 0.847 | 0.875 |
| KNN (k=5) | 0.810 | 0.840 |
| Decision Tree | 0.860 | 0.880 |
| **Tuned Decision Tree** (depth 5) | **0.920** | **0.930** |

Logistic Regression reached a **ROC-AUC of 0.923**. The tuned tree achieved **100% precision** on diabetic cases.

### Multiclass classification: diabetes stage

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 0.769 | 0.415 |
| KNN (k=5) | 0.759 | 0.422 |
| Decision Tree | 0.855 | 0.514 |
| **Tuned Decision Tree** | **0.919** | **0.551** |

The gap between accuracy and macro F1 comes from the rare classes. The model classifies Type 2, Pre-Diabetes and No Diabetes well (F1 0.91–0.93) but cannot detect Type 1 or Gestational cases.

### Regression: diabetes risk score

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Linear Regression** | **0.439** | **0.745** | **0.993** |
| Decision Tree Regressor | — | — | 0.969 |

## 📁 Project Structure

```
├── data/
│   ├── raw/diabetes_dataset.csv
│   └── processed/diabetes_processed.csv
├── models/
│   ├── binary_model.joblib
│   ├── multiclass_model.joblib
│   └── regression_model.joblib
├── notebooks/
│   └── main_analysis.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

```bash
git clone https://github.com/MMujtabaX/Diabetes-Health-Indicators-ML.git
cd Diabetes-Health-Indicators-ML
pip install -r requirements.txt
cd notebooks
jupyter notebook main_analysis.ipynb
```

To load a trained model:

```python
import joblib
model = joblib.load("models/binary_model.joblib")
```

## ⚠️ Limitations & Future Work

- **Rare classes:** Type 1 and Gestational diabetes go undetected. Class weighting, SMOTE, or merging rare classes would help.
- **Feature scaling:** features were scaled but the unscaled data was used for training, which likely hurt KNN and Logistic Regression. Using a scikit-learn `Pipeline` would fix this.
- **Near-perfect regression score:** an R² of 0.993 suggests the risk score is close to a direct formula of the input features. This is typical of synthetic datasets and would not hold on real clinical data.
- **Ensemble models:** try Random Forest, XGBoost or LightGBM.
- **Deployment:** a Streamlit app for interactive predictions.
- **Code structure:** move preprocessing and evaluation into reusable `src/` modules.

## 📚 Dataset

Diabetes Health Indicators dataset: 100,000 records, 31 features.

## 👤 Author

**Muhammad Mujtaba Khan Suri** — CS @ UBIT, University of Karachi
[GitHub](https://github.com/MMujtabaX)
