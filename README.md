# Credit Card Fraud Detection - Fraud Probability Estimation

A machine learning pipeline that estimates the **probability** that a credit card transaction is fraudulent (rather than a hard yes/no label), then routes each transaction into a risk tier for a downstream decision system (approve, monitor, review, or block).

---

# Why Probability Instead of a Label

Fraud detection is a **severely imbalanced problem** — fraudulent transactions make up a tiny fraction of all transactions. A model that always predicts "not fraud" can still score high on plain accuracy while being useless in practice.

Outputting a **calibrated probability** (e.g. "87.3% chance of fraud") is more actionable: it lets a business set its own risk thresholds and trade off false positives against false negatives, instead of being locked into one fixed cutoff.

---

# Dataset

This project uses the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — anonymized European credit card transactions collected over a two-day window.

* Features `V1`–`V28` are PCA-transformed components (original features are not disclosed for confidentiality); `Time` and `Amount` are kept in their original form.
* `Class` is the target variable: `0` = legitimate, `1` = fraud.
* ~1,081 duplicate rows are dropped during preprocessing to prevent data leakage between train/test splits.

> **Note:** The CSV file (~150 MB) **is not included in this repository** (too large for a standard git push). To run the notebook:
>
> 1. Download `creditcard.csv` from **https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud** (requires a free Kaggle account).
> 2. Place it in the project root, alongside `fraud_detection.ipynb`.

---

# Approach

## 1. Exploratory Data Analysis (EDA)

* Class distribution analysis
* Transaction amount/time patterns by class
* Correlation heatmap
* Per-feature density plots (`V1`–`V28`) to identify which PCA components separate fraud from normal transactions (notably `V4`, `V11`, `V12`, `V14`, `V17`)

## 2. Feature Engineering

* Log-transform of `Amount`
* Extracted `Hour` feature from `Time`
* Standard scaling of continuous features

## 3. Model Selection

`RandomForestClassifier`, `XGBoost`, and `LightGBM` compared via repeated stratified 5-fold cross-validation on **Average Precision (AUC-PR)**, the primary metric given the class imbalance.

## 4. Final Model

**XGBoost**, trained with early stopping and `scale_pos_weight` to handle class imbalance.

## 5. Probability Calibration

`CalibratedClassifierCV` (sigmoid/Platt scaling) applied on a held-out validation split so predicted probabilities reflect true likelihoods, not just ranking scores.

## 6. Evaluation

AUC-PR, AUC-ROC, and Brier score on a held-out test set, plus precision-recall and ROC curves.

## 7. Explainability

SHAP values are used to identify which features drive individual fraud predictions.

## 8. Decision Layer

Probabilities are mapped to risk tiers with a simple, adjustable business rule:

| Fraud Probability | Tier | Action |
|---|---|---|
| ≥ 70% | Critical | Decline automatically |
| 40–70% | High | Route to fraud analyst |
| 10–40% | Medium | Flag for batch review |
| < 10% | Low | Approve |

---

# Results

Cross-validated performance (5-fold × 5 repeats) of the final XGBoost model:

| Metric | Validation |
|---|---|
| AUC-PR (primary) | ~0.836 ± 0.028 |
| AUC-ROC | ~0.979 ± 0.009 |
| Brier score | ~0.00092 |

RandomForest and XGBoost performed comparably in the initial comparison (AUC-PR ~0.84 and ~0.83 respectively); XGBoost was selected for the final pipeline for its calibration support and early-stopping capability.

---

# Project Structure

```text
├── fraud_detection.ipynb   # Full analysis: EDA, feature engineering, modeling, evaluation, SHAP
├── creditcard.csv          # Dataset (not tracked in git — see above)
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

# Technology & Libraries

## Language

* Python 3

## Data Processing

* pandas
* numpy

## Data Visualization

* matplotlib
* seaborn

## Machine Learning

* scikit-learn
* XGBoost
* LightGBM
* SHAP (model explainability)

---

# Setup & Run Instructions

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Set Up the Data

Download `creditcard.csv` from Kaggle and place it in the project root:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

## 4. Run the Jupyter Notebook

```bash
jupyter notebook fraud_detection.ipynb
```

---

# Limitations & Next Steps

* Model calibration and thresholds were tuned on this dataset's specific fraud rate and cost structure; both should be revisited if deployed on a different transaction population.
* `Time` alone is a weak feature, but the derived `Hour` feature captures some diurnal pattern; a real deployment would likely benefit from additional temporal/behavioral features (e.g. spending velocity, merchant category) not available in this anonymized dataset.
* No live/streaming inference pipeline is included — this repository covers offline model development only.

---

# Expected Outcomes

Upon completion of the notebook:

* A calibrated fraud probability estimate for each transaction, reflecting true likelihood rather than just a ranking score
* Transactions classified into risk tiers to support downstream decision-making
* Identification of the features that most influence individual fraud predictions, via SHAP
* A foundation for setting risk thresholds and fraud-handling policy aligned with business needs
