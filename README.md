# IBM HR Attrition — Employee Turnover Prediction

Predicting which employees are most likely to leave, and understanding **why**, so HR teams can intervene before it's too late.

---

## Problem

Employee attrition is expensive — replacing a single employee can cost 50-200% of their annual salary. This project builds a classification model on IBM's HR dataset (1,470 employees, 35 features) to **identify at-risk employees** and surface the key drivers behind turnover.

## Key Findings from EDA

- **Overtime is the dominant signal.** Employees working overtime leave at 3× the rate of those who don't (30.5% vs 10.4%).
- **Low pay + overtime = highest risk.** Among the lowest income quartile with overtime, attrition reaches **58%**.
- **Entry-level roles are most vulnerable.** JobLevel 1 + Overtime employees show **52% attrition** — the single riskiest group in the dataset.
- **No single numeric feature explains attrition well** (max |corr| < 0.22). The problem is driven by **feature interactions**, not individual variables.

## Feature Engineering

Built a custom sklearn transformer (`RiskFeatureAdder`) inside the pipeline to avoid data leakage:

- **`HighRisk_Flag`** — A composite binary feature capturing the riskiest employee profile: overtime + low job level + income below Q1 (learned from training data only).
- **`MonthlyIncome_log`** — Log-transformed income to handle right skew while dropping the raw column.

```python
class RiskFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.income_q1_ = X["MonthlyIncome"].quantile(0.25)  # train only
        return self

    def transform(self, X):
        X = X.copy()
        X["MonthlyIncome_log"] = np.log1p(X["MonthlyIncome"])
        X["HighRisk_Flag"] = (
            (X["OverTime"] == "Yes") &
            (X["JobLevel"] <= 2) &
            (X["MonthlyIncome"] < self.income_q1_)
        ).astype(int)
        return X.drop(columns=["MonthlyIncome"])
```

## Modeling

| Model | ROC-AUC | Threshold | Approach |
|-------|---------|-----------|----------|
| **Logistic Regression** | 0.822 | 0.682 | `class_weight='balanced'` |
| **XGBoost** | 0.817 | 0.577 | `scale_pos_weight=5`, RandomizedSearchCV (50 iter) |

Both models were evaluated with **5-fold stratified CV** for threshold optimization, maximizing F1-score on the minority class.

**Why two models?** Logistic Regression serves as an interpretable baseline; XGBoost was tuned for better minority-class detection. LR slightly outperformed on AUC in this dataset — a useful reminder that complex models don't always win on small, tabular data.

## Explainability (SHAP)

SHAP TreeExplainer was used on the XGBoost model to provide both global feature importance and individual-level explanations:

- **Global:** `shap.summary_plot` reveals which features drive predictions across all employees.
- **Individual:** `shap.waterfall_plot` for the highest-risk employee shows exactly which factors pushed their score up.

## Pipeline Architecture

```
Raw Data
  → RiskFeatureAdder (custom feature engineering)
    → ColumnTransformer
        ├─ OrdinalEncoder (BusinessTravel)
        ├─ OneHotEncoder (Department, JobRole, MaritalStatus, ...)
        └─ StandardScaler (all numeric + HighRisk_Flag)
      → Model (LogisticRegression / XGBClassifier)
```

Everything runs through a single `sklearn.Pipeline` — no data leakage, fully reproducible, ready to serialize with `joblib`.

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SHAP` · `matplotlib` · `seaborn`

## How to Run

```bash
git clone https://github.com/<your-username>/ibm-hr-attrition.git
cd ibm-hr-attrition
pip install -r requirements.txt
jupyter notebook ibm_eda.ipynb
```

## What I'd Do Next

- [ ] Add FastAPI endpoint for real-time predictions
- [ ] Dockerize for deployment
- [ ] Experiment with SMOTE vs. cost-sensitive learning comparison
- [ ] Build a simple Streamlit dashboard for HR users
