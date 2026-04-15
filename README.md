# 🧠 IBM HR Attrition — Employee Turnover Prediction

Predicting which employees are most likely to leave, and understanding **why**, so HR teams can intervene before it's too late.

---

## Problem

Employee attrition is expensive — replacing a single employee can cost 50-200% of their annual salary. This project builds a classification model on IBM's HR dataset (1,470 employees, 35 features) to **identify at-risk employees** and surface the key drivers behind turnover.

---

## Key Findings from EDA

- **Overtime is the dominant signal.** Employees working overtime leave at 3× the rate of those who don't (30.5% vs 10.4%).
- **Low pay + overtime = highest risk.** Among the lowest income quartile with overtime, attrition reaches **58%**.
- **Entry-level roles are most vulnerable.** JobLevel 1 + Overtime employees show **52% attrition**.
- **No single numeric feature explains attrition well** (max |corr| < 0.22). The problem is driven by **feature interactions**.

---

## Feature Engineering

Built a custom sklearn transformer (`RiskFeatureAdder`) inside the pipeline to avoid data leakage:

- **HighRisk_Flag** — overtime + low job level + low income (train-derived threshold)
- **MonthlyIncome_log** — log transformation for skew handling
- **Ratio-based features** — e.g. income per experience, promotion ratios

```python
X["Income_per_TotalWorkingYear"] = X["MonthlyIncome"] / (X["TotalWorkingYears"] + 1)
X["YearsSinceLastPromotion_ratio"] = X["YearsSinceLastPromotion"] / (X["YearsAtCompany"] + 1)
```

---

## Modeling

| Model | ROC-AUC | Threshold | Approach |
|------|--------|----------|----------|
| Logistic Regression (Baseline) | 0.822 | 0.682 | class_weight='balanced' |
| Logistic Regression (With FE) | **0.837** | 0.6 | feature engineering + threshold tuning |
| XGBoost | 0.817 | 0.577 | RandomizedSearchCV |

Both models were evaluated with **5-fold stratified CV** and threshold optimization.

---

## Key Insight

Feature engineering improved performance from **0.82 → 0.83+**, while more complex models (XGBoost) did not provide additional gains.

> This suggests that the primary limitation was **feature representation**, not model complexity.

---

## Explainability (SHAP)

- **Global:** `shap.summary_plot`
- **Individual:** `shap.waterfall_plot`

Used to interpret both global drivers and individual predictions.

---

## Pipeline Architecture

```
Raw Data
  → RiskFeatureAdder
    → ColumnTransformer
        ├─ OrdinalEncoder
        ├─ OneHotEncoder
        └─ StandardScaler
      → Model
```

Fully encapsulated in a single sklearn Pipeline (no leakage, reproducible).

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SHAP` · `matplotlib` · `seaborn`

---

## How to Run

```bash
git clone https://github.com/<your-username>/ibm-hr-attrition.git
cd ibm-hr-attrition
pip install -r requirements.txt
jupyter notebook ibm-project.ipynb
```

---

## What I'd Do Next

- [ ] Add FastAPI endpoint
- [ ] Dockerize project
- [ ] Compare SMOTE vs cost-sensitive learning
- [ ] Build Streamlit dashboard
