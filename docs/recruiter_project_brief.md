# Recruiter Project Brief

## Project

**Employee Attrition Prediction & Workforce Intelligence** is an end-to-end People Analytics project that predicts employee attrition risk, explains model drivers, segments workforce patterns, and presents results through a polished Streamlit decision-support dashboard.

## Why This Project Matters

Employee attrition is a high-impact workforce planning problem. This project converts a public HR analytics dataset into a professional decision-support workflow: data quality validation, leakage-aware modeling, explainability, employee risk scoring, fairness review, SQL reporting, R statistical analysis, and executive dashboard delivery.

## What This Demonstrates

- Business problem framing for HR and People Analytics.
- Clean data validation with missing-value, duplicate, target, and leakage checks.
- Feature engineering with HR-readable signals such as overtime, tenure, satisfaction, reward, promotion, and workload indicators.
- Imbalanced classification using PR-AUC, recall, F2, balanced accuracy, threshold optimization, lift, and gain.
- Explainable risk scoring with human-readable review explanations.
- Responsible AI framing: model outputs are decision support, not automated employee decisions.
- Multi-tool analytics workflow across Python, SQL, R, notebooks, tests, reports, and Streamlit.

## Key Results

| Area | Result |
| --- | --- |
| Dataset health | 1,470 rows, 35 columns, 0 missing values, 0 duplicates |
| Attrition rate | 16.12% observed attrition |
| Selected model | Logistic Regression |
| ROC-AUC | 0.829 |
| PR-AUC | 0.558 |
| Recall | 0.702 |
| Balanced accuracy | 0.750 |
| Top 20% risk deciles | Captures 55.3% of observed attrition in the test set |

## Portfolio Differentiators

- Production-style folder structure with reusable `src/` modules.
- Reproducible `scripts/run_pipeline.py` workflow.
- Streamlit dashboard with six executive analytics pages.
- Dashboard screenshots committed under `outputs/dashboard_screenshots/`.
- Model card, fairness report, data quality report, and executive insight report.
- GitHub Actions workflow for automated validation.
- Python tests and static validation with `pyright`.
- Optional R scripts for statistical workforce analysis.

## Responsible Use Position

The model should only support structured HR review. It should not be used for punitive decisions, employee surveillance, automated employment action, or causal claims. This responsible framing is intentional because HR analytics is a sensitive decision domain.

## Ideal Roles This Project Supports

- Data Scientist
- Junior Data Scientist
- People Analytics Analyst
- HR Data Analyst
- Machine Learning Analyst
- Business Intelligence Analyst with ML exposure

