# Data Quality and Leakage Audit

Dataset source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Raw Dataset Health
- Rows: 1,470
- Columns: 35
- Missing values: 0
- Duplicate rows: 0
- Target distribution: No = 1,233, Yes = 237
- Attrition rate: 16.12%

## Modeling Exclusions
- `Attrition` is the target and is never used as a feature.
- `EmployeeNumber` is retained only as an identifier for reporting and risk scoring.
- `EmployeeCount`, `Over18`, and `StandardHours` are constant columns and are excluded from modeling.

## Leakage Control
The model uses a stratified train/test split, performs threshold selection only on the validation split, and reports final metrics on the untouched test split.
Correlation and feature importance are presented as associations, not causal proof.
