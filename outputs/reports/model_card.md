# Model Card

## Intended Use
The model ranks employees by attrition risk to support proactive People Analytics review, workforce planning, and transparent retention prioritization.

## Not Intended For
This project does not claim guaranteed retention, real-time HR monitoring, causal proof, or actual ROI. Any cost scenario should be treated as an assumption-based simulation.

## Model Selection
Selected model: **Logistic Regression**.
The selection balances PR-AUC, recall, and ROC-AUC because the target class is imbalanced.

## Test Metrics at Selected Threshold
- Threshold: 0.50
- ROC-AUC: 0.829
- PR-AUC: 0.558
- Precision: 0.398
- Recall: 0.702
- F1: 0.508
- F2: 0.609
- Balanced accuracy: 0.750
- Confusion matrix: TN=197, FP=50, FN=14, TP=33

## Monitoring Recommendations
Revalidate model performance before any real HR use, review group-level error patterns, document human review decisions, and retrain only with approved HR governance.
