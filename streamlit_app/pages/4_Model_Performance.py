from __future__ import annotations

import streamlit as st

from components import (
    PROJECT_ROOT,
    format_percentage_columns,
    load_csv,
    load_metrics,
    metric_card,
    page_header,
    page_setup,
    section_header,
    table_note,
)


page_setup("Model Performance")
page_header(
    "Model Performance",
    "Model selection, thresholding, and lift analysis for an imbalanced attrition classification problem where recall and intervention capacity matter more than accuracy alone.",
    "Model governance",
)

metrics = load_metrics()

cols = st.columns(5, gap="large")
for col, label, key, detail in zip(
    cols,
    ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F2"],
    ["roc_auc", "pr_auc", "recall", "precision", "fbeta_2"],
    ["Ranking quality", "Imbalanced metric", "Attrition capture", "Review precision", "Recall-weighted"],
):
    with col:
        metric_card(label, f"{metrics[key]:.3f}", detail)

cols = st.columns(4, gap="large")
with cols[0]:
    metric_card("Threshold", f"{metrics['threshold']:.2f}", "Validation-selected")
with cols[1]:
    metric_card("Selected Employees", f"{metrics['selected_employees']:,}", "Test-set positives")
with cols[2]:
    metric_card("Intervention Rate", f"{metrics['intervention_rate']:.1%}", "Capacity signal")
with cols[3]:
    metric_card("Balanced Accuracy", f"{metrics['balanced_accuracy']:.3f}", "Class-aware score")

section_header(
    "Model Comparison",
    "Candidate models are compared with cross-validation metrics and a validation selection score aligned with PR-AUC, recall, and ROC-AUC.",
)
comparison = load_csv("outputs/metrics/model_comparison.csv")
st.dataframe(
    format_percentage_columns(
        comparison,
        [
            "cv_roc_auc_mean",
            "cv_pr_auc_mean",
            "cv_recall_mean",
            "cv_balanced_accuracy_mean",
            "validation_roc_auc",
            "validation_pr_auc",
            "validation_recall_at_0_50",
            "validation_precision_at_0_50",
            "selection_score",
            "baseline_pr_auc",
        ],
    ),
    width="stretch",
    hide_index=True,
    height=220,
)

section_header(
    "Threshold Optimization and Lift",
    "Threshold tuning is performed on validation data. Lift and gain analysis shows whether the model concentrates observed attrition in the highest-risk slices.",
)
left, right = st.columns(2, gap="large")
with left:
    st.image(str(PROJECT_ROOT / "outputs" / "charts" / "threshold_optimization.png"), width="stretch")
with right:
    st.image(str(PROJECT_ROOT / "outputs" / "charts" / "lift_gain_chart.png"), width="stretch")

threshold_table = load_csv("outputs/metrics/threshold_optimization.csv")
lift_table = load_csv("outputs/metrics/lift_gain_analysis.csv")
tab_threshold, tab_lift = st.tabs(["Threshold Table", "Lift and Gain Table"])
with tab_threshold:
    table_note("Use this table to choose an operating threshold that matches HR intervention capacity without tuning on the final test set.")
    st.dataframe(
        format_percentage_columns(
            threshold_table,
            ["roc_auc", "pr_auc", "precision", "recall", "f1", "fbeta_2", "balanced_accuracy", "intervention_rate"],
        ),
        width="stretch",
        hide_index=True,
        height=430,
    )
with tab_lift:
    table_note("Risk decile 1 represents the highest predicted attrition risk. Capture rate measures the share of observed attrition found by each cumulative review slice.")
    st.dataframe(
        format_percentage_columns(
            lift_table,
            ["attrition_rate", "lift", "cumulative_attrition_capture", "cumulative_population_share", "gain"],
        ),
        width="stretch",
        hide_index=True,
        height=430,
    )

section_header("Diagnostic Curves", "ROC, precision-recall, and confusion matrix artifacts from the final holdout evaluation.")
cols = st.columns(3, gap="large")
for col, image in zip(cols, ["roc_curve.png", "precision_recall_curve.png", "confusion_matrix.png"]):
    with col:
        st.image(str(PROJECT_ROOT / "outputs" / "charts" / image), width="stretch")