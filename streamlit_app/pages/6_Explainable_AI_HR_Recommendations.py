from __future__ import annotations

import streamlit as st

from components import (
    PROJECT_ROOT,
    format_percentage_columns,
    insight,
    load_csv,
    metric_card,
    page_header,
    page_setup,
    section_header,
    table_note,
)


page_setup("Explainable AI & HR Recommendations")
page_header(
    "Explainable AI & HR Recommendations",
    "Interpret model behavior, inspect group-level risk concentration, and convert attrition intelligence into responsible HR review actions.",
    "Explainability and governance",
)

importance = load_csv("outputs/metrics/permutation_importance.csv")
top_drivers = importance.head(5)

cols = st.columns(5, gap="large")
for col, (_, row) in zip(cols, top_drivers.iterrows()):
    with col:
        feature = str(row["feature"])
        importance_value = float(row["importance_mean"])
        metric_card(
            feature,
            f"{importance_value:.3f}",
            "Average precision impact",
        )

insight(
    "Feature importance is model behavior evidence, not causal proof. Use it to make review conversations more transparent and to prioritize deeper HR analysis."
)

section_header(
    "Model Driver Importance",
    "Permutation importance measures how much model quality drops when each feature is shuffled on the holdout set.",
)
st.image(str(PROJECT_ROOT / "outputs" / "charts" / "feature_importance.png"), width="stretch")
st.dataframe(
    importance.head(20).style.format({"importance_mean": "{:.4f}", "importance_std": "{:.4f}"}),
    width="stretch",
    hide_index=True,
    height=520,
)

section_header(
    "Fairness and Group-Level Monitoring",
    "Group summaries help reviewers inspect risk concentration. These are monitoring views, not automated fairness certification.",
)
tab_gender, tab_department, tab_jobrole = st.tabs(["Gender", "Department", "Job Role"])
with tab_gender:
    gender = load_csv("outputs/metrics/fairness_by_gender.csv")
    st.dataframe(
        format_percentage_columns(gender, ["actual_attrition_rate", "mean_attrition_probability", "high_risk_share"]),
        width="stretch",
        hide_index=True,
    )
with tab_department:
    department = load_csv("outputs/metrics/fairness_by_department.csv")
    st.dataframe(
        format_percentage_columns(department, ["actual_attrition_rate", "mean_attrition_probability", "high_risk_share"]),
        width="stretch",
        hide_index=True,
    )
with tab_jobrole:
    jobrole = load_csv("outputs/metrics/fairness_by_jobrole.csv")
    st.dataframe(
        format_percentage_columns(jobrole, ["actual_attrition_rate", "mean_attrition_probability", "high_risk_share"]),
        width="stretch",
        hide_index=True,
        height=420,
    )

section_header("Responsible HR Recommendations")
table_note("Recommended actions are intentionally framed as human review workflows, not automated decisions.")
st.markdown(
    """
    - Prioritize employees in the highest risk deciles for structured HR partner review.
    - Review overtime-heavy roles where attrition rates and model risk are elevated.
    - Pair low satisfaction signals with manager context, engagement history, and voluntary feedback.
    - Review career progression patterns for employees with long promotion gaps or stagnation flags.
    - Use group-level monitoring before any operational use to avoid unfair risk concentration.
    - Document human review outcomes so future model iterations can be audited responsibly.
    """
)