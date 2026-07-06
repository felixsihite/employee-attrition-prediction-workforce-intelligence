from __future__ import annotations

import streamlit as st

from components import (
    format_percentage_columns,
    insight,
    load_featured_data,
    load_metrics,
    load_risk_scores,
    metric_card,
    page_header,
    page_setup,
    risk_score_view,
    section_header,
    table_note,
)


page_setup("Executive Workforce Overview")
page_header(
    "Executive Workforce Overview",
    "A board-level view of workforce size, observed attrition, risk concentration, and the model operating threshold selected for HR review capacity.",
    "Executive overview",
)

df = load_featured_data()
risk = load_risk_scores()
metrics = load_metrics()

priority_queue = risk[risk["risk_decile"] <= 2]
critical_queue = risk[risk["risk_band"].eq("Critical")]
top_decile_attrition = (
    risk.loc[risk["risk_decile"].eq(1), "attrition_flag"].sum() / risk["attrition_flag"].sum()
    if risk["attrition_flag"].sum()
    else 0
)

cols = st.columns(4, gap="large")
with cols[0]:
    metric_card("Employees", f"{len(df):,}", "Validated records")
with cols[1]:
    metric_card("Observed Attrition", f"{int(df['attrition_flag'].sum()):,}", f"{df['attrition_flag'].mean():.1%} rate")
with cols[2]:
    metric_card("Priority Queue", f"{len(priority_queue):,}", "Top 20% risk deciles")
with cols[3]:
    metric_card("Selected Threshold", f"{metrics['threshold']:.2f}", "Validation-selected")

cols = st.columns(4, gap="large")
with cols[0]:
    metric_card("Critical Risk Band", f"{len(critical_queue):,}", "Probability above 75%")
with cols[1]:
    metric_card("Top Decile Capture", f"{top_decile_attrition:.1%}", "Observed attrition captured")
with cols[2]:
    metric_card("Recall", f"{metrics['recall']:.3f}", "Attrition class sensitivity")
with cols[3]:
    metric_card("Balanced Accuracy", f"{metrics['balanced_accuracy']:.3f}", "Class-aware evaluation")

insight(
    "The project prioritizes attrition recall and risk concentration while preserving human review before any HR action. Risk scores should trigger structured review, not automatic employment decisions."
)

section_header(
    "Priority Queue Snapshot",
    "Highest-risk employees are sorted by predicted probability. Explanations are heuristic summaries for review triage, while full model behavior is documented on the Explainable AI page.",
)
table_note("Columns are intentionally business-readable: identifier, department, role, risk probability, risk band, priority, and top risk explanation.")

st.dataframe(
    format_percentage_columns(risk_score_view(risk.head(25)), ["attrition_probability"]),
    width="stretch",
    hide_index=True,
    height=620,
)