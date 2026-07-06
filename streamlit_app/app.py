from __future__ import annotations

from typing import cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import (
    THEME,
    attrition_rate_by,
    bar_chart,
    format_percentage_columns,
    insight,
    load_featured_data,
    load_metrics,
    load_risk_scores,
    metric_card,
    page_header,
    page_setup,
    render_chart,
    risk_score_view,
    section_header,
)


page_setup("Employee Attrition Workforce Intelligence")

page_header(
    "Employee Attrition Prediction & Workforce Intelligence",
    "Executive People Analytics dashboard for attrition risk prioritization, workforce segmentation, explainable model monitoring, and responsible HR review.",
    "Executive dashboard",
)

df = load_featured_data()
risk = load_risk_scores()
metrics = load_metrics()

total_employees = len(df)
attrition_count = int(df["attrition_flag"].sum())
attrition_rate = df["attrition_flag"].mean()
high_risk = int((risk["risk_decile"] <= 2).sum())
avg_income = df["MonthlyIncome"].mean()
avg_tenure = df["YearsAtCompany"].mean()
avg_satisfaction = df["satisfaction_score"].mean()
top_decile_capture = (
    risk.loc[risk["risk_decile"].eq(1), "attrition_flag"].sum() / risk["attrition_flag"].sum()
    if risk["attrition_flag"].sum()
    else 0
)

cols = st.columns(4, gap="large")
with cols[0]:
    metric_card("Total Employees", f"{total_employees:,}", "Validated IBM HR Analytics population")
with cols[1]:
    metric_card("Observed Attrition", f"{attrition_rate:.1%}", f"{attrition_count:,} attrition cases")
with cols[2]:
    metric_card("Priority Review Queue", f"{high_risk:,}", "Top two model risk deciles")
with cols[3]:
    metric_card("Model PR-AUC", f"{metrics['pr_auc']:.3f}", "Primary imbalanced-class metric")

cols = st.columns(4, gap="large")
with cols[0]:
    metric_card("Average Monthly Income", f"${avg_income:,.0f}", "Source compensation field")
with cols[1]:
    metric_card("Average Tenure", f"{avg_tenure:.1f} years", "Years at company")
with cols[2]:
    metric_card("Average Satisfaction", f"{avg_satisfaction:.2f}/4", "Composite satisfaction score")
with cols[3]:
    metric_card("Top Decile Capture", f"{top_decile_capture:.1%}", "Observed attrition in highest-risk decile")

insight(
    "This dashboard ranks attrition risk for structured retention review. It is designed for transparent HR decision support, not automated employee judgment or punitive action."
)

section_header(
    "Executive Risk Signals",
    "The two charts below show where observed attrition is concentrated and how predicted risk separates the employee population into reviewable deciles.",
)

left, right = st.columns(2, gap="large")
with left:
    render_chart(
        bar_chart(
            attrition_rate_by(df, "Department"),
            "Department",
            "attrition_rate",
            "Attrition Rate by Department",
            THEME["accent_blue"],
            height=430,
        )
    )

with right:
    decile = cast(
        pd.DataFrame,
        risk.groupby("risk_decile", as_index=False, sort=True).agg(
            employees=("EmployeeNumber", "size"),
            mean_probability=("attrition_probability", "mean"),
            observed_attrition_rate=("attrition_flag", "mean"),
        ),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=decile["risk_decile"],
            y=decile["mean_probability"],
            mode="lines+markers",
            name="Predicted risk",
            line={"color": THEME["risk_red"], "width": 3},
            marker={"size": 9},
            hovertemplate="Decile %{x}<br>Mean probability %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=decile["risk_decile"],
            y=decile["observed_attrition_rate"],
            name="Observed attrition",
            marker={"color": "#8FB4F3", "line": {"color": "#2563EB", "width": 1}},
            hovertemplate="Decile %{x}<br>Observed attrition %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="#F7FAFC",
        plot_bgcolor="#F7FAFC",
        title={"text": "Risk Separation by Decile", "x": 0.02, "xanchor": "left"},
        font={"color": THEME["text_primary"]},
        legend={"orientation": "h", "y": -0.24, "font": {"color": THEME["text_primary"], "size": 12}},
        yaxis_tickformat=".0%",
        xaxis={
            "title": "Risk decile (1 = highest risk)",
            "tickfont": {"color": THEME["text_primary"]},
            "title_font": {"color": THEME["text_secondary"]},
        },
        yaxis={
            "title": "Rate",
            "tickformat": ".0%",
            "gridcolor": "#CED8E2",
            "tickfont": {"color": THEME["text_primary"]},
            "title_font": {"color": THEME["text_secondary"]},
        },
        height=430,
        margin={"l": 66, "r": 36, "t": 74, "b": 96},
    )
    render_chart(fig)

section_header(
    "Priority Review Preview",
    "A compact view of the highest-priority employee risk queue. The downloadable full table is available on the Employee Risk Scoring page.",
)
preview = risk_score_view(risk.head(12))
st.dataframe(
    format_percentage_columns(preview, ["attrition_probability"]),
    width="stretch",
    hide_index=True,
    height=420,
)