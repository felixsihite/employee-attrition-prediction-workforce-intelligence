from __future__ import annotations

import streamlit as st

from components import (
    THEME,
    attrition_rate_by,
    bar_chart,
    format_percentage_columns,
    insight,
    load_featured_data,
    page_header,
    page_setup,
    render_chart,
    section_header,
)


page_setup("Attrition Intelligence")
page_header(
    "Attrition Intelligence",
    "Compare observed attrition across organizational, workload, tenure, compensation, and engagement dimensions without treating correlation as causation.",
    "Workforce diagnostics",
)

df = load_featured_data()

section_header(
    "Organizational and Role Risk",
    "Long labels are shown as horizontal bars so department and job role patterns remain readable on desktop and laptop screens.",
)

left, right = st.columns(2, gap="large")
with left:
    render_chart(
        bar_chart(
            attrition_rate_by(df, "JobRole"),
            "JobRole",
            "attrition_rate",
            "Attrition Rate by Job Role",
            THEME["accent_teal"],
            horizontal=True,
            height=560,
        )
    )
with right:
    render_chart(
        bar_chart(
            attrition_rate_by(df, "Department"),
            "Department",
            "attrition_rate",
            "Attrition Rate by Department",
            THEME["accent_blue"],
            height=560,
        )
    )

section_header(
    "Workload, Mobility, and Tenure Signals",
    "These views help HR teams focus review conversations on workload, travel intensity, career stage, and compensation bands.",
)

cols = st.columns(2, gap="large")
with cols[0]:
    render_chart(
        bar_chart(
            attrition_rate_by(df, "OverTime"),
            "OverTime",
            "attrition_rate",
            "Attrition Rate by Overtime",
            THEME["risk_red"],
            height=410,
        )
    )
    render_chart(
        bar_chart(
            attrition_rate_by(df, "tenure_band"),
            "tenure_band",
            "attrition_rate",
            "Attrition Rate by Tenure Band",
            THEME["accent_blue"],
            height=430,
        )
    )
with cols[1]:
    render_chart(
        bar_chart(
            attrition_rate_by(df, "BusinessTravel"),
            "BusinessTravel",
            "attrition_rate",
            "Attrition Rate by Business Travel",
            THEME["warning_amber"],
            height=410,
        )
    )
    render_chart(
        bar_chart(
            attrition_rate_by(df, "income_band"),
            "income_band",
            "attrition_rate",
            "Attrition Rate by Income Band",
            THEME["positive_green"],
            height=430,
        )
    )

section_header(
    "Detailed Segment Table",
    "Sorted observed attrition rates provide a compact audit trail behind the executive charts.",
)
summary = attrition_rate_by(df, "attrition_risk_segment").rename(columns={"attrition_risk_segment": "segment"})
st.dataframe(format_percentage_columns(summary, ["attrition_rate"]), width="stretch", hide_index=True)

insight(
    "The strongest segment patterns should be used as starting points for HR discovery conversations. They are not proof that one factor alone causes resignation."
)