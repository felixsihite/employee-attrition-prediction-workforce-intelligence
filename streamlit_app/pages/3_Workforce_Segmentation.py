from __future__ import annotations

from typing import cast

import pandas as pd
import streamlit as st

from components import (
    THEME,
    attrition_rate_by,
    bar_chart,
    format_percentage_columns,
    load_featured_data,
    page_header,
    page_setup,
    render_chart,
    risk_matrix_heatmap,
    section_header,
    table_note,
)


page_setup("Workforce Segmentation")
page_header(
    "Workforce Segmentation",
    "Segment the workforce by risk profile, age band, department, and job role to support focused retention review and workforce planning.",
    "Segmentation intelligence",
)

df = load_featured_data()

section_header(
    "Department and Job Role Risk Matrix",
    "Primary segmentation view. Labels are wrapped, values are printed inside cells, and hover text gives employee counts for every department-role combination.",
)
render_chart(risk_matrix_heatmap(df))

section_header(
    "Risk Segment and Age Patterns",
    "Segment-level views are designed for quick comparison. Bars show observed attrition rate and retain clear labels on dark or light dashboard modes.",
)

left, right = st.columns(2, gap="large")
with left:
    render_chart(
        bar_chart(
            attrition_rate_by(df, "attrition_risk_segment"),
            "attrition_risk_segment",
            "attrition_rate",
            "Attrition Rate by HR Risk Segment",
            THEME["accent_teal"],
            horizontal=True,
            height=500,
        )
    )
with right:
    render_chart(
        bar_chart(
            attrition_rate_by(df, "age_band"),
            "age_band",
            "attrition_rate",
            "Attrition Rate by Age Band",
            THEME["accent_blue"],
            height=500,
        )
    )

section_header(
    "Segment Audit Table",
    "This table gives the numeric detail behind the visual segmentation layer for review, documentation, and portfolio evaluation.",
)
table_note("Risk segment definitions come from transparent HR feature rules such as overtime, early tenure, satisfaction flags, stock options, and promotion gap indicators.")

segment_table = cast(
    pd.DataFrame,
    attrition_rate_by(df, "attrition_risk_segment").rename(columns={"attrition_risk_segment": "risk_segment"}),
)
segment_table["population_share"] = segment_table["employees"] / len(df)
segment_table = cast(
    pd.DataFrame,
    segment_table.loc[:, ["risk_segment", "employees", "population_share", "attrition_count", "attrition_rate"]],
)
st.dataframe(
    format_percentage_columns(segment_table, ["population_share", "attrition_rate"]),
    width="stretch",
    hide_index=True,
    height=260,
)