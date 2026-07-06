from __future__ import annotations

from typing import cast

import pandas as pd
import streamlit as st

from components import (
    format_percentage_columns,
    insight,
    load_risk_scores,
    metric_card,
    page_header,
    page_setup,
    risk_score_view,
    section_header,
    table_note,
)


page_setup("Employee Risk Scoring")
page_header(
    "Employee Risk Scoring",
    "A prioritized review queue that ranks employees by predicted attrition probability, risk band, decile, and transparent HR-facing risk explanations.",
    "Risk scoring",
)

risk = load_risk_scores()


def sorted_unique_strings(column: str) -> list[str]:
    return sorted(risk[column].dropna().astype(str).unique().tolist())


def filter_equals(frame: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    mask = cast(pd.Series, frame[column]).astype(str).eq(value)
    return cast(pd.DataFrame, frame.loc[mask].copy())


def filter_in(frame: pd.DataFrame, column: str, values: list[str]) -> pd.DataFrame:
    mask = cast(pd.Series, frame[column]).astype(str).isin(values)
    return cast(pd.DataFrame, frame.loc[mask].copy())


st.sidebar.markdown("### Review Filters")
departments = ["All", *sorted_unique_strings("Department")]
roles = ["All", *sorted_unique_strings("JobRole")]
department = st.sidebar.selectbox("Department", departments)
role = st.sidebar.selectbox("Job Role", roles)
risk_band_options = sorted_unique_strings("risk_band")
risk_band = st.sidebar.multiselect(
    "Risk Band",
    risk_band_options,
    default=risk_band_options,
)
max_rows = st.sidebar.slider("Rows to show", min_value=25, max_value=300, value=100, step=25)

filtered = cast(pd.DataFrame, risk.copy())
if department != "All":
    filtered = filter_equals(filtered, "Department", str(department))
if role != "All":
    filtered = filter_equals(filtered, "JobRole", str(role))
if risk_band:
    filtered = filter_in(filtered, "risk_band", [str(value) for value in risk_band])

filtered = cast(pd.DataFrame, filtered.sort_values(by=["attrition_probability"], ascending=False))

cols = st.columns(4, gap="large")
with cols[0]:
    metric_card("Filtered Employees", f"{len(filtered):,}", "After sidebar filters")
with cols[1]:
    metric_card("Mean Risk", f"{filtered['attrition_probability'].mean():.1%}" if len(filtered) else "0.0%", "Predicted probability")
with cols[2]:
    metric_card("Top 20% Decile Share", f"{(filtered['risk_decile'] <= 2).mean():.1%}" if len(filtered) else "0.0%", "Priority queue density")
with cols[3]:
    metric_card("Observed Attrition", f"{filtered['attrition_flag'].mean():.1%}" if len(filtered) else "0.0%", "Historical label rate")

insight(
    "Use this table for structured review planning. A high score means the employee resembles historical attrition patterns; it is not a final judgment about intent or performance."
)

section_header(
    "Filtered Risk Queue",
    "The queue is sorted by predicted attrition probability. Risk explanations summarize observable HR factors for review conversations.",
)
table_note("For privacy-conscious presentation, the table keeps the employee identifier and HR review fields visible while avoiding unnecessary raw feature overload.")

view = risk_score_view(filtered.head(max_rows))
st.dataframe(
    format_percentage_columns(view, ["attrition_probability"]),
    width="stretch",
    hide_index=True,
    height=650,
)

st.download_button(
    "Download filtered scored employee list",
    filtered.to_csv(index=False),
    file_name="employee_attrition_risk_scores_filtered.csv",
    mime="text/csv",
)