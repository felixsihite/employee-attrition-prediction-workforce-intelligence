from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import shorten
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

THEME = {
    "background_light": "#D6E4F0",
    "card": "#F7FAFC",
    "secondary_surface": "#E2E2E2",
    "text_primary": "#172033",
    "text_secondary": "#52616B",
    "primary_navy": "#0B1F33",
    "accent_blue": "#2563EB",
    "accent_teal": "#1F7A8C",
    "positive_green": "#2E7D32",
    "warning_amber": "#B7791F",
    "risk_red": "#C62828",
    "neutral_gray": "#64748B",
    "dark_background": "#0B1F33",
    "dark_surface": "#132F4C",
    "dark_text": "#F8FAFC",
    "dark_secondary_text": "#CBD5E1",
    "dark_accent_blue": "#38BDF8",
    "dark_positive_green": "#22C55E",
    "dark_warning_amber": "#F59E0B",
    "dark_risk_red": "#EF4444",
}

CHART_TEMPLATE = {
    "paper_bgcolor": "#F7FAFC",
    "plot_bgcolor": "#F7FAFC",
    "font": {"family": "Inter, Segoe UI, Arial, sans-serif", "color": THEME["text_primary"], "size": 12},
    "title": {"font": {"color": THEME["primary_navy"], "size": 17}, "x": 0.02, "xanchor": "left"},
    "margin": {"l": 72, "r": 42, "t": 72, "b": 86},
    "height": 460,
}


def chart_layout(**overrides) -> dict:
    layout = dict(CHART_TEMPLATE)
    layout.update(overrides)
    return layout


def page_setup(title: str) -> None:
    st.set_page_config(page_title=title, page_icon=None, layout="wide")
    css_path = APP_DIR / "assets" / "theme.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str, kicker: str = "People Analytics") -> None:
    st.markdown(f"<div class='page-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def section_header(title: str, caption: str | None = None) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<div class='section-caption'>{caption}</div>", unsafe_allow_html=True)


def insight(text: str) -> None:
    st.markdown(f"<div class='insight-panel'>{text}</div>", unsafe_allow_html=True)


def table_note(text: str) -> None:
    st.markdown(f"<div class='table-note'>{text}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, detail: str = "") -> None:
    detail_html = f"<div class='metric-detail'>{detail}</div>" if detail else "<div class='metric-detail'>&nbsp;</div>"
    st.markdown(
        f"""
        <div class="metric-card">
          <div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
          </div>
          {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_featured_data() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "employee_attrition_featured_dataset.csv"
    if not path.exists():
        st.error("Run `python scripts/run_pipeline.py` before opening the dashboard.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_risk_scores() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "risk_scores" / "employee_attrition_risk_scores.csv"
    if not path.exists():
        st.error("Run `python scripts/run_pipeline.py` before opening the dashboard.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    path = PROJECT_ROOT / "outputs" / "metrics" / "test_metrics.json"
    if not path.exists():
        st.error("Run `python scripts/run_pipeline.py` before opening the dashboard.")
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(relative_path: str) -> pd.DataFrame:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        st.error(f"Required artifact is missing: {relative_path}")
        st.stop()
    return pd.read_csv(path)


def attrition_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        df.groupby(column, dropna=False)
        .agg(
            employees=("attrition_flag", "size"),
            attrition_count=("attrition_flag", "sum"),
            attrition_rate=("attrition_flag", "mean"),
        )
        .reset_index()
        .sort_values("attrition_rate", ascending=False)
    )
    grouped["attrition_count"] = grouped["attrition_count"].astype(int)
    return grouped


def wrap_label(value: object, width: int = 18) -> str:
    text = str(value).replace("_", " ")
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


def percent_axis_layout(fig: go.Figure) -> go.Figure:
    fig.update_yaxes(
        tickformat=".0%",
        gridcolor="#CED8E2",
        zeroline=False,
        title_font={"color": THEME["text_secondary"]},
        tickfont={"color": THEME["text_primary"]},
    )
    fig.update_xaxes(
        title_font={"color": THEME["text_secondary"]},
        tickfont={"color": THEME["text_primary"]},
    )
    return fig


def bar_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str = "#1F7A8C",
    *,
    horizontal: bool = False,
    height: int = 460,
    sort_by_value: bool = True,
) -> go.Figure:
    chart_frame = frame.copy()
    if sort_by_value:
        chart_frame = chart_frame.sort_values(y, ascending=horizontal)
    text_values = [f"{value:.1%}" if "rate" in y or "probability" in y else f"{value:,.0f}" for value in chart_frame[y]]

    if horizontal:
        trace = go.Bar(
            x=chart_frame[y],
            y=[wrap_label(value, 26) for value in chart_frame[x]],
            orientation="h",
            marker={"color": color, "line": {"color": "rgba(11,31,51,0.18)", "width": 1}},
            text=text_values,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Rate: %{x:.1%}<extra></extra>",
        )
        fig = go.Figure(trace)
        fig.update_layout(
            xaxis_tickformat=".0%",
            yaxis={"categoryorder": "array", "categoryarray": [wrap_label(v, 26) for v in chart_frame[x]]},
            margin={"l": 190, "r": 48, "t": 72, "b": 86},
        )
    else:
        trace = go.Bar(
            x=[wrap_label(value, 16) for value in chart_frame[x]],
            y=chart_frame[y],
            marker={"color": color, "line": {"color": "rgba(11,31,51,0.18)", "width": 1}},
            text=text_values,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Rate: %{y:.1%}<extra></extra>",
        )
        fig = go.Figure(trace)
        fig.update_layout(yaxis_tickformat=".0%")

    fig.update_layout(**chart_layout(title=title, height=height))
    fig.update_yaxes(range=[0, max(float(chart_frame[y].max()) * 1.24, 0.08)] if not horizontal else None)
    fig.update_xaxes(range=[0, max(float(chart_frame[y].max()) * 1.24, 0.08)] if horizontal else None)
    return percent_axis_layout(fig)


def risk_matrix_heatmap(df: pd.DataFrame) -> go.Figure:
    matrix = (
        df.pivot_table(index="Department", columns="JobRole", values="attrition_flag", aggfunc="mean")
        .fillna(0)
        .sort_index()
    )
    employee_counts = (
        df.pivot_table(index="Department", columns="JobRole", values="attrition_flag", aggfunc="size")
        .reindex_like(matrix)
        .fillna(0)
        .astype(int)
    )

    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    employee_counts.index = employee_counts.index.astype(str)
    employee_counts.columns = employee_counts.columns.astype(str)

    departments = [str(value) for value in matrix.index.tolist()]
    roles = [str(value) for value in matrix.columns.tolist()]
    rate_lookup: dict[str, dict[str, float]] = {
        str(department): {str(role): float(value) for role, value in row.items()}
        for department, row in matrix.to_dict(orient="index").items()
    }
    count_lookup: dict[str, dict[str, int]] = {
        str(department): {str(role): int(value) for role, value in row.items()}
        for department, row in employee_counts.to_dict(orient="index").items()
    }
    z = [[rate_lookup[department][role] for role in roles] for department in departments]

    hover: list[list[str]] = []
    annotations: list[dict[str, Any]] = []
    x_labels = [wrap_label(role, 14) for role in roles]
    y_labels = [wrap_label(department, 20) for department in departments]
    for department in departments:
        hover_row: list[str] = []
        for role in roles:
            rate = rate_lookup[department][role]
            count = count_lookup[department][role]
            if count:
                hover_row.append(
                    f"Department: {department}<br>Job role: {role}<br>Employees: {count:,}<br>Attrition rate: {rate:.1%}"
                )
            else:
                hover_row.append(
                    f"Department: {department}<br>Job role: {role}<br>No employee records in this combination"
                )
            if count > 0:
                annotations.append(
                    {
                        "x": wrap_label(role, 14),
                        "y": wrap_label(department, 20),
                        "text": f"{rate:.0%}",
                        "showarrow": False,
                        "font": {
                            "color": "#F8FAFC" if rate >= 0.28 else THEME["primary_navy"],
                            "size": 12,
                            "family": "Inter, Segoe UI, Arial, sans-serif",
                        },
                    }
                )
            else:
                annotations.append(
                    {
                        "x": wrap_label(role, 14),
                        "y": wrap_label(department, 20),
                        "text": "-",
                        "showarrow": False,
                        "font": {
                            "color": "#8AA0B4",
                            "size": 13,
                            "family": "Inter, Segoe UI, Arial, sans-serif",
                        },
                    }
                )
        hover.append(hover_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale=[
                [0.0, "#F7FAFC"],
                [0.2, "#D6E4F0"],
                [0.45, "#BFD7EA"],
                [0.7, "#1F7A8C"],
                [1.0, "#0B1F33"],
            ],
            colorbar={
                "title": {"text": "Attrition<br>rate", "font": {"color": THEME["text_primary"]}},
                "tickformat": ".0%",
                "tickfont": {"color": THEME["text_primary"]},
                "len": 0.76,
            },
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        **chart_layout(
            title="Department and Job Role Risk Matrix",
            height=660,
            margin={"l": 132, "r": 70, "t": 82, "b": 160},
            xaxis={"title": "Job Role", "side": "bottom", "tickangle": 0, "automargin": True},
            xaxis_tickfont={"color": THEME["text_primary"], "size": 11},
            xaxis_title_font={"color": THEME["text_secondary"], "size": 12},
            yaxis={"title": "Department", "automargin": True},
            yaxis_tickfont={"color": THEME["text_primary"], "size": 11},
            yaxis_title_font={"color": THEME["text_secondary"], "size": 12},
            annotations=annotations,
        ),
    )
    fig.add_annotation(
        x=0,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text="Cell labels show observed attrition rate. '-' cells mean no employee records exist for that department-role combination.",
        font={"size": 12, "color": THEME["text_primary"]},
    )
    return fig


def format_percentage_columns(frame: pd.DataFrame, columns: list[str]) -> Any:
    formatter = {column: "{:.1%}" for column in columns if column in frame.columns}
    number_formatter = {
        column: "{:,.0f}"
        for column in frame.columns
        if column not in formatter and pd.api.types.is_numeric_dtype(frame[column])
    }
    return frame.style.format({**formatter, **number_formatter})


def risk_score_view(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "EmployeeNumber",
        "Department",
        "JobRole",
        "OverTime",
        "BusinessTravel",
        "MonthlyIncome",
        "YearsAtCompany",
        "attrition_probability",
        "risk_decile",
        "risk_band",
        "intervention_priority",
        "risk_explanation",
    ]
    view = cast(pd.DataFrame, frame.loc[:, columns].copy())
    explanation = cast(pd.Series, view["risk_explanation"]).to_list()
    view.loc[:, "risk_explanation"] = [shorten(str(value), width=92, placeholder="...") for value in explanation]
    return view


def chart_title_text(fig: go.Figure) -> str:
    layout = fig.to_plotly_json().get("layout", {})
    if not isinstance(layout, dict):
        return ""
    title = layout.get("title", {})
    if isinstance(title, dict):
        text = title.get("text")
    else:
        text = title
    return str(text) if text else ""


def render_chart(fig: go.Figure) -> None:
    title = chart_title_text(fig)
    if title:
        st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
        fig.update_layout(title=None)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})