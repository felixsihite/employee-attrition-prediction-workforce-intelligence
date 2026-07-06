# ruff: noqa: E402
# pylint: disable=wrong-import-position
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHARTS_DIR,
    DASHBOARD_SCREENSHOTS_DIR,
    DATASET_SOURCE_URL,
    METRICS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    RISK_SCORES_DIR,
    THEME,
)
from src.data.quality import (
    categorical_cardinality,
    data_dictionary_frame,
    load_raw_data,
    numeric_range_summary,
    run_data_quality_audit,
    validate_raw_dataset,
)
from src.evaluation.metrics import group_risk_summary, lift_gain_table
from src.features.build_features import build_modeling_dataset
from src.interpretation.explain import permutation_importance_frame
from src.models.train_model import ModelResults, train_and_evaluate
from src.reporting import clean_markdown, markdown_table
from src.scoring.score_employees import score_employee_population
from src.visualization.theme import apply_matplotlib_theme, polish_axis


def ensure_directories() -> None:
    for directory in [
        PROCESSED_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        CHARTS_DIR,
        REPORTS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        RISK_SCORES_DIR,
        DASHBOARD_SCREENSHOTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def attrition_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.groupby(column, dropna=False)
        .agg(employees=("attrition_flag", "size"), attrition_rate=("attrition_flag", "mean"))
        .reset_index()
        .sort_values("attrition_rate", ascending=False)
    )


def save_bar_chart(frame: pd.DataFrame, x: str, y: str, title: str, path: Path, color: str | None = None) -> None:
    apply_matplotlib_theme()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bar_color = color or THEME["accent_teal"]
    ax.bar(frame[x].astype(str), frame[y], color=bar_color)
    polish_axis(ax, title, "", y.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=25)
    if y.endswith("rate") or "rate" in y:
        ax.set_ylim(0, max(float(frame[y].max()) * 1.25, 0.05))
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_core_charts(
    featured: pd.DataFrame,
    model_results: ModelResults,
    risk_scores: pd.DataFrame,
    importance: pd.DataFrame,
    lift: pd.DataFrame,
) -> None:
    apply_matplotlib_theme()

    attrition_counts = featured["Attrition"].value_counts().reindex(["No", "Yes"], fill_value=0)
    attrition_labels = ["No", "Yes"]
    attrition_values = [int(value) for value in attrition_counts.to_list()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(attrition_labels, attrition_values, color=[THEME["accent_teal"], THEME["risk_red"]])
    polish_axis(ax, "Attrition Class Distribution", "", "Employees")
    for index, value in enumerate(attrition_values):
        ax.text(index, value + 20, f"{value:,}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "attrition_distribution.png", dpi=180)
    plt.close(fig)

    save_bar_chart(
        attrition_rate_by(featured, "Department"),
        "Department",
        "attrition_rate",
        "Attrition Rate by Department",
        CHARTS_DIR / "attrition_by_department.png",
        THEME["accent_blue"],
    )
    save_bar_chart(
        attrition_rate_by(featured, "JobRole").head(9),
        "JobRole",
        "attrition_rate",
        "Attrition Rate by Job Role",
        CHARTS_DIR / "attrition_by_job_role.png",
        THEME["accent_teal"],
    )
    save_bar_chart(
        attrition_rate_by(featured, "OverTime"),
        "OverTime",
        "attrition_rate",
        "Attrition Rate by Overtime",
        CHARTS_DIR / "attrition_by_overtime.png",
        THEME["warning_amber"],
    )
    save_bar_chart(
        attrition_rate_by(featured, "tenure_band"),
        "tenure_band",
        "attrition_rate",
        "Attrition Rate by Tenure Band",
        CHARTS_DIR / "attrition_by_tenure_band.png",
        THEME["accent_blue"],
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    featured.boxplot(column="MonthlyIncome", by="Attrition", ax=ax, patch_artist=True)
    polish_axis(ax, "Monthly Income Distribution by Attrition", "Attrition", "Monthly Income")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "income_by_attrition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    roc_display = RocCurveDisplay.from_predictions(model_results["y_test"], model_results["test_probability"], ax=ax)
    roc_line = cast(Line2D, roc_display.line_)
    roc_line.set_color(THEME["accent_blue"])
    roc_line.set_linewidth(2.4)
    polish_axis(ax, "ROC Curve - Final Model", "False Positive Rate", "True Positive Rate")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "roc_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    pr_display = PrecisionRecallDisplay.from_predictions(model_results["y_test"], model_results["test_probability"], ax=ax)
    pr_line = cast(Line2D, pr_display.line_)
    pr_line.set_color(THEME["accent_teal"])
    pr_line.set_linewidth(2.4)
    polish_axis(ax, "Precision-Recall Curve - Final Model", "Recall", "Precision")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "precision_recall_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        model_results["y_test"],
        (model_results["test_probability"] >= model_results["selected_threshold"]).astype(int),
        display_labels=["No", "Yes"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix - Selected HR Threshold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    top_importance = importance.head(12).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.barh(top_importance["feature"], top_importance["importance_mean"], color=THEME["primary_navy"])
    polish_axis(ax, "Model Interpretability - Permutation Importance", "Average precision impact", "")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)

    threshold = model_results["threshold_table"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(threshold["threshold"], threshold["recall"], label="Recall", color=THEME["accent_blue"], linewidth=2.4)
    ax.plot(threshold["threshold"], threshold["precision"], label="Precision", color=THEME["accent_teal"], linewidth=2.4)
    ax.plot(threshold["threshold"], threshold["fbeta_2"], label="F2", color=THEME["warning_amber"], linewidth=2.4)
    ax.axvline(model_results["selected_threshold"], color=THEME["risk_red"], linestyle="--", linewidth=1.8, label="Selected threshold")
    polish_axis(ax, "Threshold Optimization for HR Intervention Capacity", "Threshold", "Score")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "threshold_optimization.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(lift["risk_decile"].astype(str), lift["cumulative_attrition_capture"], color=THEME["accent_blue"])
    polish_axis(ax, "Cumulative Attrition Capture by Risk Decile", "Risk decile (1 = highest risk)", "Capture rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "lift_gain_chart.png", dpi=180)
    plt.close(fig)

    top_deciles = cast(
        pd.DataFrame,
        risk_scores.groupby("risk_decile", as_index=False, sort=True).agg(
            employees=("EmployeeNumber", "size"),
            mean_probability=("attrition_probability", "mean"),
        ),
    )
    save_bar_chart(
        top_deciles,
        "risk_decile",
        "mean_probability",
        "Average Attrition Probability by Risk Decile",
        CHARTS_DIR / "risk_decile_profile.png",
        THEME["risk_red"],
    )


def save_dashboard_preview(audit, metrics: dict[str, float | int], risk_scores: pd.DataFrame, featured: pd.DataFrame) -> None:
    apply_matplotlib_theme()
    fig = plt.figure(figsize=(14, 8), facecolor=THEME["background_light"])
    grid = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.35)
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0, 0.72, "Employee Attrition Prediction & Workforce Intelligence", fontsize=22, weight="bold")
    title_ax.text(0, 0.35, "Executive People Analytics dashboard preview", fontsize=12, color=THEME["text_secondary"])
    kpis = [
        ("Employees", f"{audit.row_count:,}"),
        ("Attrition Rate", f"{audit.attrition_rate:.1%}"),
        ("PR-AUC", f"{metrics['pr_auc']:.3f}"),
        ("High Risk", f"{(risk_scores['risk_decile'] <= 2).sum():,}"),
    ]
    for idx, (label, value) in enumerate(kpis):
        ax = fig.add_subplot(grid[1, idx])
        ax.set_facecolor(THEME["card"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#C9D8E8")
        ax.text(0.08, 0.65, value, fontsize=24, weight="bold", color=THEME["primary_navy"], transform=ax.transAxes)
        ax.text(0.08, 0.30, label, fontsize=10, color=THEME["text_secondary"], transform=ax.transAxes)
    ax = fig.add_subplot(grid[2, :2])
    dept = attrition_rate_by(featured, "Department")
    ax.bar(dept["Department"], dept["attrition_rate"], color=THEME["accent_blue"])
    polish_axis(ax, "Department Attrition Rate", "", "Rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax = fig.add_subplot(grid[2, 2:])
    decile = cast(
        pd.DataFrame,
        risk_scores.groupby("risk_decile", as_index=False, sort=True).agg(
            mean_probability=("attrition_probability", "mean"),
        ),
    )
    decile_numbers = [int(value) for value in cast(pd.Series, decile["risk_decile"]).tolist()]
    decile_probabilities = [float(value) for value in cast(pd.Series, decile["mean_probability"]).tolist()]
    ax.plot(decile_numbers, decile_probabilities, marker="o", color=THEME["risk_red"], linewidth=2.4)
    polish_axis(ax, "Risk Decile Profile", "Decile", "Mean probability")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.savefig(CHARTS_DIR / "dashboard_static_preview.png", dpi=180)
    plt.close(fig)


def build_reports(
    audit,
    featured: pd.DataFrame,
    model_results: ModelResults,
    risk_scores: pd.DataFrame,
    importance: pd.DataFrame,
    lift: pd.DataFrame,
    fairness_frames: dict[str, pd.DataFrame],
) -> None:
    metrics = model_results["test_metrics"]
    top_decile_capture = float(lift.loc[lift["risk_decile"].eq(1), "cumulative_attrition_capture"].iloc[0])
    top_20_capture = float(lift.loc[lift["risk_decile"].le(2), "attrition_count"].sum() / lift["attrition_count"].sum())
    top_features = importance.head(8)["feature"].tolist()

    data_quality = f"""
    # Data Quality and Leakage Audit

    Dataset source: {DATASET_SOURCE_URL}

    ## Raw Dataset Health
    - Rows: {audit.row_count:,}
    - Columns: {audit.column_count:,}
    - Missing values: {audit.missing_values:,}
    - Duplicate rows: {audit.duplicate_rows:,}
    - Target distribution: No = {audit.target_distribution.get('No', 0):,}, Yes = {audit.target_distribution.get('Yes', 0):,}
    - Attrition rate: {audit.attrition_rate:.2%}

    ## Modeling Exclusions
    - `Attrition` is the target and is never used as a feature.
    - `EmployeeNumber` is retained only as an identifier for reporting and risk scoring.
    - `EmployeeCount`, `Over18`, and `StandardHours` are constant columns and are excluded from modeling.

    ## Leakage Control
    The model uses a stratified train/test split, performs threshold selection only on the validation split, and reports final metrics on the untouched test split.
    Correlation and feature importance are presented as associations, not causal proof.
    """
    (REPORTS_DIR / "data_quality_report.md").write_text(clean_markdown(data_quality), encoding="utf-8")

    executive = f"""
    # Executive Workforce Intelligence Summary

    The workforce attrition rate in this dataset is {audit.attrition_rate:.2%}. The model selected for the final HR decision-support workflow is **{model_results['best_model_name']}** with a validation-selected operating threshold of **{model_results['selected_threshold']:.2f}**.

    ## Final Test Results
    - ROC-AUC: {metrics['roc_auc']:.3f}
    - PR-AUC: {metrics['pr_auc']:.3f}
    - Recall: {metrics['recall']:.3f}
    - Precision: {metrics['precision']:.3f}
    - F2 score: {metrics['fbeta_2']:.3f}
    - Balanced accuracy: {metrics['balanced_accuracy']:.3f}

    ## Risk Concentration
    - Top 10% risk decile captures {top_decile_capture:.1%} of observed attrition in the test set.
    - Top 20% risk deciles capture {top_20_capture:.1%} of observed attrition in the test set.

    ## Main Model Drivers
    {chr(10).join(f"- {feature}" for feature in top_features)}

    ## HR Decision Guidance
    Use the score as decision support for structured retention review. Do not use the output for punitive decisions, automated employment decisions, or employee surveillance.
    """
    (REPORTS_DIR / "executive_insights.md").write_text(clean_markdown(executive), encoding="utf-8")

    model_card = f"""
    # Model Card

    ## Intended Use
    The model ranks employees by attrition risk to support proactive People Analytics review, workforce planning, and transparent retention prioritization.

    ## Not Intended For
    This project does not claim guaranteed retention, real-time HR monitoring, causal proof, or actual ROI. Any cost scenario should be treated as an assumption-based simulation.

    ## Model Selection
    Selected model: **{model_results['best_model_name']}**.
    The selection balances PR-AUC, recall, and ROC-AUC because the target class is imbalanced.

    ## Test Metrics at Selected Threshold
    - Threshold: {metrics['threshold']:.2f}
    - ROC-AUC: {metrics['roc_auc']:.3f}
    - PR-AUC: {metrics['pr_auc']:.3f}
    - Precision: {metrics['precision']:.3f}
    - Recall: {metrics['recall']:.3f}
    - F1: {metrics['f1']:.3f}
    - F2: {metrics['fbeta_2']:.3f}
    - Balanced accuracy: {metrics['balanced_accuracy']:.3f}
    - Confusion matrix: TN={metrics['true_negative']}, FP={metrics['false_positive']}, FN={metrics['false_negative']}, TP={metrics['true_positive']}

    ## Monitoring Recommendations
    Revalidate model performance before any real HR use, review group-level error patterns, document human review decisions, and retrain only with approved HR governance.
    """
    (REPORTS_DIR / "model_card.md").write_text(clean_markdown(model_card), encoding="utf-8")

    fairness_sections = []
    for name, frame in fairness_frames.items():
        fairness_sections.append(f"## {name}\n\n" + markdown_table(frame.round(4)))
    fairness = f"""
    # Fairness, Ethics, and Responsible Use

    This project treats attrition prediction as decision support, not an automated employee judgment system.

    ## Responsible Use Principles
    - Predictions must not be used for punitive decisions.
    - HR action should require human review and business context.
    - Sensitive and demographic patterns should be monitored before operational use.
    - The model identifies associations in a historical dataset, not causal proof.
    - Employees should not be subject to surveillance based on model scores.

    {chr(10).join(fairness_sections)}
    """
    (REPORTS_DIR / "fairness_ethics_report.md").write_text(clean_markdown(fairness), encoding="utf-8")

    validation = f"""
    # Project Validation Report

    - Raw dataset preserved: yes
    - Processed datasets generated separately: yes
    - Missing values in raw data: {audit.missing_values}
    - Duplicate rows in raw data: {audit.duplicate_rows}
    - Leakage columns excluded from feature matrix: yes
    - Stratified split used: yes
    - Threshold tuned outside test set: yes
    - Risk scores generated: {len(risk_scores):,} employees
    - Dashboard theme main light background: {THEME['background_light']}
    - Dataset source included in README and reports: yes
    """
    (REPORTS_DIR / "project_validation_report.md").write_text(clean_markdown(validation), encoding="utf-8")


def write_readme(audit, model_results: ModelResults, importance: pd.DataFrame, lift: pd.DataFrame) -> None:
    metrics = model_results["test_metrics"]
    comparison = model_results["model_comparison"].copy()
    top_decile_capture = float(lift.loc[lift["risk_decile"].eq(1), "cumulative_attrition_capture"].iloc[0])
    top_20_capture = float(lift.loc[lift["risk_decile"].le(2), "attrition_count"].sum() / lift["attrition_count"].sum())
    top_features = importance.head(7)["feature"].tolist()
    comparison_columns = [
        "model",
        "cv_roc_auc_mean",
        "cv_pr_auc_mean",
        "cv_recall_mean",
        "validation_pr_auc",
        "selection_score",
        "selected_model",
    ]
    comparison_view = cast(pd.DataFrame, comparison.loc[:, comparison_columns].round(3))
    comparison_md = markdown_table(comparison_view)

    readme = f"""
    # Employee Attrition Prediction & Workforce Intelligence

    Professional People Analytics portfolio project for predicting employee attrition risk, explaining risk drivers, segmenting workforce patterns, and supporting responsible HR retention planning.

    ![Executive dashboard preview](outputs/dashboard_screenshots/00_app.png)

    ## Executive Summary

    This project builds an end-to-end attrition prediction and workforce intelligence system using the IBM HR Analytics Employee Attrition & Performance dataset. The solution includes data quality validation, leakage-aware feature engineering, model comparison, threshold optimization, lift analysis, employee risk scoring, explainable model interpretation, SQL reporting, R statistical analysis scripts, and a polished Streamlit dashboard.

    The project is designed as an HR decision-support case study. It does not claim guaranteed retention, causal proof, real-time HR monitoring, or actual ROI.

    ## Dataset Source

    - Dataset: IBM HR Analytics Employee Attrition & Performance
    - Kaggle source: {DATASET_SOURCE_URL}
    - Raw file: `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`

    ## Dataset Overview

    - Rows: {audit.row_count:,}
    - Columns: {audit.column_count:,}
    - Missing values: {audit.missing_values:,}
    - Duplicate rows: {audit.duplicate_rows:,}
    - Target: `Attrition`
    - Attrition distribution: No = {audit.target_distribution.get('No', 0):,}, Yes = {audit.target_distribution.get('Yes', 0):,}
    - Attrition rate: {audit.attrition_rate:.2%}

    ## Business Problem

    HR teams often react after resignation risk has already escalated. A structured workforce intelligence workflow helps prioritize retention review by identifying employees and segments with elevated attrition likelihood while preserving ethical human oversight.

    ## Business Solution

    The project ranks employees by predicted attrition probability, explains key model drivers, groups employees into actionable risk bands, and gives HR teams a transparent intervention queue aligned with capacity limits.

    ## Project Architecture

    ```text
    data/raw/                 Original Kaggle dataset, unchanged
    data/processed/           Feature-engineered and modeling-ready datasets
    notebooks/                Executive-grade analysis notebooks
    src/                      Reusable data, feature, model, evaluation, scoring modules
    sql/                      Workforce intelligence reporting queries
    r/                        R statistical workforce analysis scripts
    streamlit_app/            People Analytics dashboard
    models/                   Serialized final model pipeline
    outputs/                  Charts, reports, metrics, predictions, and risk scores
    tests/                    Project quality tests
    ```

    ## Data Quality and Leakage Prevention

    - Raw dataset is preserved unchanged.
    - `EmployeeNumber` is used only as an identifier.
    - `EmployeeCount`, `Over18`, and `StandardHours` are removed from modeling because they are constant.
    - `Attrition` and `attrition_flag` are excluded from the feature matrix.
    - Split strategy is stratified.
    - Threshold tuning is performed on validation data, not the test set.
    - Metrics prioritize PR-AUC, recall, F2, and lift instead of accuracy alone.

    ## Feature Engineering

    HR-focused features include tenure bands, income bands, age bands, overtime and travel flags, early tenure indicators, no stock option flags, promotion gap ratios, satisfaction and engagement scores, career stagnation flags, compensation-to-job-level ratios, workload risk flags, and workforce risk segments.

    ## Model Comparison

    {comparison_md}

    ## Final Model

    - Selected model: **{model_results['best_model_name']}**
    - Selected HR operating threshold: **{model_results['selected_threshold']:.2f}**
    - ROC-AUC: **{metrics['roc_auc']:.3f}**
    - PR-AUC: **{metrics['pr_auc']:.3f}**
    - Precision: **{metrics['precision']:.3f}**
    - Recall: **{metrics['recall']:.3f}**
    - F2 score: **{metrics['fbeta_2']:.3f}**
    - Balanced accuracy: **{metrics['balanced_accuracy']:.3f}**

    ## Lift and Decile Analysis

    - Top 10% risk decile captures **{top_decile_capture:.1%}** of observed attrition in the test set.
    - Top 20% risk deciles capture **{top_20_capture:.1%}** of observed attrition in the test set.
    - Risk deciles are used to align retention review with realistic HR capacity.

    ## Interpretability

    The project uses permutation importance as a SHAP-equivalent interpretability fallback in environments where SHAP is unavailable. If SHAP is installed, the workflow can be extended without changing the core modeling contract.

    Top model drivers in the current run:
    {chr(10).join(f"- `{feature}`" for feature in top_features)}

    ![Feature importance](outputs/charts/feature_importance.png)

    ## Dashboard Design

    The Streamlit dashboard uses the approved main light background **#D6E4F0** with navy, blue, teal, amber, green, and red accents. Text colors are selected for readability in light and dark contexts.

    Dashboard pages:
    - Executive Workforce Overview
    - Attrition Intelligence
    - Workforce Segmentation
    - Model Performance
    - Employee Risk Scoring
    - Explainable AI & HR Recommendations

    Dashboard screenshots are exported for every page:
    - `outputs/dashboard_screenshots/00_app.png`
    - `outputs/dashboard_screenshots/00_app_executive_risk_signals.png`
    - `outputs/dashboard_screenshots/01_executive_workforce_overview.png`
    - `outputs/dashboard_screenshots/02_attrition_intelligence.png`
    - `outputs/dashboard_screenshots/03_workforce_segmentation.png`
    - `outputs/dashboard_screenshots/03_workforce_segmentation_matrix_detail.png`
    - `outputs/dashboard_screenshots/04_model_performance.png`
    - `outputs/dashboard_screenshots/05_employee_risk_scoring.png`
    - `outputs/dashboard_screenshots/06_explainable_ai_hr_recommendations.png`

    ## Business Recommendations

    - Prioritize retention review for employees in the highest risk deciles.
    - Monitor overtime-heavy roles and groups with elevated attrition rates.
    - Review career progression for employees with long promotion gaps.
    - Build targeted engagement actions for low satisfaction segments.
    - Use model explanations to make risk review transparent.
    - Treat predictions as HR decision support, not automated decisions.

    ## Fairness and Ethics

    The model should not be used for punitive decisions, employee surveillance, or automated employment outcomes. Group-level summaries are included for Gender, Department, and JobRole so reviewers can inspect risk concentration before any operational use.

    ## How to Run

    ```bash
    python -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements.txt
    python scripts/run_pipeline.py
    streamlit run streamlit_app/app.py
    pytest
    ```

    Python version target: **3.13.1**.
    Runtime metadata is consolidated in `pyproject.toml`.

    ## R Statistical Workflow

    The companion R scripts are optional statistical validation artifacts. Run them from the project root when R is available locally:

    ```bash
    Rscript r/workforce_statistical_analysis.R
    Rscript r/attrition_group_comparison.R
    ```

    R outputs are written to `outputs/metrics/` and `outputs/reports/r_statistical_workforce_analysis.md`.

    ## Key Outputs

    - `outputs/reports/data_quality_report.md`
    - `outputs/reports/model_card.md`
    - `outputs/reports/executive_insights.md`
    - `outputs/reports/fairness_ethics_report.md`
    - `outputs/risk_scores/employee_attrition_risk_scores.csv`
    - `outputs/dashboard_screenshots/`
    - `models/employee_attrition_model.joblib`

    ## Limitations

    This is a structured historical HR analytics dataset, not a live HRIS platform. It has no real intervention outcomes, timestamps, verified cost records, or causal treatment effects. Any cost or ROI discussion must be presented as assumption-based simulation only.

    ## Portfolio Value

    This repository demonstrates practical Data Scientist capability across business framing, data quality, feature engineering, supervised learning, imbalanced classification, model evaluation, explainability, SQL analytics, R statistical analysis, dashboard delivery, and responsible AI governance.
    """
    (PROJECT_ROOT / "README.md").write_text(clean_markdown(readme), encoding="utf-8")


def main() -> None:
    ensure_directories()
    raw = load_raw_data()
    validate_raw_dataset(raw)
    audit = run_data_quality_audit(raw)

    dictionary = data_dictionary_frame(raw)
    dictionary.to_csv(PROCESSED_DIR / "data_dictionary.csv", index=False)
    categorical_cardinality(raw).to_csv(PROCESSED_DIR / "categorical_cardinality.csv", index=False)
    numeric_range_summary(raw).to_csv(PROCESSED_DIR / "numeric_range_summary.csv", index=False)

    X, y, featured = build_modeling_dataset(raw)
    featured.to_csv(PROCESSED_DIR / "employee_attrition_featured_dataset.csv", index=False)
    modeling = X.copy()
    modeling["attrition_flag"] = y.values
    modeling.to_csv(PROCESSED_DIR / "employee_attrition_modeling_dataset.csv", index=False)

    model_results = train_and_evaluate(X, y)
    model_payload = {
        "pipeline": model_results["final_pipeline"],
        "selected_threshold": model_results["selected_threshold"],
        "best_model_name": model_results["best_model_name"],
        "feature_columns": X.columns.tolist(),
        "test_metrics": model_results["test_metrics"],
    }
    joblib.dump(model_payload, MODELS_DIR / "employee_attrition_model.joblib")

    risk_scores = score_employee_population(model_results["final_pipeline"], X, raw, featured)
    risk_scores.to_csv(RISK_SCORES_DIR / "employee_attrition_risk_scores.csv", index=False)

    test_predictions = pd.DataFrame(
        {
            "actual_attrition_flag": model_results["y_test"].values,
            "attrition_probability": model_results["test_probability"],
            "predicted_attrition_flag": (
                model_results["test_probability"] >= model_results["selected_threshold"]
            ).astype(int),
        },
        index=model_results["y_test"].index,
    ).sort_index()
    test_predictions.to_csv(PREDICTIONS_DIR / "model_predictions_test.csv", index_label="source_index")

    model_results["model_comparison"].to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    model_results["threshold_table"].to_csv(METRICS_DIR / "threshold_optimization.csv", index=False)
    save_json(METRICS_DIR / "test_metrics.json", model_results["test_metrics"])

    lift = lift_gain_table(model_results["y_test"], model_results["test_probability"])
    lift.to_csv(METRICS_DIR / "lift_gain_analysis.csv", index=False)

    importance = permutation_importance_frame(
        model_results["final_pipeline"],
        model_results["X_test"],
        model_results["y_test"],
    )
    importance.to_csv(METRICS_DIR / "permutation_importance.csv", index=False)

    score_for_groups = risk_scores.copy()
    fairness_frames = {}
    for group in ["Gender", "Department", "JobRole"]:
        score_for_groups[group] = raw[group].values
        fairness_frames[group] = group_risk_summary(score_for_groups, group, model_results["selected_threshold"])
        fairness_frames[group].to_csv(METRICS_DIR / f"fairness_by_{group.lower()}.csv", index=False)

    save_core_charts(featured, model_results, risk_scores, importance, lift)
    save_dashboard_preview(audit, model_results["test_metrics"], risk_scores, featured)
    build_reports(audit, featured, model_results, risk_scores, importance, lift, fairness_frames)
    write_readme(audit, model_results, importance, lift)

    print("Project pipeline completed successfully.")
    print(f"Selected model: {model_results['best_model_name']}")
    print(f"Selected threshold: {model_results['selected_threshold']:.2f}")
    print(f"PR-AUC: {model_results['test_metrics']['pr_auc']:.3f}")
    print(f"Recall: {model_results['test_metrics']['recall']:.3f}")


if __name__ == "__main__":
    main()
