from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import APPROVED_MAIN_BACKGROUND, DATASET_SOURCE_URL


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_outputs_exist_after_run() -> None:
    required = [
        PROJECT_ROOT / "data" / "processed" / "employee_attrition_featured_dataset.csv",
        PROJECT_ROOT / "data" / "processed" / "employee_attrition_modeling_dataset.csv",
        PROJECT_ROOT / "models" / "employee_attrition_model.joblib",
        PROJECT_ROOT / "outputs" / "risk_scores" / "employee_attrition_risk_scores.csv",
        PROJECT_ROOT / "outputs" / "metrics" / "test_metrics.json",
        PROJECT_ROOT / "outputs" / "reports" / "model_card.md",
        PROJECT_ROOT / "outputs" / "reports" / "fairness_ethics_report.md",
    ]
    for path in required:
        assert path.exists(), f"Missing expected pipeline output: {path}"
        assert path.stat().st_size > 0, f"Output exists but is empty: {path}"


def test_risk_scores_are_complete_ranked_and_explainable() -> None:
    path = PROJECT_ROOT / "outputs" / "risk_scores" / "employee_attrition_risk_scores.csv"
    scores = pd.read_csv(path)

    assert len(scores) == 1470
    assert scores["attrition_probability"].between(0, 1).all()
    assert scores["attrition_probability"].is_monotonic_decreasing
    assert scores["risk_band"].notna().all()
    assert scores["risk_decile"].between(1, 10).all()
    assert scores["risk_explanation"].str.len().gt(0).all()


def test_metrics_json_contains_hr_relevant_model_evaluation() -> None:
    metrics = json.loads((PROJECT_ROOT / "outputs" / "metrics" / "test_metrics.json").read_text(encoding="utf-8"))
    expected_keys = {
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "fbeta_2",
        "balanced_accuracy",
        "intervention_rate",
        "selected_employees",
    }

    assert expected_keys.issubset(metrics)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1
    assert metrics["recall"] >= 0.5
    assert 0 < metrics["intervention_rate"] <= 0.35


def test_readme_documents_dataset_source_theme_and_limitations() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert DATASET_SOURCE_URL in readme
    assert APPROVED_MAIN_BACKGROUND in readme
    assert "does not claim guaranteed retention" in readme
    assert "Python version target: **3.13.1**" in readme


def test_dashboard_assets_cover_core_pages() -> None:
    screenshot_dir = PROJECT_ROOT / "outputs" / "dashboard_screenshots"
    expected = {
        "00_app.png",
        "01_executive_workforce_overview.png",
        "02_attrition_intelligence.png",
        "03_workforce_segmentation.png",
        "04_model_performance.png",
        "05_employee_risk_scoring.png",
        "06_explainable_ai_hr_recommendations.png",
    }

    existing = {path.name for path in screenshot_dir.glob("*.png")}
    missing = expected - existing
    assert not missing, f"Missing dashboard screenshots: {sorted(missing)}"