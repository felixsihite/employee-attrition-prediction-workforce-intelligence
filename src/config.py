from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
RISK_SCORES_DIR = OUTPUTS_DIR / "risk_scores"
DASHBOARD_SCREENSHOTS_DIR = OUTPUTS_DIR / "dashboard_screenshots"

RANDOM_STATE = 42
TARGET_COLUMN = "Attrition"
TARGET_FLAG = "attrition_flag"
IDENTIFIER_COLUMN = "EmployeeNumber"
CONSTANT_COLUMNS = ["EmployeeCount", "Over18", "StandardHours"]
LEAKAGE_COLUMNS = [TARGET_COLUMN, TARGET_FLAG, IDENTIFIER_COLUMN, *CONSTANT_COLUMNS]

DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"

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

APPROVED_MAIN_BACKGROUND = "#D6E4F0"