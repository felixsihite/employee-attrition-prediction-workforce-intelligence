"""Reusable analytics package for Employee Attrition Prediction.

The package is intentionally organized around production-style boundaries:
data quality validation, HR feature engineering, model training, evaluation,
interpretability, employee risk scoring, and visualization theming. Keeping
these capabilities importable makes the portfolio more credible than a single
notebook-only experiment and allows the Streamlit dashboard, tests, and
pipeline script to share the same audited logic.
"""

from __future__ import annotations

from src.config import (
    APPROVED_MAIN_BACKGROUND,
    DATASET_SOURCE_URL,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_FLAG,
)

__version__ = "1.0.0"
__project_name__ = "Employee Attrition Prediction & Workforce Intelligence"
__python_version__ = "3.13.1"

__all__ = [
    "__project_name__",
    "__python_version__",
    "__version__",
    "APPROVED_MAIN_BACKGROUND",
    "DATASET_SOURCE_URL",
    "RANDOM_STATE",
    "TARGET_COLUMN",
    "TARGET_FLAG",
]