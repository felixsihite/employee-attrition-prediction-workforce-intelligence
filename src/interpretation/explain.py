from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import RANDOM_STATE


def permutation_importance_frame(pipeline: Any, X: pd.DataFrame, y: pd.Series, n_repeats: int = 8) -> pd.DataFrame:
    result = cast(
        dict[str, np.ndarray],
        permutation_importance(
            pipeline,
            X,
            y,
            scoring="average_precision",
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )
    frame = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result["importances_mean"],
            "importance_std": result["importances_std"],
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _numeric_value(row: pd.Series, column: str, default: float) -> float:
    value = row.get(column, default)
    if value is None:
        return default
    return float(value)


def hr_risk_explanation(row: pd.Series) -> list[str]:
    drivers: list[str] = []
    if row.get("OverTime") == "Yes":
        drivers.append("Works overtime")
    if row.get("BusinessTravel") == "Travel_Frequently":
        drivers.append("Frequent business travel")
    if _numeric_value(row, "JobSatisfaction", 4) <= 2:
        drivers.append("Low job satisfaction")
    if _numeric_value(row, "EnvironmentSatisfaction", 4) <= 2:
        drivers.append("Low environment satisfaction")
    if _numeric_value(row, "WorkLifeBalance", 4) <= 2:
        drivers.append("Low work-life balance")
    if _numeric_value(row, "YearsAtCompany", 99) <= 2:
        drivers.append("Early tenure")
    if _numeric_value(row, "YearsSinceLastPromotion", 0) >= 4:
        drivers.append("Long promotion gap")
    if row.get("StockOptionLevel", 1) == 0:
        drivers.append("No stock option")
    return drivers[:5] if drivers else ["No dominant heuristic risk driver"]