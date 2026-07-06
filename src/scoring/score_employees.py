from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from src.config import IDENTIFIER_COLUMN, TARGET_FLAG
from src.interpretation.explain import hr_risk_explanation


def assign_risk_band(probabilities: Any) -> pd.Series:
    return cast(
        pd.Series,
        pd.cut(
            pd.Series(probabilities),
            bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
            labels=["Low", "Elevated", "High", "Critical"],
        ),
    ).astype(str)


def assign_priority(risk_decile: int) -> str:
    if risk_decile <= 1:
        return "Priority 1 - executive review"
    if risk_decile <= 2:
        return "Priority 2 - targeted HR review"
    if risk_decile <= 3:
        return "Priority 3 - manager check-in"
    return "Monitor"


def score_employee_population(pipeline, X: pd.DataFrame, raw_df: pd.DataFrame, featured_df: pd.DataFrame) -> pd.DataFrame:
    probabilities = pipeline.predict_proba(X)[:, 1]
    score_columns = [
        IDENTIFIER_COLUMN,
        "Department",
        "JobRole",
        "OverTime",
        "BusinessTravel",
        "MonthlyIncome",
        "YearsAtCompany",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "WorkLifeBalance",
        "Attrition",
    ]
    scores = cast(pd.DataFrame, raw_df.loc[:, score_columns].copy())
    scores[TARGET_FLAG] = featured_df[TARGET_FLAG].values
    scores["attrition_probability"] = probabilities
    ranked = pd.Series(probabilities).rank(method="first", ascending=True)
    decile_codes = cast(pd.Series, pd.qcut(ranked, q=10, labels=False, duplicates="drop")).astype(int)
    scores["risk_decile"] = (10 - decile_codes).astype(int)
    scores["risk_band"] = assign_risk_band(scores["attrition_probability"])
    scores["intervention_priority"] = cast(pd.Series, scores["risk_decile"]).apply(lambda value: assign_priority(int(value)))
    scores["risk_explanation"] = ["; ".join(hr_risk_explanation(row)) for _, row in raw_df.iterrows()]
    sort_order = np.argsort(-np.asarray(scores["attrition_probability"], dtype=float))
    return cast(pd.DataFrame, scores.iloc[sort_order].reset_index(drop=True))