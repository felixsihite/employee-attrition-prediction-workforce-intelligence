from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.config import LEAKAGE_COLUMNS, TARGET_COLUMN, TARGET_FLAG


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.replace(0, np.nan).astype(float)


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _cut_band(series: pd.Series, bins: list[int], labels: list[str], *, include_lowest: bool = False) -> pd.Series:
    return cast(
        pd.Series,
        pd.cut(series, bins=bins, labels=labels, include_lowest=include_lowest),
    ).astype(str)


def add_hr_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    target = _series(featured, TARGET_COLUMN).astype(str)
    featured[TARGET_FLAG] = np.where(target.eq("Yes"), 1, 0).astype(int)

    featured["age_band"] = _cut_band(
        _series(featured, "Age"),
        bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-60"],
        include_lowest=True,
    )
    featured["income_band"] = _cut_band(
        _series(featured, "MonthlyIncome"),
        bins=[0, 3000, 6000, 10000, 15000, 25000],
        labels=["Entry income", "Developing income", "Mid income", "Senior income", "Executive income"],
        include_lowest=True,
    )
    featured["distance_band"] = _cut_band(
        _series(featured, "DistanceFromHome"),
        bins=[0, 5, 10, 20, 100],
        labels=["Near", "Moderate", "Far", "Very far"],
        include_lowest=True,
    )
    featured["tenure_band"] = _cut_band(
        _series(featured, "YearsAtCompany"),
        bins=[-1, 1, 3, 7, 15, 50],
        labels=["0-1 years", "2-3 years", "4-7 years", "8-15 years", "16+ years"],
    )
    featured["total_working_years_band"] = _cut_band(
        _series(featured, "TotalWorkingYears"),
        bins=[-1, 3, 7, 12, 20, 50],
        labels=["0-3 years", "4-7 years", "8-12 years", "13-20 years", "21+ years"],
    )
    featured["years_at_company_band"] = featured["tenure_band"]

    featured["overtime_flag"] = (featured["OverTime"] == "Yes").astype(int)
    featured["frequent_travel_flag"] = (featured["BusinessTravel"] == "Travel_Frequently").astype(int)
    featured["early_tenure_flag"] = (featured["YearsAtCompany"] <= 2).astype(int)
    featured["long_tenure_flag"] = (featured["YearsAtCompany"] >= 10).astype(int)
    featured["no_stock_option_flag"] = (featured["StockOptionLevel"] == 0).astype(int)
    featured["no_recent_promotion_flag"] = (featured["YearsSinceLastPromotion"] >= 4).astype(int)
    featured["career_stagnation_flag"] = (
        (featured["YearsSinceLastPromotion"] >= 4) & (featured["YearsInCurrentRole"] >= 4)
    ).astype(int)
    featured["low_job_satisfaction_flag"] = (featured["JobSatisfaction"] <= 2).astype(int)
    featured["low_environment_satisfaction_flag"] = (featured["EnvironmentSatisfaction"] <= 2).astype(int)
    featured["low_work_life_balance_flag"] = (featured["WorkLifeBalance"] <= 2).astype(int)
    featured["low_relationship_satisfaction_flag"] = (featured["RelationshipSatisfaction"] <= 2).astype(int)

    featured["satisfaction_score"] = featured[
        [
            "JobSatisfaction",
            "EnvironmentSatisfaction",
            "RelationshipSatisfaction",
            "WorkLifeBalance",
        ]
    ].mean(axis=1)
    featured["engagement_score"] = featured[
        [
            "JobInvolvement",
            "JobSatisfaction",
            "EnvironmentSatisfaction",
            "RelationshipSatisfaction",
        ]
    ].mean(axis=1)
    featured["career_growth_score"] = (
        featured["JobLevel"].rank(pct=True)
        + featured["PercentSalaryHike"].rank(pct=True)
        + (1 - featured["YearsSinceLastPromotion"].rank(pct=True))
    ) / 3
    featured["compensation_to_job_level_ratio"] = featured["MonthlyIncome"] / featured["JobLevel"].clip(lower=1)
    featured["years_with_manager_ratio"] = _safe_ratio(
        _series(featured, "YearsWithCurrManager"), _series(featured, "YearsAtCompany") + 1
    ).fillna(0)
    featured["promotion_gap_ratio"] = _safe_ratio(
        _series(featured, "YearsSinceLastPromotion"), _series(featured, "YearsAtCompany") + 1
    ).fillna(0)
    featured["high_risk_workload_flag"] = (
        (featured["overtime_flag"] == 1)
        & ((featured["WorkLifeBalance"] <= 2) | (featured["JobSatisfaction"] <= 2))
    ).astype(int)

    risk_conditions = [
        (featured["high_risk_workload_flag"] == 1) & (featured["early_tenure_flag"] == 1),
        (featured["career_stagnation_flag"] == 1) & (featured["no_recent_promotion_flag"] == 1),
        (featured["low_job_satisfaction_flag"] == 1) | (featured["low_environment_satisfaction_flag"] == 1),
        (featured["frequent_travel_flag"] == 1) | (featured["no_stock_option_flag"] == 1),
    ]
    risk_labels = [
        "Early-tenure workload risk",
        "Career progression risk",
        "Satisfaction risk",
        "Mobility and reward risk",
    ]
    featured["attrition_risk_segment"] = np.select(risk_conditions, risk_labels, default="Baseline workforce")
    return featured


def build_modeling_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    featured = add_hr_features(df)
    y = _series(featured, TARGET_FLAG)
    X = cast(pd.DataFrame, featured.drop(columns=[column for column in LEAKAGE_COLUMNS if column in featured.columns]))
    if X.isna().sum().sum() != 0:
        missing = X.isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(f"Feature matrix contains missing values: {missing}")
    return X, y, featured