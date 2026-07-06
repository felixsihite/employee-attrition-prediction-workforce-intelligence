from __future__ import annotations

import pandas as pd

from src.config import CONSTANT_COLUMNS, IDENTIFIER_COLUMN, TARGET_COLUMN, TARGET_FLAG
from src.data.quality import categorical_columns, load_raw_data
from src.features.build_features import add_hr_features, build_modeling_dataset


def test_feature_matrix_excludes_leakage_columns_and_has_no_missing_values() -> None:
    df = load_raw_data()
    X, y, featured = build_modeling_dataset(df)

    excluded = {TARGET_COLUMN, TARGET_FLAG, IDENTIFIER_COLUMN, *CONSTANT_COLUMNS}
    assert excluded.isdisjoint(set(X.columns))
    assert X.isna().sum().sum() == 0
    assert y.name == TARGET_FLAG
    assert bool(featured[TARGET_FLAG].isin([0, 1]).all())
    assert len(X) == len(y) == len(featured) == 1470


def test_hr_feature_set_contains_business_readable_segments() -> None:
    featured = add_hr_features(load_raw_data())
    expected_columns = {
        "age_band",
        "income_band",
        "distance_band",
        "tenure_band",
        "overtime_flag",
        "frequent_travel_flag",
        "early_tenure_flag",
        "career_stagnation_flag",
        "satisfaction_score",
        "engagement_score",
        "career_growth_score",
        "compensation_to_job_level_ratio",
        "years_with_manager_ratio",
        "promotion_gap_ratio",
        "high_risk_workload_flag",
        "attrition_risk_segment",
    }

    assert expected_columns.issubset(featured.columns)
    assert featured["satisfaction_score"].between(1, 4).all()
    assert featured["engagement_score"].between(1, 4).all()
    assert featured["career_growth_score"].between(0, 1).all()
    assert featured["attrition_risk_segment"].nunique() >= 4


def test_engineered_flags_are_binary_and_interpretable() -> None:
    featured = add_hr_features(load_raw_data())
    flag_columns = [column for column in featured.columns if column.endswith("_flag")]

    assert flag_columns
    for column in flag_columns:
        assert set(featured[column].dropna().unique()).issubset({0, 1}), column


def test_modeling_dataset_preserves_mixed_feature_types() -> None:
    X, _, _ = build_modeling_dataset(load_raw_data())
    numeric_columns = X.select_dtypes(include="number").columns
    categorical_feature_columns = categorical_columns(X)

    assert len(numeric_columns) > 20
    assert len(categorical_feature_columns) >= 10
    assert isinstance(X, pd.DataFrame)
