from __future__ import annotations

import pandas as pd

from src.config import CONSTANT_COLUMNS, DATASET_SOURCE_URL, IDENTIFIER_COLUMN
from src.data.quality import (
    categorical_cardinality,
    data_dictionary_frame,
    load_raw_data,
    numeric_range_summary,
    run_data_quality_audit,
    validate_raw_dataset,
)


def test_raw_dataset_health_contract() -> None:
    df = load_raw_data()
    validate_raw_dataset(df)
    audit = run_data_quality_audit(df)

    assert audit.row_count == 1470
    assert audit.column_count == 35
    assert audit.missing_values == 0
    assert audit.duplicate_rows == 0
    assert audit.target_distribution == {"No": 1233, "Yes": 237}
    assert round(audit.attrition_rate, 4) == 0.1612
    assert set(CONSTANT_COLUMNS).issubset(audit.constant_columns)
    assert IDENTIFIER_COLUMN in audit.identifier_columns
    assert DATASET_SOURCE_URL.startswith("https://www.kaggle.com/")


def test_data_dictionary_and_audit_tables_are_complete() -> None:
    df = load_raw_data()
    dictionary = data_dictionary_frame(df)
    cardinality = categorical_cardinality(df)
    numeric_ranges = numeric_range_summary(df)

    assert len(dictionary) == df.shape[1]
    assert dictionary["column"].is_unique
    assert dictionary["missing_values"].sum() == 0
    assert {"target", "identifier_excluded", "constant_excluded"}.issubset(set(dictionary["role"]))
    assert set(cardinality["column"]).issubset(df.columns)
    assert set(numeric_ranges["column"]).issubset(df.select_dtypes(include="number").columns)


def test_numeric_ranges_match_ibm_hr_dataset_expectations() -> None:
    df = load_raw_data()
    expected_ranges = {
        "Age": (18, 60),
        "DistanceFromHome": (1, 29),
        "MonthlyIncome": (1009, 19999),
        "StandardHours": (80, 80),
    }

    for column, (expected_min, expected_max) in expected_ranges.items():
        assert int(df[column].min()) == expected_min
        assert int(df[column].max()) == expected_max


def test_raw_data_is_loaded_as_dataframe_with_stable_columns() -> None:
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert df.columns[0] == "Age"
    assert df.columns[-1] == "YearsWithCurrManager"