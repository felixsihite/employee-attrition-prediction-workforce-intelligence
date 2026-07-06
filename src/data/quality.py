from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.config import CONSTANT_COLUMNS, IDENTIFIER_COLUMN, RAW_DATA_PATH, TARGET_COLUMN


EXPECTED_ROWS = 1470
EXPECTED_COLUMNS = 35


@dataclass(frozen=True)
class DataQualityResult:
    row_count: int
    column_count: int
    missing_values: int
    duplicate_rows: int
    target_distribution: dict[str, int]
    attrition_rate: float
    constant_columns: list[str]
    identifier_columns: list[str]


RAW_COLUMN_DESCRIPTIONS = {
    "Age": "Employee age in years.",
    "Attrition": "Target label indicating whether the employee left the company.",
    "BusinessTravel": "Business travel frequency.",
    "DailyRate": "Daily compensation rate recorded in the source data.",
    "Department": "Employee department.",
    "DistanceFromHome": "Distance from home to workplace.",
    "Education": "Education level coded from 1 to 5.",
    "EducationField": "Field of education.",
    "EmployeeCount": "Constant source column; excluded from modeling.",
    "EmployeeNumber": "Employee identifier; retained for reporting but excluded from modeling.",
    "EnvironmentSatisfaction": "Work environment satisfaction score from 1 to 4.",
    "Gender": "Employee gender category in the source data.",
    "HourlyRate": "Hourly rate recorded in the source data.",
    "JobInvolvement": "Job involvement score from 1 to 4.",
    "JobLevel": "Internal job level.",
    "JobRole": "Employee job role.",
    "JobSatisfaction": "Job satisfaction score from 1 to 4.",
    "MaritalStatus": "Marital status category.",
    "MonthlyIncome": "Monthly income.",
    "MonthlyRate": "Monthly rate recorded in the source data.",
    "NumCompaniesWorked": "Number of companies worked before or including the current employer.",
    "Over18": "Constant source column; excluded from modeling.",
    "OverTime": "Whether the employee works overtime.",
    "PercentSalaryHike": "Percent salary hike.",
    "PerformanceRating": "Performance rating.",
    "RelationshipSatisfaction": "Relationship satisfaction score from 1 to 4.",
    "StandardHours": "Constant source column; excluded from modeling.",
    "StockOptionLevel": "Stock option level.",
    "TotalWorkingYears": "Total years of work experience.",
    "TrainingTimesLastYear": "Training count in the last year.",
    "WorkLifeBalance": "Work-life balance score from 1 to 4.",
    "YearsAtCompany": "Tenure at the company in years.",
    "YearsInCurrentRole": "Years in the current role.",
    "YearsSinceLastPromotion": "Years since last promotion.",
    "YearsWithCurrManager": "Years working with the current manager.",
}


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    data_path = RAW_DATA_PATH if path is None else path
    return pd.read_csv(data_path)


def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if df[column].nunique(dropna=False) <= 1]


def detect_identifier_columns(df: pd.DataFrame) -> list[str]:
    identifiers = []
    for column in df.columns:
        if df[column].is_unique and column != TARGET_COLUMN:
            identifiers.append(column)
    if IDENTIFIER_COLUMN not in identifiers and IDENTIFIER_COLUMN in df.columns:
        identifiers.append(IDENTIFIER_COLUMN)
    return identifiers


def target_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = df[TARGET_COLUMN].value_counts(dropna=False)
    return {str(label): int(count) for label, count in counts.items()}


def run_data_quality_audit(df: pd.DataFrame) -> DataQualityResult:
    distribution = target_distribution(df)
    total = len(df)
    attrition_yes = int(distribution.get("Yes", 0))
    return DataQualityResult(
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        missing_values=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        target_distribution={str(k): int(v) for k, v in distribution.items()},
        attrition_rate=attrition_yes / total if total else 0.0,
        constant_columns=detect_constant_columns(df),
        identifier_columns=detect_identifier_columns(df),
    )


def validate_raw_dataset(df: pd.DataFrame) -> None:
    audit = run_data_quality_audit(df)
    errors: list[str] = []
    if audit.row_count != EXPECTED_ROWS:
        errors.append(f"Expected {EXPECTED_ROWS} rows, found {audit.row_count}.")
    if audit.column_count != EXPECTED_COLUMNS:
        errors.append(f"Expected {EXPECTED_COLUMNS} columns, found {audit.column_count}.")
    if audit.missing_values != 0:
        errors.append(f"Expected no missing values, found {audit.missing_values}.")
    if audit.duplicate_rows != 0:
        errors.append(f"Expected no duplicate rows, found {audit.duplicate_rows}.")
    for column in CONSTANT_COLUMNS:
        if column not in audit.constant_columns:
            errors.append(f"Expected constant column missing from audit: {column}.")
    if errors:
        raise ValueError("Raw dataset validation failed: " + " ".join(errors))


def is_categorical_series(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if is_categorical_series(cast(pd.Series, df[column]))]


def categorical_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in categorical_columns(df):
        series = cast(pd.Series, df[column])
        rows.append(
            {
                "column": column,
                "unique_values": int(series.nunique(dropna=False)),
                "top_value": series.mode(dropna=False).iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values("unique_values", ascending=False)


def numeric_range_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    summary = numeric.agg(["min", "median", "mean", "max"]).T.reset_index()
    summary = summary.rename(columns={"index": "column"})
    return summary


def stable_dtype_name(series: pd.Series) -> str:
    if is_categorical_series(series) and not isinstance(series.dtype, pd.CategoricalDtype):
        return "string"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    return str(series.dtype)


def data_dictionary_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    constants = set(detect_constant_columns(df))
    identifiers = set(detect_identifier_columns(df))
    for column in df.columns:
        series = cast(pd.Series, df[column])
        if column == TARGET_COLUMN:
            role = "target"
        elif column in constants:
            role = "constant_excluded"
        elif column in identifiers:
            role = "identifier_excluded"
        elif pd.api.types.is_numeric_dtype(series):
            role = "numeric_feature"
        else:
            role = "categorical_feature"
        rows.append(
            {
                "column": column,
                "dtype": stable_dtype_name(series),
                "role": role,
                "missing_values": int(series.isna().sum()),
                "unique_values": int(series.nunique(dropna=False)),
                "description": RAW_COLUMN_DESCRIPTIONS.get(column, "Source dataset field."),
            }
        )
    return pd.DataFrame(rows)
