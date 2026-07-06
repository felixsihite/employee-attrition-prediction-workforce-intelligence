from __future__ import annotations

from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.data.quality import categorical_columns
from src.evaluation.metrics import (
    classification_metrics,
    select_operating_threshold,
    threshold_optimization_table,
)


class ModelResults(TypedDict):
    final_pipeline: Pipeline
    best_model_name: str
    selected_threshold: float
    model_comparison: pd.DataFrame
    threshold_table: pd.DataFrame
    test_metrics: dict[str, float | int]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_probability: np.ndarray


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = categorical_columns(X)
    numeric_features = [column for column in X.columns.tolist() if column not in categorical_features]
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models() -> dict[str, BaseEstimator]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=220,
            learning_rate=0.035,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(model: BaseEstimator, X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X)),
            ("model", model),
        ]
    )


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> ModelResults:
    X_train, X_test, y_train, y_test = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        ),
    )
    X_fit, X_valid, y_fit, y_valid = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(
            X_train,
            y_train,
            test_size=0.25,
            stratify=y_train,
            random_state=RANDOM_STATE,
        ),
    )

    scoring: dict[str, Any] = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": make_scorer(recall_score, zero_division=0),
        "balanced_accuracy": "balanced_accuracy",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison_rows: list[dict[str, object]] = []
    fitted_validation_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in candidate_models().items():
        pipeline = build_pipeline(cast(BaseEstimator, clone(estimator)), X_fit)
        cv_pipeline = build_pipeline(cast(BaseEstimator, clone(estimator)), X_train)
        cv_scores = cast(
            dict[str, np.ndarray],
            cross_validate(
                cv_pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score=cast(float, "raise"),
            ),
        )
        pipeline.fit(X_fit, y_fit)
        valid_probability = pipeline.predict_proba(X_valid)[:, 1]
        valid_metrics = classification_metrics(y_valid, valid_probability, threshold=0.50)
        selection_score = (
            0.45 * valid_metrics["pr_auc"]
            + 0.35 * valid_metrics["recall"]
            + 0.20 * valid_metrics["roc_auc"]
        )
        comparison_rows.append(
            {
                "model": model_name,
                "cv_roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
                "cv_pr_auc_mean": float(cv_scores["test_pr_auc"].mean()),
                "cv_recall_mean": float(cv_scores["test_recall"].mean()),
                "cv_balanced_accuracy_mean": float(cv_scores["test_balanced_accuracy"].mean()),
                "validation_roc_auc": valid_metrics["roc_auc"],
                "validation_pr_auc": valid_metrics["pr_auc"],
                "validation_recall_at_0_50": valid_metrics["recall"],
                "validation_precision_at_0_50": valid_metrics["precision"],
                "selection_score": selection_score,
            }
        )
        fitted_validation_pipelines[model_name] = pipeline

    comparison_rows = sorted(
        comparison_rows,
        key=lambda row: float(cast(float | int, row["selection_score"])),
        reverse=True,
    )
    model_comparison = pd.DataFrame(comparison_rows)
    best_model_name = str(model_comparison.iloc[0]["model"])
    validation_pipeline = fitted_validation_pipelines[best_model_name]
    validation_probability = validation_pipeline.predict_proba(X_valid)[:, 1]
    threshold_table = threshold_optimization_table(y_valid, validation_probability)
    selected_threshold = select_operating_threshold(threshold_table)

    final_estimator = cast(BaseEstimator, clone(candidate_models()[best_model_name]))
    final_pipeline = build_pipeline(final_estimator, X_train)
    final_pipeline.fit(X_train, y_train)

    test_probability = final_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_probability, threshold=selected_threshold)

    baseline_pr_auc = float(y_train.mean())
    model_comparison["baseline_pr_auc"] = baseline_pr_auc
    model_comparison["selected_model"] = model_comparison["model"].eq(best_model_name)

    return {
        "final_pipeline": final_pipeline,
        "best_model_name": best_model_name,
        "selected_threshold": selected_threshold,
        "model_comparison": model_comparison,
        "threshold_table": threshold_table,
        "test_metrics": test_metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_probability": test_probability,
    }
