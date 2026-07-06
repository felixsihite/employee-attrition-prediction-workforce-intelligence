from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def classification_metrics(y_true, y_probability, threshold: float = 0.5) -> dict[str, float | int]:
    y_pred = (np.asarray(y_probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    fbeta_2 = float((5 * precision * recall) / ((4 * precision) + recall)) if ((4 * precision) + recall) else 0.0
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fbeta_2": fbeta_2,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "selected_employees": int(y_pred.sum()),
        "intervention_rate": float(y_pred.mean()),
    }


def threshold_optimization_table(y_true, y_probability) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.05), 2):
        rows.append(classification_metrics(y_true, y_probability, float(threshold)))
    return pd.DataFrame(rows)


def select_operating_threshold(
    threshold_table: pd.DataFrame,
    max_intervention_rate: float = 0.30,
    min_intervention_rate: float = 0.05,
) -> float:
    eligible = threshold_table[
        (threshold_table["intervention_rate"] <= max_intervention_rate)
        & (threshold_table["intervention_rate"] >= min_intervention_rate)
    ].copy()
    if eligible.empty:
        eligible = threshold_table.copy()
    score_columns = ["fbeta_2", "recall", "precision", "balanced_accuracy"]
    selection_scores = eligible.loc[:, score_columns].to_numpy(dtype=float)
    best_position = max(range(len(selection_scores)), key=lambda position: tuple(selection_scores[position]))
    thresholds = np.asarray(eligible["threshold"], dtype=float)
    return float(thresholds[best_position])


def lift_gain_table(y_true, y_probability, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": np.asarray(y_true), "probability": np.asarray(y_probability)})
    sort_order = np.argsort(-np.asarray(frame["probability"], dtype=float))
    frame = frame.iloc[sort_order].reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["risk_decile"] = np.ceil(frame["rank"] / (len(frame) / bins)).clip(1, bins).astype(int)
    base_rate = float(frame["actual"].mean())
    total_events = float(frame["actual"].sum())
    grouped = (
        frame.groupby("risk_decile", as_index=False, sort=True)
        .agg(
            employees=("actual", "size"),
            attrition_count=("actual", "sum"),
            min_probability=("probability", "min"),
            max_probability=("probability", "max"),
        )
    )
    grouped["attrition_rate"] = grouped["attrition_count"] / grouped["employees"]
    grouped["lift"] = grouped["attrition_rate"] / base_rate if base_rate else 0
    grouped["cumulative_attrition"] = grouped["attrition_count"].cumsum()
    grouped["cumulative_employees"] = grouped["employees"].cumsum()
    grouped["cumulative_attrition_capture"] = grouped["cumulative_attrition"] / total_events if total_events else 0
    grouped["cumulative_population_share"] = grouped["cumulative_employees"] / len(frame)
    grouped["gain"] = grouped["cumulative_attrition_capture"] - grouped["cumulative_population_share"]
    return grouped


def group_risk_summary(df: pd.DataFrame, group_column: str, threshold: float) -> pd.DataFrame:
    summary = (
        df.groupby(group_column, dropna=False)
        .agg(
            employees=("attrition_probability", "size"),
            actual_attrition_rate=("attrition_flag", "mean"),
            mean_attrition_probability=("attrition_probability", "mean"),
            high_risk_share=("attrition_probability", lambda s: (s >= threshold).mean()),
        )
        .reset_index()
        .sort_values("mean_attrition_probability", ascending=False)
    )
    return summary