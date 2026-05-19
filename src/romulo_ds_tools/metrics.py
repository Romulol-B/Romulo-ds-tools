from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def compute_metrics(problem_type: str, y_true: Any, y_pred: Any) -> dict[str, float]:
    if problem_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    if problem_type == "regression":
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    raise ValueError("problem_type must be 'classification' or 'regression'")
