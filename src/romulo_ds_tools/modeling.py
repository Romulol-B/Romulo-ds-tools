from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from romulo_ds_tools.features import select_feature_columns


@dataclass(frozen=True)
class PreparedData:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]


def build_estimator(
    problem_type: str, model_name: str, params: dict[str, Any] | None = None
) -> Any:
    params = dict(params or {})
    registry = {
        "classification": {
            "logistic_regression": LogisticRegression,
            "random_forest_classifier": RandomForestClassifier,
            "gradient_boosting_classifier": GradientBoostingClassifier,
        },
        "regression": {
            "linear_regression": LinearRegression,
            "ridge": Ridge,
            "random_forest_regressor": RandomForestRegressor,
            "gradient_boosting_regressor": GradientBoostingRegressor,
        },
    }
    try:
        estimator_cls = registry[problem_type][model_name]
    except KeyError as exc:
        valid = sorted(registry.get(problem_type, {}))
        raise ValueError(
            f"Unsupported {problem_type} model {model_name!r}. Valid: {valid}"
        ) from exc
    return estimator_cls(**params)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include="number")),
            ("cat", categorical_pipeline, make_column_selector(dtype_exclude="number")),
        ],
        remainder="drop",
    )


def build_model_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def prepare_data(
    df: pd.DataFrame,
    *,
    target_column: str,
    problem_type: str,
    test_size: float,
    random_state: int,
    feature_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    id_columns: list[str] | None = None,
    time_column: str | None = None,
    drop_missing_target: bool = True,
    drop_missing_features: bool = True,
) -> PreparedData:
    working = df.copy()
    if drop_missing_target:
        working = working.dropna(subset=[target_column])

    selected_features = select_feature_columns(
        working,
        target_column=target_column,
        feature_columns=feature_columns,
        exclude_columns=exclude_columns,
        id_columns=id_columns,
        time_column=time_column,
    )
    if drop_missing_features:
        working = working.dropna(subset=selected_features)

    x = working[selected_features]
    y = working[target_column]
    if time_column:
        ordered = working.sort_values(time_column)
        split_at = int(len(ordered) * (1 - test_size))
        split_at = max(1, min(split_at, len(ordered) - 1))
        train = ordered.iloc[:split_at]
        test = ordered.iloc[split_at:]
        return PreparedData(
            train[selected_features],
            test[selected_features],
            train[target_column],
            test[target_column],
            selected_features,
        )

    stratify = None
    if problem_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2:
        stratify = y
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return PreparedData(x_train, x_test, y_train, y_test, selected_features)
