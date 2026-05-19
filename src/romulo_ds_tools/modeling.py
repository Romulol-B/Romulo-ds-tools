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

from romulo_ds_tools.custom import call_optional_factory
from romulo_ds_tools.features import select_feature_columns


@dataclass(frozen=True)
class PreparedData:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]


@dataclass(frozen=True)
class DtypeExcludingColumnSelector:
    dtype_include: Any | None = None
    dtype_exclude: Any | None = None
    exclude_columns: frozenset[str] = frozenset()

    def __call__(self, df: pd.DataFrame) -> list[str]:
        selected_columns = make_column_selector(
            dtype_include=self.dtype_include,
            dtype_exclude=self.dtype_exclude,
        )(df)
        return [column for column in selected_columns if column not in self.exclude_columns]


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


def _normalize_custom_columns(columns: Any, transformer_name: str) -> list[str]:
    if isinstance(columns, str):
        return [columns]
    if isinstance(columns, (list, tuple, set)) and all(
        isinstance(column, str) for column in columns
    ):
        return list(columns)
    raise TypeError(
        "project.preprocessing.get_column_transformers must use explicit string "
        f"column names for transformer {transformer_name!r}"
    )


def _load_custom_column_transformers(
    cfg: Any | None,
) -> tuple[list[tuple[str, Any, list[str]]], set[str]]:
    raw_transformers = call_optional_factory(
        "project.preprocessing",
        "get_column_transformers",
        cfg,
    )
    if raw_transformers is None:
        return [], set()
    if not isinstance(raw_transformers, (list, tuple)):
        raise TypeError(
            "project.preprocessing.get_column_transformers must return a list of "
            "(name, transformer, columns) tuples"
        )

    transformers: list[tuple[str, Any, list[str]]] = []
    transformed_columns: set[str] = set()
    transformer_names: set[str] = {"num", "cat"}
    for index, raw_transformer in enumerate(raw_transformers):
        if not isinstance(raw_transformer, (list, tuple)) or len(raw_transformer) != 3:
            raise TypeError(
                "project.preprocessing.get_column_transformers must return a list of "
                "(name, transformer, columns) tuples"
            )
        name, transformer, columns = raw_transformer
        if not isinstance(name, str) or not name:
            raise TypeError(
                "project.preprocessing.get_column_transformers transformer names "
                f"must be non-empty strings; item {index} is invalid"
            )
        if name in transformer_names:
            raise ValueError(
                "project.preprocessing.get_column_transformers transformer names "
                f"must be unique; {name!r} was repeated"
            )
        normalized_columns = _normalize_custom_columns(columns, name)
        transformer_names.add(name)
        transformed_columns.update(normalized_columns)
        transformers.append((name, transformer, normalized_columns))
    return transformers, transformed_columns


def build_preprocessor(cfg: Any | None = None) -> ColumnTransformer:
    custom_transformers, custom_columns = _load_custom_column_transformers(cfg)
    excluded_columns = frozenset(custom_columns)
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
            *custom_transformers,
            (
                "num",
                numeric_pipeline,
                DtypeExcludingColumnSelector(
                    dtype_include="number",
                    exclude_columns=excluded_columns,
                ),
            ),
            (
                "cat",
                categorical_pipeline,
                DtypeExcludingColumnSelector(
                    dtype_exclude="number",
                    exclude_columns=excluded_columns,
                ),
            ),
        ],
        remainder="drop",
    )


def build_model_pipeline(estimator: Any, cfg: Any | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(cfg)),
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
