from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from romulo_ds_tools.io import require_columns


def _normalize_ints(values: Iterable[int] | None) -> list[int]:
    if values is None:
        return []
    normalized = sorted({int(value) for value in values})
    if any(value <= 0 for value in normalized):
        raise ValueError("Lag and rolling window values must be positive integers")
    return normalized


def add_time_series_features(
    df: pd.DataFrame,
    *,
    target_column: str,
    time_column: str,
    lags: Iterable[int] | None = None,
    rolling_windows: Iterable[int] | None = None,
    group_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Create lag and rolling features using only past target values.

    The rolling features are based on ``target.shift(1)`` so the current row's target is
    never used to predict itself.
    """
    group_cols = list(group_columns or [])
    require_columns(df, [target_column, time_column, *group_cols], "time-series feature input")
    lag_values = _normalize_ints(lags)
    window_values = _normalize_ints(rolling_windows)

    working = df.copy()
    working[time_column] = pd.to_datetime(working[time_column])

    def add_for_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(time_column).copy()
        shifted_target = group[target_column].shift(1)
        for lag in lag_values:
            group[f"{target_column}_lag_{lag}"] = group[target_column].shift(lag)
        for window in window_values:
            rolling = shifted_target.rolling(window=window, min_periods=1)
            group[f"{target_column}_roll_mean_{window}"] = rolling.mean()
            group[f"{target_column}_roll_std_{window}"] = rolling.std(ddof=0)
        return group

    if not group_cols:
        return add_for_group(working).reset_index(drop=True)

    pieces = [
        add_for_group(group) for _, group in working.groupby(group_cols, sort=False, dropna=False)
    ]
    return pd.concat(pieces, ignore_index=True)


def select_feature_columns(
    df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    id_columns: list[str] | None = None,
    time_column: str | None = None,
) -> list[str]:
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found")

    excluded = {target_column, *(exclude_columns or []), *(id_columns or [])}
    if time_column:
        excluded.add(time_column)

    if feature_columns:
        missing = [column for column in feature_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Configured feature columns are missing: {missing}")
        selected = [column for column in feature_columns if column not in excluded]
    else:
        selected = [column for column in df.columns if column not in excluded]

    if not selected:
        raise ValueError("No feature columns selected")
    return selected
