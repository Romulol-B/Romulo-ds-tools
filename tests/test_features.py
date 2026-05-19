from __future__ import annotations

import numpy as np
import pandas as pd

from romulo_ds_tools.features import add_time_series_features, select_feature_columns


def test_time_series_features_use_only_past_target_values():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "target": [10.0, 20.0, 999.0, 40.0],
        }
    )

    result = add_time_series_features(
        df,
        target_column="target",
        time_column="date",
        lags=[1, 2],
        rolling_windows=[2],
    )

    assert np.isnan(result.loc[0, "target_lag_1"])
    assert result.loc[1, "target_lag_1"] == 10.0
    assert result.loc[2, "target_lag_1"] == 20.0
    assert result.loc[2, "target_lag_2"] == 10.0
    assert result.loc[2, "target_roll_mean_2"] == 15.0
    assert result.loc[2, "target_roll_mean_2"] != 999.0


def test_time_series_features_respect_groups():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "store": ["a", "a", "b", "b"],
            "target": [1, 2, 100, 200],
        }
    )

    result = add_time_series_features(
        df,
        target_column="target",
        time_column="date",
        lags=[1],
        group_columns=["store"],
    )

    second_store_b = result[
        (result["store"] == "b") & (result["date"] == pd.Timestamp("2024-01-02"))
    ]
    assert second_store_b["target_lag_1"].iloc[0] == 100


def test_select_feature_columns_excludes_target_ids_and_time():
    df = pd.DataFrame({"id": [1], "date": ["2024-01-01"], "x": [2], "target": [3]})

    selected = select_feature_columns(
        df,
        target_column="target",
        id_columns=["id"],
        time_column="date",
    )

    assert selected == ["x"]
