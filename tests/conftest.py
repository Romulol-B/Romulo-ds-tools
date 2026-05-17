from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf


@pytest.fixture()
def classification_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_num": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85, 0.25, 0.75],
            "feature_cat": ["a", "a", "b", "b", "a", "b", "a", "b"],
            "target": [0, 0, 1, 1, 0, 1, 0, 1],
            "row_id": list(range(8)),
        }
    )


@pytest.fixture()
def regression_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x2": ["low", "low", "mid", "mid", "high", "high"],
            "target": [1.5, 2.1, 2.9, 4.2, 4.8, 6.1],
        }
    )


@pytest.fixture()
def time_series_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "target": [10.0, 12.0, 13.0, 15.0, 18.0, 21.0, 22.0, 25.0],
            "store": ["a"] * 8,
        }
    )


def make_config(
    tmp_path: Path,
    data_path: Path,
    *,
    problem_type: str = "classification",
    model_name: str = "random_forest_classifier",
    time_column: str | None = None,
    time_series_enabled: bool = False,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "project_name": "test-project",
            "paths": {
                "data": str(data_path),
                "artifacts_dir": str(tmp_path / "artifacts"),
                "reports_dir": str(tmp_path / "reports"),
                "model_dir": str(tmp_path / "models"),
                "model_path": str(tmp_path / "models" / "model.joblib"),
            },
            "validation": {
                "enabled": True,
                "schema_module": "project.schema",
                "output_path": str(tmp_path / "reports" / "validation.json"),
            },
            "data": {
                "target": "target",
                "problem_type": problem_type,
                "time_column": time_column,
                "feature_columns": None,
                "exclude_columns": [],
                "id_columns": ["row_id"],
                "test_size": 0.25,
            },
            "features": {
                "time_series": {
                    "enabled": time_series_enabled,
                    "group_columns": [],
                    "lags": [1, 2],
                    "rolling_windows": [2],
                }
            },
            "model": {
                "name": model_name,
                "params": {"n_estimators": 5, "random_state": 42} if "forest" in model_name else {},
            },
            "training": {
                "random_state": 42,
                "drop_rows_with_missing_target": True,
                "drop_rows_with_missing_features": True,
            },
            "tracking": {"enabled": False},
            "dashboard": {"host": "127.0.0.1", "port": 8050},
        }
    )


@pytest.fixture()
def config_factory(tmp_path):
    return lambda data_path, **kwargs: make_config(tmp_path, data_path, **kwargs)
