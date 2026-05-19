from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from romulo_ds_tools.visualization import (
    select_visualization_columns,
    write_standard_visualizations,
)


def test_select_visualization_columns_applies_include_and_exclude_rules():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "target": [0, 1],
            "x1": [1.0, 2.0],
            "x2": [3.0, 4.0],
            "category": ["a", "b"],
        }
    )
    cfg = OmegaConf.create(
        {
            "data": {
                "target": "target",
                "time_column": None,
                "exclude_columns": [],
                "id_columns": ["id"],
            },
            "visualization": {
                "columns": ["id", "target", "x1", "x2", "category"],
                "exclude_columns": ["x2"],
                "include_target": False,
            },
        }
    )

    assert select_visualization_columns(df, cfg) == ["x1"]


def test_write_standard_visualizations_creates_enabled_artifacts(tmp_path):
    df = pd.DataFrame(
        {
            "target": ["a", "b", "a", "b"],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 3.0, 4.0, 5.0],
            "x3": [5.0, 4.0, 3.0, 2.0],
        }
    )
    cfg = OmegaConf.create(
        {
            "data": {
                "target": "target",
                "time_column": None,
                "exclude_columns": [],
                "id_columns": [],
            },
            "visualization": {
                "enabled": True,
                "output_dir": str(tmp_path / "eda"),
                "columns": None,
                "exclude_columns": ["x3"],
                "include_target": False,
                "boxplots": {"enabled": True},
                "heatmap": {"enabled": True},
                "pairplot": {
                    "enabled": True,
                    "max_features": 2,
                    "color_column": "target",
                },
            },
        }
    )

    manifest = write_standard_visualizations(df, cfg)

    assert manifest["columns"] == ["x1", "x2"]
    assert set(manifest["artifacts"]) == {"boxplots", "heatmap", "pairplot"}
    assert Path(manifest["artifacts"]["boxplots"]).exists()
    assert Path(manifest["artifacts"]["heatmap"]).exists()
    assert Path(manifest["artifacts"]["pairplot"]).exists()
    assert (tmp_path / "eda" / "manifest.json").exists()


def test_write_standard_visualizations_skips_when_dataset_has_no_numeric_columns(tmp_path):
    df = pd.DataFrame({"target": ["a", "b"], "feature": ["x", "y"]})
    cfg = OmegaConf.create(
        {
            "data": {
                "target": "target",
                "time_column": None,
                "exclude_columns": [],
                "id_columns": [],
            },
            "visualization": {
                "enabled": True,
                "output_dir": str(tmp_path / "eda"),
                "columns": None,
                "exclude_columns": [],
                "include_target": False,
                "boxplots": {"enabled": True},
                "heatmap": {"enabled": True},
                "pairplot": {"enabled": True, "max_features": 2, "color_column": "target"},
            },
        }
    )

    manifest = write_standard_visualizations(df, cfg)

    assert manifest["artifacts"] == {}
    assert "No numeric columns" in manifest["skipped_reason"]
    assert (tmp_path / "eda" / "manifest.json").exists()
