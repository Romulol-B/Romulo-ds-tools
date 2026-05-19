from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from romulo_ds_tools.config import select
from romulo_ds_tools.io import write_json


def _list_config(cfg: DictConfig, key: str, default: list[str] | None = None) -> list[str] | None:
    value = select(cfg, key, default)
    if value is None:
        return None
    return list(value)


def select_visualization_columns(df: pd.DataFrame, cfg: DictConfig) -> list[str]:
    configured_columns = _list_config(cfg, "visualization.columns")
    include_target = bool(select(cfg, "visualization.include_target", False))
    excluded = set(_list_config(cfg, "visualization.exclude_columns", []) or [])
    excluded.update(_list_config(cfg, "data.exclude_columns", []) or [])
    excluded.update(_list_config(cfg, "data.id_columns", []) or [])

    target = select(cfg, "data.target")
    time_column = select(cfg, "data.time_column")
    if target and not include_target:
        excluded.add(target)
    if time_column:
        excluded.add(time_column)

    candidates = configured_columns or list(df.columns)
    missing = [column for column in candidates if column not in df.columns]
    if missing:
        raise ValueError(f"Visualization columns are missing from dataframe: {missing}")

    selected = [column for column in candidates if column not in excluded]
    numeric_selected = [column for column in selected if pd.api.types.is_numeric_dtype(df[column])]
    if not numeric_selected:
        raise ValueError("No numeric columns available for standard visualizations")
    return numeric_selected


def write_boxplots(df: pd.DataFrame, columns: list[str], path: str | Path) -> Path:
    import plotly.express as px

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    melted = df[columns].melt(var_name="feature", value_name="value")
    figure = px.box(
        melted,
        x="feature",
        y="value",
        points=False,
        title="Feature box plots",
    )
    figure.write_html(output_path)
    return output_path


def write_correlation_heatmap(df: pd.DataFrame, columns: list[str], path: str | Path) -> Path:
    import plotly.express as px

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correlation = df[columns].corr(numeric_only=True)
    figure = px.imshow(
        correlation,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
        title="Feature correlation heatmap",
        aspect="auto",
    )
    figure.write_html(output_path)
    return output_path


def write_pairplot(
    df: pd.DataFrame, columns: list[str], path: str | Path, color_column: str | None = None
) -> Path:
    import plotly.express as px

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    color = color_column if color_column in df.columns else None
    figure = px.scatter_matrix(
        df,
        dimensions=columns,
        color=color,
        title="Feature pairplot",
    )
    figure.update_traces(diagonal_visible=False)
    figure.write_html(output_path)
    return output_path


def write_standard_visualizations(df: pd.DataFrame, cfg: DictConfig) -> dict[str, Any]:
    if not select(cfg, "visualization.enabled", True):
        return {"enabled": False, "columns": [], "artifacts": {}}

    output_dir = Path(select(cfg, "visualization.output_dir", "reports/eda"))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        selected_columns = select_visualization_columns(df, cfg)
    except ValueError as exc:
        if "No numeric columns" not in str(exc):
            raise
        manifest = {
            "enabled": True,
            "columns": [],
            "artifacts": {},
            "skipped_reason": str(exc),
        }
        write_json(manifest, output_dir / "manifest.json")
        return manifest

    artifacts: dict[str, str] = {}

    if select(cfg, "visualization.boxplots.enabled", True):
        artifacts["boxplots"] = str(
            write_boxplots(df, selected_columns, output_dir / "boxplots.html")
        )

    if select(cfg, "visualization.heatmap.enabled", True):
        artifacts["heatmap"] = str(
            write_correlation_heatmap(df, selected_columns, output_dir / "correlation_heatmap.html")
        )

    max_pairplot_features = int(select(cfg, "visualization.pairplot.max_features", 6))
    pairplot_columns = selected_columns[:max_pairplot_features]
    if select(cfg, "visualization.pairplot.enabled", True) and len(pairplot_columns) >= 2:
        artifacts["pairplot"] = str(
            write_pairplot(
                df,
                pairplot_columns,
                output_dir / "pairplot.html",
                color_column=select(cfg, "visualization.pairplot.color_column", None),
            )
        )

    manifest = {
        "enabled": True,
        "columns": selected_columns,
        "artifacts": artifacts,
    }
    write_json(manifest, output_dir / "manifest.json")
    return manifest
