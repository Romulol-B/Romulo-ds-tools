from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from romulo_ds_tools.config import select
from romulo_ds_tools.io import load_dataset, write_json


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isna().sum().sort_values(ascending=False)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing": {column: int(value) for column, value in missing.items()},
        "numeric_summary": df.describe(include="number").to_dict(),
    }


def write_missingness_plot(df: pd.DataFrame, path: str | Path) -> Path:
    import plotly.express as px

    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    figure = px.bar(missing, x="column", y="missing_count", title="Missing values by column")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path)
    return output_path


def run_profile(cfg: DictConfig) -> dict[str, Any]:
    df = load_dataset(select(cfg, "paths.data"))
    reports_dir = Path(select(cfg, "paths.reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_dataframe(df)
    write_json(summary, reports_dir / "profile.json")
    write_missingness_plot(df, reports_dir / "missingness.html")
    return summary
