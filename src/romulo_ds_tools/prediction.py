from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from omegaconf import DictConfig

from romulo_ds_tools.config import select
from romulo_ds_tools.features import select_feature_columns
from romulo_ds_tools.io import load_dataset, save_dataframe
from romulo_ds_tools.training import apply_project_transforms


def run_predict(
    cfg: DictConfig, input_path: str | Path | None = None, output_path: str | Path | None = None
) -> Path:
    df = load_dataset(input_path or select(cfg, "paths.data"))
    df = apply_project_transforms(df, cfg)
    model = joblib.load(select(cfg, "paths.model_path", "models/model.joblib"))
    target_column = select(cfg, "data.target")
    if target_column in df.columns:
        feature_columns = select_feature_columns(
            df,
            target_column=target_column,
            feature_columns=None,
            exclude_columns=list(select(cfg, "data.exclude_columns", [])),
            id_columns=list(select(cfg, "data.id_columns", [])),
            time_column=select(cfg, "data.time_column"),
        )
        x = df[feature_columns]
    else:
        x = df
    predictions = pd.DataFrame({"prediction": model.predict(x)})
    destination = Path(
        output_path or Path(select(cfg, "paths.reports_dir", "reports")) / "batch_predictions.csv"
    )
    return save_dataframe(predictions, destination)
