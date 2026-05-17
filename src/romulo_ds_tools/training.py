from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from romulo_ds_tools.config import select
from romulo_ds_tools.custom import call_dataframe_hook, call_optional_factory
from romulo_ds_tools.features import add_time_series_features
from romulo_ds_tools.io import ensure_dir, load_dataset, save_dataframe, write_json
from romulo_ds_tools.metrics import compute_metrics
from romulo_ds_tools.modeling import build_estimator, build_model_pipeline, prepare_data
from romulo_ds_tools.tracking import log_training_run
from romulo_ds_tools.validation import validate_dataframe


@dataclass(frozen=True)
class TrainResult:
    metrics: dict[str, float]
    model_path: Path
    predictions_path: Path
    feature_columns: list[str]
    mlflow_run_id: str | None


def load_project_frame(cfg: DictConfig, *, validate: bool = True) -> pd.DataFrame:
    df = load_dataset(select(cfg, "paths.data"))
    if validate and select(cfg, "validation.enabled", True):
        result = validate_dataframe(df, select(cfg, "validation.schema_module"))
        write_json(
            {
                "success": result.success,
                "rows": result.rows,
                "columns": result.columns,
                "schema_module": result.schema_module,
                "errors": result.errors,
            },
            select(cfg, "validation.output_path", "reports/validation.json"),
        )
        if not result.success:
            raise ValueError(f"Data validation failed with {len(result.errors)} error(s)")
    return df


def apply_project_transforms(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    df = call_dataframe_hook("project.preprocessing", "preprocess", df, cfg)
    df = call_dataframe_hook("project.features", "build_features", df, cfg)
    if select(cfg, "features.time_series.enabled", False):
        df = add_time_series_features(
            df,
            target_column=select(cfg, "data.target"),
            time_column=select(cfg, "data.time_column"),
            lags=select(cfg, "features.time_series.lags", []),
            rolling_windows=select(cfg, "features.time_series.rolling_windows", []),
            group_columns=select(cfg, "features.time_series.group_columns", []),
        )
    return df


def _list_config(cfg: DictConfig, key: str) -> list[str] | None:
    value = select(cfg, key)
    if value is None:
        return None
    return list(value)


def run_train(cfg: DictConfig) -> TrainResult:
    df = apply_project_transforms(load_project_frame(cfg), cfg)

    problem_type = select(cfg, "data.problem_type")
    target_column = select(cfg, "data.target")
    time_column = select(cfg, "data.time_column")
    prepared = prepare_data(
        df,
        target_column=target_column,
        problem_type=problem_type,
        test_size=float(select(cfg, "data.test_size", 0.2)),
        random_state=int(select(cfg, "training.random_state", 42)),
        feature_columns=_list_config(cfg, "data.feature_columns"),
        exclude_columns=_list_config(cfg, "data.exclude_columns") or [],
        id_columns=_list_config(cfg, "data.id_columns") or [],
        time_column=time_column,
        drop_missing_target=bool(select(cfg, "training.drop_rows_with_missing_target", True)),
        drop_missing_features=bool(select(cfg, "training.drop_rows_with_missing_features", True)),
    )

    estimator = call_optional_factory("project.models", "get_model", cfg)
    if estimator is None:
        estimator = build_estimator(
            problem_type,
            select(cfg, "model.name"),
            OmegaConf.to_container(select(cfg, "model.params", {}), resolve=True),
        )
    model = build_model_pipeline(estimator)
    model.fit(prepared.x_train, prepared.y_train)
    y_pred = model.predict(prepared.x_test)
    metrics = compute_metrics(problem_type, prepared.y_test, y_pred)

    model_path = Path(select(cfg, "paths.model_path", "models/model.joblib"))
    ensure_dir(model_path.parent)
    joblib.dump(model, model_path)

    reports_dir = ensure_dir(select(cfg, "paths.reports_dir", "reports"))
    predictions_path = reports_dir / "predictions.csv"
    predictions = pd.DataFrame({"y_true": prepared.y_test.to_numpy(), "y_pred": y_pred})
    save_dataframe(predictions, predictions_path)
    metrics_path = reports_dir / "metrics.json"
    write_json(
        {
            "metrics": metrics,
            "problem_type": problem_type,
            "target": target_column,
            "feature_columns": prepared.feature_columns,
            "rows_train": int(len(prepared.x_train)),
            "rows_test": int(len(prepared.x_test)),
        },
        metrics_path,
    )
    run_id = log_training_run(
        cfg,
        metrics=metrics,
        model_path=model_path,
        artifact_paths=[metrics_path, predictions_path],
    )
    return TrainResult(metrics, model_path, predictions_path, prepared.feature_columns, run_id)


def run_evaluate(cfg: DictConfig) -> dict[str, Any]:
    predictions_path = Path(select(cfg, "paths.reports_dir", "reports")) / "predictions.csv"
    predictions = pd.read_csv(predictions_path)
    metrics = compute_metrics(
        select(cfg, "data.problem_type"), predictions["y_true"], predictions["y_pred"]
    )
    output = {
        "metrics": metrics,
        "predictions_path": str(predictions_path),
    }
    write_json(output, Path(select(cfg, "paths.reports_dir", "reports")) / "evaluation.json")
    return output
