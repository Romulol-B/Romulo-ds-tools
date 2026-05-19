from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from romulo_ds_tools.config import select, to_plain_dict


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(next_prefix, item))
        return flattened
    return {prefix: value}


def log_training_run(
    cfg: DictConfig,
    *,
    metrics: dict[str, float],
    model_path: str | Path,
    artifact_paths: list[str | Path] | None = None,
) -> str | None:
    if not select(cfg, "tracking.enabled", True):
        return None

    import mlflow

    tracking_uri = select(cfg, "tracking.mlflow_tracking_uri", "file:./mlruns")
    experiment_name = select(cfg, "tracking.experiment_name", "romulo-ds-tools")
    run_name = select(cfg, "tracking.run_name", None)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        plain_cfg = to_plain_dict(cfg)
        for key, value in _flatten("", plain_cfg).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                mlflow.log_param(key, value)
        mlflow.log_metrics(metrics)
        if Path(model_path).exists():
            mlflow.log_artifact(str(model_path), artifact_path="model")
        for artifact_path in artifact_paths or []:
            if Path(artifact_path).exists():
                mlflow.log_artifact(str(artifact_path))
        return run.info.run_id


def config_to_yaml(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)
