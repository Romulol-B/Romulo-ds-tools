from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import OmegaConf

from romulo_ds_tools.config import (
    load_config,
    load_registry,
    resolve_config_path,
    save_registry,
    select,
)
from romulo_ds_tools.dashboard import run_dashboard
from romulo_ds_tools.datasets import registry_snapshot, scaffold_dataset
from romulo_ds_tools.io import load_dataset
from romulo_ds_tools.prediction import run_predict
from romulo_ds_tools.profiling import run_profile
from romulo_ds_tools.training import run_evaluate, run_train
from romulo_ds_tools.validation import load_schema, validate_dataset

app = typer.Typer(help="Reusable commands for the Romulo data science project template.")
datasets_app = typer.Typer(help="Manage dataset registry entries.")
app.add_typer(datasets_app, name="datasets")


ConfigPath = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Path to a YAML config file. Takes precedence over --dataset.",
    ),
]
DatasetName = Annotated[
    str | None,
    typer.Option("--dataset", "-d", help="Registered dataset name from conf/datasets.yaml."),
]
Overrides = Annotated[
    list[str] | None,
    typer.Option("--override", "-o", help="OmegaConf dotlist override, e.g. data.target=price."),
]


def _add_project_import_paths(config: Path) -> None:
    """Make project hooks importable when the console script starts outside repo root."""
    resolved_config = config.resolve()
    candidates = [
        Path.cwd(),
        resolved_config.parent,
        resolved_config.parent.parent,
    ]
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate.exists() and candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
        project_dir = candidate / "project"
        if project_dir.exists() and "project" in sys.modules:
            project_path = getattr(sys.modules["project"], "__path__", None)
            project_dir_text = str(project_dir)
            if project_path is not None and project_dir_text not in project_path:
                project_path.insert(0, project_dir_text)
        schemas_dir = project_dir / "schemas"
        if schemas_dir.exists() and "project.schemas" in sys.modules:
            schemas_path = getattr(sys.modules["project.schemas"], "__path__", None)
            schemas_dir_text = str(schemas_dir)
            if schemas_path is not None and schemas_dir_text not in schemas_path:
                schemas_path.insert(0, schemas_dir_text)
    importlib.invalidate_caches()


def _config_path(config: Path | None, dataset: str | None) -> Path:
    return resolve_config_path(config, dataset)


def _cfg(config: Path | None, dataset: str | None, override: list[str] | None):
    resolved_config = _config_path(config, dataset)
    _add_project_import_paths(resolved_config)
    return load_config(resolved_config, override)


@app.command()
def validate(
    config: ConfigPath = None,
    dataset: DatasetName = None,
    override: Overrides = None,
) -> None:
    cfg = _cfg(config, dataset, override)
    result = validate_dataset(
        select(cfg, "paths.data"),
        select(cfg, "validation.schema_module"),
        select(cfg, "validation.output_path", "reports/validation.json"),
    )
    if not result.success:
        raise typer.Exit(code=1)
    typer.echo(f"Validation passed: {result.rows} rows, {result.columns} columns")


@app.command()
def profile(
    config: ConfigPath = None,
    dataset: DatasetName = None,
    override: Overrides = None,
) -> None:
    cfg = _cfg(config, dataset, override)
    summary = run_profile(cfg)
    typer.echo(f"Profile written for {summary['rows']} rows and {summary['columns']} columns")


@app.command()
def train(
    config: ConfigPath = None,
    dataset: DatasetName = None,
    override: Overrides = None,
) -> None:
    cfg = _cfg(config, dataset, override)
    result = run_train(cfg)
    typer.echo(f"Model written to {result.model_path}")
    typer.echo(f"Predictions written to {result.predictions_path}")


@app.command()
def evaluate(
    config: ConfigPath = None,
    dataset: DatasetName = None,
    override: Overrides = None,
) -> None:
    cfg = _cfg(config, dataset, override)
    result = run_evaluate(cfg)
    typer.echo(f"Evaluation written for {result['predictions_path']}")


@app.command()
def predict(
    config: ConfigPath = None,
    dataset: DatasetName = None,
    input_path: Annotated[Path | None, typer.Option("--input", "-i")] = None,
    output_path: Annotated[Path | None, typer.Option("--output", "-p")] = None,
    override: Overrides = None,
) -> None:
    cfg = _cfg(config, dataset, override)
    destination = run_predict(cfg, input_path=input_path, output_path=output_path)
    typer.echo(f"Predictions written to {destination}")


@app.command()
def dashboard(
    config: ConfigPath = None,
    dataset: DatasetName = None,
    override: Overrides = None,
) -> None:
    cfg = _cfg(config, dataset, override)
    run_dashboard(cfg)


@app.command()
def doctor(
    config: ConfigPath = None,
    dataset: DatasetName = None,
    override: Overrides = None,
) -> None:
    config_path = _config_path(config, dataset)
    _add_project_import_paths(config_path)
    cfg = load_config(config_path, override)
    failures: list[str] = []
    typer.echo(f"Config: {config_path}")

    data_path = Path(select(cfg, "paths.data", ""))
    if not data_path.exists():
        failures.append(f"Data file does not exist: {data_path}")
    else:
        df = load_dataset(data_path)
        typer.echo(f"Data: {data_path} ({len(df)} rows, {len(df.columns)} columns)")
        target = select(cfg, "data.target")
        if target not in df.columns:
            failures.append(f"Target column {target!r} was not found in {data_path}")

    schema_module = select(cfg, "validation.schema_module")
    if not schema_module:
        failures.append("validation.schema_module is not configured")
    else:
        try:
            load_schema(schema_module)
            typer.echo(f"Schema: {schema_module}")
        except Exception as exc:
            failures.append(f"Schema {schema_module!r} could not be loaded: {exc}")

    for key in ("paths.reports_dir", "paths.model_path"):
        if not select(cfg, key):
            failures.append(f"{key} is not configured")

    if failures:
        typer.echo("Problems:")
        for failure in failures:
            typer.echo(f"- {failure}")
        raise typer.Exit(code=1)

    typer.echo("Doctor passed")


@datasets_app.command("list")
def list_datasets() -> None:
    snapshot = registry_snapshot()
    active = snapshot.get("active")
    datasets = snapshot.get("datasets") or {}
    if not datasets:
        typer.echo("No datasets registered")
        return
    for name, item in sorted(datasets.items()):
        marker = "*" if name == active else " "
        typer.echo(f"{marker} {name}: {item['config']}")


@datasets_app.command("show")
def show_dataset(dataset: Annotated[str, typer.Argument(help="Registered dataset name.")]) -> None:
    config_path = resolve_config_path(None, dataset)
    _add_project_import_paths(config_path)
    cfg = load_config(config_path)
    typer.echo(OmegaConf.to_yaml(cfg, resolve=True))


@datasets_app.command("set-active")
def set_active_dataset(
    dataset: Annotated[str, typer.Argument(help="Registered dataset name.")],
) -> None:
    registry = load_registry()
    if OmegaConf.select(registry, f"datasets.{dataset}.config") is None:
        available = sorted((OmegaConf.select(registry, "datasets") or {}).keys())
        typer.echo(f"Dataset {dataset!r} is not registered. Available datasets: {available}")
        raise typer.Exit(code=1)
    registry.active = dataset
    save_registry(registry)
    typer.echo(f"Active dataset set to {dataset}")


@datasets_app.command("init")
def init_dataset(
    name: Annotated[str, typer.Argument(help="Dataset registry name.")],
    data: Annotated[
        Path, typer.Option("--data", help="Path to local CSV, Excel, or Parquet file.")
    ],
    target: Annotated[str, typer.Option("--target", help="Target column name.")],
    problem_type: Annotated[
        str, typer.Option("--problem-type", help="classification or regression.")
    ] = "classification",
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite generated files if present.")
    ] = False,
    make_active: Annotated[
        bool,
        typer.Option("--active", help="Make this dataset the active default after registering it."),
    ] = False,
) -> None:
    created = scaffold_dataset(
        name,
        data_path=data,
        target=target,
        problem_type=problem_type,
        force=force,
        make_active=make_active,
    )
    typer.echo(f"Config written to {created['config']}")
    typer.echo(f"Schema written to {created['schema']}")
    typer.echo(f"Registry updated at {created['registry']}")
