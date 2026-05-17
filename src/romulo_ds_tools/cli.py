from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from romulo_ds_tools.config import load_config, select
from romulo_ds_tools.dashboard import run_dashboard
from romulo_ds_tools.prediction import run_predict
from romulo_ds_tools.profiling import run_profile
from romulo_ds_tools.training import run_evaluate, run_train
from romulo_ds_tools.validation import validate_dataset

app = typer.Typer(help="Reusable commands for the Romulo data science project template.")


ConfigPath = Annotated[Path, typer.Option("--config", "-c", help="Path to a YAML config file.")]
Overrides = Annotated[
    list[str] | None,
    typer.Option("--override", "-o", help="OmegaConf dotlist override, e.g. data.target=price."),
]


def _cfg(config: Path, override: list[str] | None):
    return load_config(config, override)


@app.command()
def validate(config: ConfigPath = Path("conf/config.yaml"), override: Overrides = None) -> None:
    cfg = _cfg(config, override)
    result = validate_dataset(
        select(cfg, "paths.data"),
        select(cfg, "validation.schema_module"),
        select(cfg, "validation.output_path", "reports/validation.json"),
    )
    if not result.success:
        raise typer.Exit(code=1)
    typer.echo(f"Validation passed: {result.rows} rows, {result.columns} columns")


@app.command()
def profile(config: ConfigPath = Path("conf/config.yaml"), override: Overrides = None) -> None:
    cfg = _cfg(config, override)
    summary = run_profile(cfg)
    typer.echo(f"Profile written for {summary['rows']} rows and {summary['columns']} columns")


@app.command()
def train(config: ConfigPath = Path("conf/config.yaml"), override: Overrides = None) -> None:
    cfg = _cfg(config, override)
    result = run_train(cfg)
    typer.echo(f"Model written to {result.model_path}")
    typer.echo(f"Predictions written to {result.predictions_path}")


@app.command()
def evaluate(config: ConfigPath = Path("conf/config.yaml"), override: Overrides = None) -> None:
    cfg = _cfg(config, override)
    result = run_evaluate(cfg)
    typer.echo(f"Evaluation written for {result['predictions_path']}")


@app.command()
def predict(
    config: ConfigPath = Path("conf/config.yaml"),
    input_path: Annotated[Path | None, typer.Option("--input", "-i")] = None,
    output_path: Annotated[Path | None, typer.Option("--output", "-p")] = None,
    override: Overrides = None,
) -> None:
    cfg = _cfg(config, override)
    destination = run_predict(cfg, input_path=input_path, output_path=output_path)
    typer.echo(f"Predictions written to {destination}")


@app.command()
def dashboard(config: ConfigPath = Path("conf/config.yaml"), override: Overrides = None) -> None:
    cfg = _cfg(config, override)
    run_dashboard(cfg)
