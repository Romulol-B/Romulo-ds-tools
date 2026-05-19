from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from romulo_ds_tools.config import (
    DEFAULT_REGISTRY_PATH,
    load_config,
    load_registry,
    resolve_config_path,
    save_registry,
    select,
)
from romulo_ds_tools.io import load_dataset


@dataclass(frozen=True)
class DeletedPath:
    path: Path
    existed: bool
    deleted: bool


def normalize_dataset_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("Dataset name cannot be empty")
    if normalized[0].isdigit():
        normalized = f"dataset_{normalized}"
    return normalized


def schema_module_name(dataset_name: str) -> str:
    return f"project.schemas.{normalize_dataset_name(dataset_name)}"


def schema_path(dataset_name: str) -> Path:
    return Path("project") / "schemas" / f"{normalize_dataset_name(dataset_name)}.py"


def dataset_config_path(dataset_name: str) -> Path:
    normalized = normalize_dataset_name(dataset_name)
    return Path("conf") / f"{normalized}_v1" / f"{normalized}_01.yaml"


def _quote(value: Any) -> str:
    return repr(value)


def _column_definition(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "pa.Column(int, nullable=True, coerce=True)"
    if pd.api.types.is_float_dtype(series):
        return "pa.Column(float, nullable=True, coerce=True)"
    if pd.api.types.is_bool_dtype(series):
        return "pa.Column(bool, nullable=True, coerce=True)"

    non_null = series.dropna().astype(str)
    unique_values = sorted(non_null.unique().tolist())
    if 0 < len(unique_values) <= 20:
        values = ", ".join(_quote(value) for value in unique_values)
        return f"pa.Column(str, checks=pa.Check.isin([{values}]), nullable=True, coerce=True)"
    return "pa.Column(str, nullable=True, coerce=True)"


def render_schema_module(df: pd.DataFrame, dataset_name: str) -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "try:",
        "    import pandera.pandas as pa",
        "except ImportError:  # pragma: no cover - compatibility for older Pandera releases.",
        "    import pandera as pa",
        "",
        "",
        "def get_schema() -> pa.DataFrameSchema:",
        f'    """Return the raw input data contract for {normalize_dataset_name(dataset_name)}."""',
        "    return pa.DataFrameSchema(",
        "        {",
    ]
    for column in df.columns:
        lines.append(f"            {_quote(str(column))}: {_column_definition(df[column])},")
    lines.extend(
        [
            "        },",
            "        coerce=True,",
            "        strict=True,",
            "    )",
            "",
        ]
    )
    return "\n".join(lines)


def render_dataset_config(
    *,
    dataset_name: str,
    data_path: str | Path,
    target: str,
    problem_type: str,
) -> str:
    normalized = normalize_dataset_name(dataset_name)
    return "\n".join(
        [
            "extends: ../base.yaml",
            "",
            f"project_name: {normalized}-v1",
            "",
            "paths:",
            f"  data: {Path(data_path).as_posix()}",
            f"  reports_dir: reports/{normalized}_v1",
            f"  model_dir: models/{normalized}_v1",
            f"  model_path: models/{normalized}_v1/model.joblib",
            "",
            "validation:",
            f"  schema_module: {schema_module_name(normalized)}",
            f"  output_path: reports/{normalized}_v1/validation.json",
            "",
            "data:",
            f"  target: {_quote(target)}",
            f"  problem_type: {problem_type}",
            "  id_columns: []",
            "",
            "tracking:",
            f"  run_name: {normalized}_baseline",
            "",
            "visualization:",
            f"  output_dir: reports/{normalized}_v1/eda",
            "  pairplot:",
            f"    color_column: {_quote(target)}",
            "",
        ]
    )


def register_dataset(
    dataset_name: str,
    config_path: str | Path,
    *,
    make_active: bool = False,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    normalized = normalize_dataset_name(dataset_name)
    registry_file = Path(registry_path)
    config = Path(config_path)
    try:
        config_value = config.relative_to(registry_file.parent).as_posix()
    except ValueError:
        if config.parts and config.parts[0] == registry_file.parent.name:
            config_value = Path(*config.parts[1:]).as_posix()
        else:
            config_value = config.as_posix()

    registry = load_registry(registry_path)
    datasets = OmegaConf.select(registry, "datasets")
    if datasets is None:
        registry.datasets = {}
    registry.datasets[normalized] = {"config": config_value}
    if make_active or not OmegaConf.select(registry, "active"):
        registry.active = normalized
    return save_registry(registry, registry_path)


def scaffold_dataset(
    dataset_name: str,
    *,
    data_path: str | Path,
    target: str,
    problem_type: str = "classification",
    force: bool = False,
    make_active: bool = False,
) -> dict[str, Path]:
    normalized = normalize_dataset_name(dataset_name)
    df = load_dataset(data_path)
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in {data_path}")

    config_path = dataset_config_path(normalized)
    schema_file = schema_path(normalized)
    for path in (config_path, schema_file):
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists. Use --force to overwrite it.")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    schema_file.parent.mkdir(parents=True, exist_ok=True)
    (schema_file.parent.parent / "__init__.py").touch()
    (schema_file.parent / "__init__.py").touch()
    config_path.write_text(
        render_dataset_config(
            dataset_name=normalized,
            data_path=data_path,
            target=target,
            problem_type=problem_type,
        ),
        encoding="utf-8",
    )
    schema_file.write_text(render_schema_module(df, normalized), encoding="utf-8")
    register_dataset(normalized, config_path, make_active=make_active)
    return {
        "config": config_path,
        "schema": schema_file,
        "registry": DEFAULT_REGISTRY_PATH,
    }


def registry_snapshot(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry = load_registry(registry_path)
    return OmegaConf.to_container(registry, resolve=True)  # type: ignore[return-value]


def _workspace_path(path: str | Path, workspace_root: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(workspace_root) / candidate).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _dedupe_nested_paths(paths: list[Path]) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(set(paths), key=lambda item: len(item.parts)):
        if any(path == existing or _is_relative_to(path, existing) for existing in selected):
            continue
        selected.append(path)
    return selected


def dataset_report_paths(
    dataset_name: str,
    *,
    workspace_root: str | Path = ".",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[Path]:
    config_path = resolve_config_path(None, normalize_dataset_name(dataset_name), registry_path)
    cfg = load_config(config_path)
    reports_root = _workspace_path("reports", workspace_root)
    candidates: list[Path] = []

    for key in ("paths.reports_dir", "validation.output_path", "visualization.output_dir"):
        value = select(cfg, key)
        if value:
            candidates.append(_workspace_path(value, workspace_root))

    safe_paths: list[Path] = []
    for path in candidates:
        if path == reports_root:
            raise ValueError("Refusing to delete the root reports directory for a single dataset")
        if not _is_relative_to(path, reports_root):
            raise ValueError(f"Refusing to delete report path outside reports/: {path}")
        safe_paths.append(path)

    return _dedupe_nested_paths(safe_paths)


def delete_dataset_reports(
    dataset_name: str,
    *,
    dry_run: bool = True,
    workspace_root: str | Path = ".",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[DeletedPath]:
    report_paths = dataset_report_paths(
        dataset_name,
        workspace_root=workspace_root,
        registry_path=registry_path,
    )
    results: list[DeletedPath] = []
    for path in sorted(report_paths, key=lambda item: len(item.parts), reverse=True):
        existed = path.exists()
        if existed and not dry_run:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        results.append(DeletedPath(path=path, existed=existed, deleted=existed and not dry_run))
    return results
