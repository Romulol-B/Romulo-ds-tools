from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from romulo_ds_tools.custom import import_optional_module
from romulo_ds_tools.io import load_dataset, write_json


@dataclass(frozen=True)
class ValidationResult:
    success: bool
    rows: int
    columns: int
    schema_module: str
    errors: list[dict[str, Any]]


def load_schema(schema_module: str) -> Any:
    module = import_optional_module(schema_module)
    if module is None:
        raise ModuleNotFoundError(f"Could not import schema module {schema_module!r}")
    if hasattr(module, "get_schema"):
        return module.get_schema()
    if hasattr(module, "InputSchema"):
        return module.InputSchema
    raise AttributeError(f"{schema_module} must define get_schema() or InputSchema")


def validate_dataframe(df: pd.DataFrame, schema_module: str) -> ValidationResult:
    schema = load_schema(schema_module)
    try:
        if hasattr(schema, "validate"):
            schema.validate(df, lazy=True)
        else:
            schema(df)
    except Exception as exc:  # Pandera raises different concrete classes across versions.
        failure_cases = getattr(exc, "failure_cases", None)
        errors: list[dict[str, Any]]
        if isinstance(failure_cases, pd.DataFrame):
            errors = failure_cases.astype(str).to_dict(orient="records")
        else:
            errors = [{"error": str(exc)}]
        return ValidationResult(False, len(df), len(df.columns), schema_module, errors)
    return ValidationResult(True, len(df), len(df.columns), schema_module, [])


def validate_dataset(
    data_path: str | Path,
    schema_module: str,
    output_path: str | Path | None = None,
) -> ValidationResult:
    df = load_dataset(data_path)
    result = validate_dataframe(df, schema_module)
    if output_path is not None:
        write_json(
            {
                "success": result.success,
                "rows": result.rows,
                "columns": result.columns,
                "schema_module": result.schema_module,
                "errors": result.errors,
            },
            output_path,
        )
    return result
