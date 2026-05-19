from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_dataset(path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """Load a local tabular dataset based on file extension."""
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path, **read_kwargs)
    if suffix == ".tsv":
        return pd.read_csv(dataset_path, sep="\t", **read_kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(dataset_path, **read_kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(dataset_path, **read_kwargs)
    raise ValueError(f"Unsupported dataset extension: {suffix or '<none>'}")


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    output_path = ensure_parent(path)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=index)
    elif suffix == ".parquet":
        df.to_parquet(output_path, index=index)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=index)
    else:
        raise ValueError(f"Unsupported dataframe output extension: {suffix or '<none>'}")
    return output_path


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = ensure_parent(path)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return output_path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")
