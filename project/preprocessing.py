from __future__ import annotations

from typing import Any

import pandas as pd


def preprocess(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Clean raw data before feature engineering.

    Keep this deterministic and side-effect free. Return a new DataFrame rather than
    mutating files or global state.
    """
    return df.copy()


def get_column_transformers(cfg) -> list[tuple[str, Any, list[str]]]:
    """Return optional sklearn ColumnTransformer entries for specific columns.

    Use this for transformations that must be fitted only on the training split,
    such as imputing, scaling, encoding, binning, or log transforms for selected
    columns. Each entry should be ``(name, transformer, columns)``.
    """
    return []
