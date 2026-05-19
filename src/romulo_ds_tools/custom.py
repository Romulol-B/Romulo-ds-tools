from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pandas as pd


def import_optional_module(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise


def call_dataframe_hook(
    module_name: str,
    function_name: str,
    df: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    module = import_optional_module(module_name)
    if module is None:
        return df
    function = getattr(module, function_name, None)
    if function is None:
        return df
    result = function(df.copy(), cfg)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"{module_name}.{function_name} must return a pandas DataFrame")
    return result


def call_optional_factory(
    module_name: str,
    function_name: str,
    cfg: Any,
) -> Any | None:
    module = import_optional_module(module_name)
    if module is None:
        return None
    function: Callable[..., Any] | None = getattr(module, function_name, None)
    if function is None:
        return None
    return function(cfg)
