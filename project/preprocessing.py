from __future__ import annotations

import pandas as pd


def preprocess(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Clean raw data before feature engineering.

    Keep this deterministic and side-effect free. Return a new DataFrame rather than
    mutating files or global state.
    """
    return df.copy()
