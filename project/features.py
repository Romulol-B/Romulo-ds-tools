from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add project-specific features.

    Core time-series lag/rolling features are handled by the reusable package. Put
    dataset-specific ratios, flags, encodings, and domain features here.
    """
    return df.copy()
