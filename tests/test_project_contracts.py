from __future__ import annotations

import pandas as pd

from project.features import build_features
from project.models import get_model
from project.preprocessing import preprocess
from project.schema import get_schema
from project.visualization import build_figures


def test_project_hooks_preserve_expected_contracts(classification_frame):
    preprocessed = preprocess(classification_frame, cfg={})
    featured = build_features(preprocessed, cfg={})
    figures = build_figures(featured, cfg={})

    assert isinstance(preprocessed, pd.DataFrame)
    assert isinstance(featured, pd.DataFrame)
    assert isinstance(figures, dict)
    assert get_model(cfg={}) is None
    assert hasattr(get_schema(), "validate")
