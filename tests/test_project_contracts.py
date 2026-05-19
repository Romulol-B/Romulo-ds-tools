from __future__ import annotations

import pandas as pd

from project.features import build_features
from project.models import get_model
from project.preprocessing import preprocess
from project.schema import get_schema
from project.schemas.nursery import get_schema as get_nursery_schema
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


def test_project_schema_validates_prostate_contract(prostate_frame):
    validated = get_schema().validate(prostate_frame)

    assert list(validated.columns) == list(prostate_frame.columns)


def test_nursery_schema_validates_nursery_contract():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "parents": ["usual"],
            "has_nurs": ["proper"],
            "form": ["complete"],
            "children": ["1"],
            "housing": ["convenient"],
            "finance": ["convenient"],
            "social": ["nonprob"],
            "health": ["recommended"],
            "final evaluation": ["recommend"],
        }
    )

    validated = get_nursery_schema().validate(frame)

    assert list(validated.columns) == list(frame.columns)
