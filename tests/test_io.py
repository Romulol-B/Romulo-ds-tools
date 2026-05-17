from __future__ import annotations

import pandas as pd
import pytest

from romulo_ds_tools.io import load_dataset, require_columns, save_dataframe


def test_load_and_save_csv_roundtrip(tmp_path):
    source = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "sample.csv"

    save_dataframe(source, path)
    loaded = load_dataset(path)

    pd.testing.assert_frame_equal(loaded, source)


def test_load_rejects_unknown_extension(tmp_path):
    path = tmp_path / "sample.unknown"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported dataset extension"):
        load_dataset(path)


def test_require_columns_reports_missing_columns():
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(df, ["a", "b"], "test")
