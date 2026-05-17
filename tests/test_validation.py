from __future__ import annotations

import pandas as pd

from romulo_ds_tools.validation import validate_dataframe


def test_validate_dataframe_passes_with_dynamic_schema(tmp_path, monkeypatch):
    schema_file = tmp_path / "custom_schema.py"
    schema_file.write_text(
        "\n".join(
            [
                "import pandera.pandas as pa",
                "def get_schema():",
                "    return pa.DataFrameSchema({'a': pa.Column(int), 'b': pa.Column(str)})",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    result = validate_dataframe(pd.DataFrame({"a": [1], "b": ["x"]}), "custom_schema")

    assert result.success is True
    assert result.errors == []


def test_validate_dataframe_returns_structured_failure(tmp_path, monkeypatch):
    schema_file = tmp_path / "strict_schema.py"
    schema_file.write_text(
        "\n".join(
            [
                "import pandera.pandas as pa",
                "def get_schema():",
                "    return pa.DataFrameSchema({'a': pa.Column(int)}, strict=True)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    result = validate_dataframe(pd.DataFrame({"wrong": ["x"]}), "strict_schema")

    assert result.success is False
    assert result.errors
