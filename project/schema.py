from __future__ import annotations

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover - compatibility for older Pandera releases.
    import pandera as pa


def get_schema() -> pa.DataFrameSchema:
    """Return the raw input data contract.

    The default schema is intentionally permissive. Replace it with explicit columns
    as soon as a real dataset is connected.
    """
    return pa.DataFrameSchema({}, coerce=True, strict=False)
