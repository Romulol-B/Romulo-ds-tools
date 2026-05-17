from __future__ import annotations

import pandas as pd

from romulo_ds_tools.dashboard import create_app
from romulo_ds_tools.io import write_json


def test_create_app_reads_metrics_and_predictions(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    write_json({"metrics": {"accuracy": 1.0}, "feature_columns": ["x"]}, reports / "metrics.json")
    pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1]}).to_csv(
        reports / "predictions.csv", index=False
    )

    app = create_app(reports)

    assert app.layout is not None
