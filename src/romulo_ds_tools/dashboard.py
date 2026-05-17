from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from romulo_ds_tools.config import select
from romulo_ds_tools.io import read_json


def _load_metrics(reports_dir: Path) -> dict:
    metrics_path = reports_dir / "metrics.json"
    if not metrics_path.exists():
        return {"metrics": {}, "feature_columns": []}
    return read_json(metrics_path)


def create_app(reports_dir: str | Path = "reports"):
    import plotly.express as px
    from dash import Dash, dash_table, dcc, html

    reports = Path(reports_dir)
    metrics_payload = _load_metrics(reports)
    metrics_df = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in metrics_payload.get("metrics", {}).items()
        ]
    )
    predictions_path = reports / "predictions.csv"
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()

    app = Dash(__name__)
    figures = []
    if not metrics_df.empty:
        figures.append(dcc.Graph(figure=px.bar(metrics_df, x="metric", y="value", title="Metrics")))
    if {"y_true", "y_pred"}.issubset(predictions.columns):
        figures.append(
            dcc.Graph(
                figure=px.scatter(
                    predictions, x="y_true", y="y_pred", title="Predictions vs actual"
                )
            )
        )

    app.layout = html.Div(
        [
            html.H1("Romulo DS Tools"),
            html.H2("Run metrics"),
            dash_table.DataTable(
                data=metrics_df.to_dict("records"),
                columns=[{"name": column, "id": column} for column in metrics_df.columns],
            ),
            *figures,
        ],
        style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "Arial, sans-serif"},
    )
    return app


def run_dashboard(cfg: DictConfig) -> None:
    app = create_app(select(cfg, "paths.reports_dir", "reports"))
    app.run(
        host=select(cfg, "dashboard.host", "127.0.0.1"),
        port=int(select(cfg, "dashboard.port", 8050)),
        debug=False,
    )
