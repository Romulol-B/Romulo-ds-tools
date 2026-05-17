from __future__ import annotations

import pandas as pd

from romulo_ds_tools.training import run_evaluate, run_train


def test_run_train_writes_model_metrics_and_predictions(
    tmp_path,
    classification_frame,
    config_factory,
):
    data_path = tmp_path / "train.csv"
    classification_frame.to_csv(data_path, index=False)
    cfg = config_factory(data_path)

    result = run_train(cfg)

    assert result.model_path.exists()
    assert result.predictions_path.exists()
    assert "accuracy" in result.metrics
    assert result.mlflow_run_id is None


def test_run_evaluate_recomputes_metrics_from_predictions(
    tmp_path, classification_frame, config_factory
):
    data_path = tmp_path / "train.csv"
    classification_frame.to_csv(data_path, index=False)
    cfg = config_factory(data_path)
    run_train(cfg)

    output = run_evaluate(cfg)

    assert "accuracy" in output["metrics"]
    assert (tmp_path / "reports" / "evaluation.json").exists()


def test_run_train_supports_time_series_feature_config(tmp_path, time_series_frame, config_factory):
    data_path = tmp_path / "series.csv"
    time_series_frame.to_csv(data_path, index=False)
    cfg = config_factory(
        data_path,
        problem_type="regression",
        model_name="random_forest_regressor",
        time_column="date",
        time_series_enabled=True,
    )

    result = run_train(cfg)
    predictions = pd.read_csv(result.predictions_path)

    assert result.model_path.exists()
    assert "target_lag_1" in result.feature_columns
    assert not predictions.empty
