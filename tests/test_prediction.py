from __future__ import annotations

import pandas as pd

from romulo_ds_tools.prediction import run_predict
from romulo_ds_tools.training import run_train


def test_run_predict_writes_batch_predictions(tmp_path, classification_frame, config_factory):
    data_path = tmp_path / "train.csv"
    output_path = tmp_path / "reports" / "batch.csv"
    classification_frame.to_csv(data_path, index=False)
    cfg = config_factory(data_path)
    run_train(cfg)

    destination = run_predict(cfg, output_path=output_path)
    predictions = pd.read_csv(destination)

    assert destination == output_path
    assert list(predictions.columns) == ["prediction"]
    assert len(predictions) == len(classification_frame)
