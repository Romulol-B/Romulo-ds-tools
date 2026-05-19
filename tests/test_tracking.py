from __future__ import annotations

from romulo_ds_tools.tracking import log_training_run


def test_log_training_run_is_noop_when_tracking_disabled(
    tmp_path, config_factory, classification_frame
):
    data_path = tmp_path / "train.csv"
    model_path = tmp_path / "models" / "model.joblib"
    classification_frame.to_csv(data_path, index=False)
    model_path.parent.mkdir()
    model_path.write_bytes(b"placeholder")
    cfg = config_factory(data_path)
    cfg.tracking.enabled = False

    run_id = log_training_run(cfg, metrics={"accuracy": 1.0}, model_path=model_path)

    assert run_id is None
