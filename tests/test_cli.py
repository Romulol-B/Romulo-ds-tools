from __future__ import annotations

from typer.testing import CliRunner

from romulo_ds_tools.cli import app
from romulo_ds_tools.tracking import config_to_yaml


def test_cli_validate_command_writes_validation_report(tmp_path, prostate_frame, config_factory):
    data_path = tmp_path / "train.csv"
    config_path = tmp_path / "config.yaml"
    prostate_frame.to_csv(data_path, index=False)
    cfg = config_factory(data_path)
    cfg.validation.enabled = True
    cfg.data.target = "diagnosis_result"
    config_path.write_text(config_to_yaml(cfg), encoding="utf-8")

    result = CliRunner().invoke(app, ["validate", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "validation.json").exists()


def test_cli_train_accepts_dotlist_override(tmp_path, regression_frame, config_factory):
    data_path = tmp_path / "train.csv"
    config_path = tmp_path / "config.yaml"
    regression_frame.to_csv(data_path, index=False)
    cfg = config_factory(
        data_path,
        problem_type="regression",
        model_name="random_forest_regressor",
    )
    config_path.write_text(config_to_yaml(cfg), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--config",
            str(config_path),
            "--override",
            "model.params.n_estimators=3",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "models" / "model.joblib").exists()
