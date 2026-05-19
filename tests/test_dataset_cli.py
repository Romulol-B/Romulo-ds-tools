from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from romulo_ds_tools.cli import app


def test_dataset_cli_init_list_show_set_active_and_doctor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "base.yaml").write_text("project_name: base\n", encoding="utf-8")
    data_path = tmp_path / "sample.csv"
    pd.DataFrame({"feature": [1.0, 2.0, 3.0], "target": ["a", "b", "a"]}).to_csv(
        data_path, index=False
    )
    runner = CliRunner()

    init = runner.invoke(
        app,
        [
            "datasets",
            "init",
            "sample",
            "--data",
            str(data_path),
            "--target",
            "target",
            "--active",
        ],
    )
    assert init.exit_code == 0
    assert (tmp_path / "conf" / "sample_v1" / "sample_01.yaml").exists()
    assert (tmp_path / "project" / "schemas" / "sample.py").exists()

    listing = runner.invoke(app, ["datasets", "list"])
    assert listing.exit_code == 0
    assert "* sample:" in listing.output

    show = runner.invoke(app, ["datasets", "show", "sample"])
    assert show.exit_code == 0
    assert "target: target" in show.output

    set_active = runner.invoke(app, ["datasets", "set-active", "sample"])
    assert set_active.exit_code == 0

    doctor = runner.invoke(app, ["doctor", "--dataset", "sample"])
    assert doctor.exit_code == 0
    assert "Doctor passed" in doctor.output


def test_dataset_cli_delete_reports_uses_dry_run_unless_confirmed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "base.yaml").write_text("project_name: base\n", encoding="utf-8")
    data_path = tmp_path / "sample.csv"
    report_dir = tmp_path / "reports" / "sample_v1"
    report_dir.mkdir(parents=True)
    (report_dir / "profile.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"feature": [1.0, 2.0], "target": ["a", "b"]}).to_csv(data_path, index=False)
    runner = CliRunner()
    runner.invoke(
        app,
        [
            "datasets",
            "init",
            "sample",
            "--data",
            str(data_path),
            "--target",
            "target",
        ],
    )

    dry_run = runner.invoke(app, ["datasets", "delete", "sample"])
    assert dry_run.exit_code == 0
    assert "Dry run" in dry_run.output
    assert report_dir.exists()

    delete = runner.invoke(app, ["datasets", "delete", "sample", "--yes"])
    assert delete.exit_code == 0
    assert "deleted" in delete.output
    assert not report_dir.exists()
    assert data_path.exists()
