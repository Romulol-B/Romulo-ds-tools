from __future__ import annotations

from pathlib import Path

import pandas as pd

from romulo_ds_tools.config import load_config
from romulo_ds_tools.datasets import (
    dataset_report_paths,
    delete_dataset_reports,
    normalize_dataset_name,
    registry_snapshot,
    scaffold_dataset,
)


def test_scaffold_dataset_creates_config_schema_and_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "base.yaml").write_text("project_name: base\n", encoding="utf-8")
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1.0, 2.0], "label": ["yes", "no"]}).to_csv(data_path, index=False)

    created = scaffold_dataset(
        "My Dataset",
        data_path=data_path,
        target="label",
        make_active=True,
    )

    assert normalize_dataset_name("My Dataset") == "my_dataset"
    assert created["config"].exists()
    assert created["schema"].exists()
    assert "pa.Check.isin(['no', 'yes'])" in created["schema"].read_text(encoding="utf-8")
    snapshot = registry_snapshot()
    assert snapshot["active"] == "my_dataset"
    assert snapshot["datasets"]["my_dataset"]["config"] == "my_dataset_v1/my_dataset_01.yaml"
    cfg = load_config(created["config"])
    assert cfg.data.target == "label"
    assert Path(cfg.paths.data) == data_path


def test_delete_dataset_reports_is_dry_run_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "base.yaml").write_text("project_name: base\n", encoding="utf-8")
    data_path = tmp_path / "data.csv"
    report_dir = tmp_path / "reports" / "sample_v1"
    report_dir.mkdir(parents=True)
    report_file = report_dir / "profile.json"
    report_file.write_text("{}", encoding="utf-8")
    pd.DataFrame({"x": [1.0], "label": ["yes"]}).to_csv(data_path, index=False)
    scaffold_dataset("sample", data_path=data_path, target="label")

    results = delete_dataset_reports("sample", dry_run=True)

    assert report_file.exists()
    assert results[0].existed is True
    assert results[0].deleted is False


def test_delete_dataset_reports_deletes_reports_without_touching_raw_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "base.yaml").write_text("project_name: base\n", encoding="utf-8")
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    data_path = raw_dir / "data.csv"
    report_dir = tmp_path / "reports" / "sample_v1"
    report_dir.mkdir(parents=True)
    (report_dir / "profile.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"x": [1.0], "label": ["yes"]}).to_csv(data_path, index=False)
    scaffold_dataset("sample", data_path=data_path, target="label")

    results = delete_dataset_reports("sample", dry_run=False)

    assert not report_dir.exists()
    assert data_path.exists()
    assert results[0].deleted is True


def test_dataset_report_paths_refuses_paths_outside_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conf_dir = tmp_path / "conf"
    dataset_dir = conf_dir / "unsafe_v1"
    dataset_dir.mkdir(parents=True)
    (conf_dir / "base.yaml").write_text("project_name: base\n", encoding="utf-8")
    (conf_dir / "datasets.yaml").write_text(
        "active: unsafe\ndatasets:\n  unsafe:\n    config: unsafe_v1/unsafe_01.yaml\n",
        encoding="utf-8",
    )
    (dataset_dir / "unsafe_01.yaml").write_text(
        "\n".join(
            [
                "extends: ../base.yaml",
                "paths:",
                "  data: data/raw/unsafe.csv",
                "  reports_dir: data/raw",
            ]
        ),
        encoding="utf-8",
    )

    try:
        dataset_report_paths("unsafe")
    except ValueError as exc:
        assert "outside reports" in str(exc)
    else:
        raise AssertionError("Expected unsafe report path to be rejected")
