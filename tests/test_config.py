from __future__ import annotations

from omegaconf import OmegaConf

from romulo_ds_tools.config import load_config, resolve_config_path


def test_load_config_supports_relative_extends_and_cli_overrides(tmp_path):
    base = tmp_path / "base.yaml"
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    dataset = dataset_dir / "sample.yaml"
    base.write_text(
        "\n".join(
            [
                "project_name: base",
                "paths:",
                "  reports_dir: reports/base",
                "data:",
                "  problem_type: classification",
                "  test_size: 0.2",
            ]
        ),
        encoding="utf-8",
    )
    dataset.write_text(
        "\n".join(
            [
                "extends: ../base.yaml",
                "project_name: sample",
                "paths:",
                "  data: data/raw/sample.csv",
                "data:",
                "  target: label",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(dataset, ["data.test_size=0.3"])

    assert cfg.project_name == "sample"
    assert cfg.paths.reports_dir == "reports/base"
    assert cfg.paths.data == "data/raw/sample.csv"
    assert cfg.data.problem_type == "classification"
    assert cfg.data.target == "label"
    assert cfg.data.test_size == 0.3


def test_resolve_config_path_uses_dataset_registry(tmp_path):
    conf = tmp_path / "conf"
    dataset_dir = conf / "sample_v1"
    dataset_dir.mkdir(parents=True)
    registry = conf / "datasets.yaml"
    config = dataset_dir / "sample_01.yaml"
    config.write_text("project_name: sample\n", encoding="utf-8")
    OmegaConf.save(
        config={
            "active": "sample",
            "datasets": {"sample": {"config": "sample_v1/sample_01.yaml"}},
        },
        f=registry,
    )

    assert resolve_config_path(None, "sample", registry) == config.resolve()
    assert resolve_config_path(None, None, registry) == config.resolve()
