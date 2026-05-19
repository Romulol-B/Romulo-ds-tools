from __future__ import annotations

import sys

from romulo_ds_tools.cli import _add_project_import_paths


def test_cli_adds_config_parent_root_to_import_path(tmp_path):
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "conf"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text("project_name: test\n", encoding="utf-8")
    repo_root_text = str(repo_root)
    original_path = list(sys.path)

    try:
        sys.path = [entry for entry in sys.path if entry != repo_root_text]
        _add_project_import_paths(config_path)
        assert repo_root_text in sys.path
    finally:
        sys.path = original_path
