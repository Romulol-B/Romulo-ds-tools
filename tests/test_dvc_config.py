from __future__ import annotations

from pathlib import Path


def test_dvc_yaml_declares_core_stages():
    text = Path("dvc.yaml").read_text(encoding="utf-8")

    assert "validate:" in text
    assert "profile:" in text
    assert "train:" in text
    assert "evaluate:" in text
    assert "romds train --config conf/config.yaml" in text
