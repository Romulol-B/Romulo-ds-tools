from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: Iterable[str] | None = None) -> DictConfig:
    """Load a Hydra/OmegaConf-compatible YAML file and optional dotlist overrides."""
    cfg = OmegaConf.load(Path(config_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def select(cfg: DictConfig, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def to_plain_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
