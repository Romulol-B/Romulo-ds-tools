from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

DEFAULT_REGISTRY_PATH = Path("conf/datasets.yaml")
LEGACY_CONFIG_PATH = Path("conf/config.yaml")


def _resolve_extended_path(parent_path: Path, extended_path: str | Path) -> Path:
    candidate = Path(extended_path)
    if candidate.is_absolute():
        return candidate
    return (parent_path.parent / candidate).resolve()


def _load_config_tree(config_path: Path, seen: set[Path] | None = None) -> DictConfig:
    resolved_path = config_path.resolve()
    seen = seen or set()
    if resolved_path in seen:
        chain = " -> ".join(str(path) for path in [*seen, resolved_path])
        raise ValueError(f"Circular config extends detected: {chain}")
    seen.add(resolved_path)

    cfg = OmegaConf.load(resolved_path)
    extends_value = OmegaConf.select(cfg, "extends")
    if not extends_value:
        return cfg

    cfg.pop("extends")
    extends_paths = extends_value if isinstance(extends_value, list) else [extends_value]
    base_cfg = OmegaConf.create()
    for extended_path in extends_paths:
        base_cfg = OmegaConf.merge(
            base_cfg,
            _load_config_tree(_resolve_extended_path(resolved_path, extended_path), seen.copy()),
        )
    return OmegaConf.merge(base_cfg, cfg)


def load_config(config_path: str | Path, overrides: Iterable[str] | None = None) -> DictConfig:
    """Load a YAML config with optional relative ``extends`` and dotlist overrides."""
    cfg = _load_config_tree(Path(config_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def load_registry(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> DictConfig:
    path = Path(registry_path)
    if not path.exists():
        return OmegaConf.create({"active": None, "datasets": {}})
    return OmegaConf.load(path)


def save_registry(registry: DictConfig, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> Path:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=registry, f=path)
    return path


def resolve_config_path(
    config_path: str | Path | None = None,
    dataset: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    """Resolve the config path from explicit config, dataset name, active dataset, or legacy default."""
    if config_path is not None:
        return Path(config_path)

    registry_file = Path(registry_path)
    if registry_file.exists():
        registry = load_registry(registry_file)
        dataset_name = dataset or OmegaConf.select(registry, "active")
        if not dataset_name:
            raise ValueError("No dataset selected and no active dataset configured")
        dataset_config = OmegaConf.select(registry, f"datasets.{dataset_name}.config")
        if dataset_config is None:
            available = sorted((OmegaConf.select(registry, "datasets") or {}).keys())
            raise ValueError(
                f"Dataset {dataset_name!r} is not registered. Available datasets: {available}"
            )
        return _resolve_extended_path(registry_file.resolve(), dataset_config)

    if dataset:
        raise ValueError(
            f"Dataset registry {registry_file} does not exist. Use --config or create the registry."
        )
    return LEGACY_CONFIG_PATH


def select(cfg: DictConfig, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def to_plain_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
