"""YAML configuration loading with local, explicit inheritance."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a config and, if requested, merge one parent config from its directory."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Top-level YAML configuration must be a mapping")
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = (config_path.parent / str(parent)).resolve()
    if parent_path == config_path:
        raise ValueError("Configuration cannot extend itself")
    return _deep_merge(load_config(parent_path), config)
