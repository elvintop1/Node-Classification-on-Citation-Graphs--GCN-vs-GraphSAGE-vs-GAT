from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config file: {path}")
    return cfg


def save_yaml(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    base = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _auto_cast(value: str) -> Any:
    low = value.lower()
    if low == "null":
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _set_nested(cfg: dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    cursor = cfg
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        _set_nested(cfg, key, _auto_cast(raw_value))
    return cfg
