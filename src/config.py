"""Configuration loading, defaults and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Single source of truth for defaults. Values in config.yaml override these.
DEFAULTS: dict[str, Any] = {
    "data": {
        "date_column": "date",
        "store_column": "store_id",
        "target_column": "sales",
        "date_format": None,
    },
    "features": {
        "lag_days": [1, 7, 14, 28],
        "rolling_windows": [7, 14, 28],
        "rolling_stats": ["mean", "std", "min", "max"],
        "include_date_features": True,
        "holidays": [],
    },
    "train": {
        "test_size_ratio": 0.2,
        "val_size_ratio": 0.1,
        "early_stopping_rounds": 50,
    },
    "model": {
        "type": "xgboost",
        "xgboost": {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "random_state": 42,
        },
        "sklearn": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        },
    },
    "evaluation": {
        "horizon": 14,
        "backtest_step_days": 7,
        "seasonal_period": 7,
    },
    "paths": {
        "data_raw": "data/raw/sales.csv",
        "data_processed": "data/processed/features.parquet",
        "model_artifact": "models/model.joblib",
        "metrics_output": "reports/metrics.json",
    },
}

MODEL_TYPES = ("xgboost", "sklearn")
ROLLING_STATS = ("mean", "std", "min", "max")


class ConfigError(ValueError):
    """Raised when the configuration is invalid."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _positive_int_list(value: Any, name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name} must be a non-empty list, got {value!r}")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value):
        raise ConfigError(f"{name} must contain positive integers, got {value!r}")
    if len(set(value)) != len(value):
        raise ConfigError(f"{name} contains duplicates: {value!r}")


def validate_config(config: dict[str, Any]) -> None:
    """Raise ConfigError if the configuration is inconsistent."""
    feat = config["features"]
    _positive_int_list(feat["lag_days"], "features.lag_days")
    _positive_int_list(feat["rolling_windows"], "features.rolling_windows")
    stats = feat["rolling_stats"]
    if not isinstance(stats, list) or not set(stats) <= set(ROLLING_STATS):
        raise ConfigError(f"features.rolling_stats must be a subset of {ROLLING_STATS}, got {stats!r}")
    if not isinstance(feat["holidays"], list):
        raise ConfigError("features.holidays must be a list of dates")

    tr = config["train"]
    test_r, val_r = tr["test_size_ratio"], tr["val_size_ratio"]
    if not 0 < test_r < 1:
        raise ConfigError(f"train.test_size_ratio must be in (0, 1), got {test_r}")
    if not 0 <= val_r < 1:
        raise ConfigError(f"train.val_size_ratio must be in [0, 1), got {val_r}")
    if test_r + val_r >= 1:
        raise ConfigError("train.test_size_ratio + train.val_size_ratio must be < 1")

    model_type = config["model"]["type"]
    if model_type not in MODEL_TYPES:
        raise ConfigError(f"model.type must be one of {MODEL_TYPES}, got {model_type!r}")

    ev = config["evaluation"]
    for key in ("horizon", "backtest_step_days", "seasonal_period"):
        if not isinstance(ev[key], int) or ev[key] < 1:
            raise ConfigError(f"evaluation.{key} must be a positive integer, got {ev[key]!r}")


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML config, fill in defaults and validate it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    config = _deep_merge(DEFAULTS, user_cfg)
    validate_config(config)
    return config
