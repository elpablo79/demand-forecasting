from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import DEFAULTS, _deep_merge
from src.features import Columns


def make_sales(n_days: int = 200, stores: tuple[str, ...] = ("1", "2"), seed: int = 0) -> pd.DataFrame:
    """Synthetic daily sales with a weekly pattern and noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    parts = []
    for i, store in enumerate(stores):
        weekly = 20 * np.sin(2 * np.pi * dates.dayofweek / 7)
        sales = 100 + 10 * i + weekly + rng.normal(0, 5, n_days)
        parts.append(pd.DataFrame({"date": dates, "store_id": store, "sales": sales.round()}))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def cols() -> Columns:
    return Columns()


@pytest.fixture
def sales() -> pd.DataFrame:
    return make_sales()


def write_config(tmp_path: Path, data: pd.DataFrame, **overrides: dict) -> Path:
    data_path = tmp_path / "sales.csv"
    data.to_csv(data_path, index=False)
    cfg = _deep_merge(
        DEFAULTS,
        {
            "model": {"xgboost": {"n_estimators": 50}},
            "evaluation": {"horizon": 5},
            "paths": {
                "data_raw": str(data_path),
                "data_processed": None,
                "model_artifact": str(tmp_path / "model.joblib"),
                "metrics_output": str(tmp_path / "metrics.json"),
            },
        },
    )
    cfg = _deep_merge(cfg, overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def config_path(tmp_path: Path, sales: pd.DataFrame) -> Path:
    return write_config(tmp_path, sales)
