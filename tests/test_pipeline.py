from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ConfigError, load_config
from src.evaluate import compute_metrics, evaluate
from src.model import Artifact
from src.predict import forecast, predict
from src.train import compute_split_dates, split_by_dates, train
from tests.conftest import write_config


def test_split_has_no_shared_dates(sales: pd.DataFrame) -> None:
    split = compute_split_dates(sales["date"], test_ratio=0.2, val_ratio=0.1)
    tr, va, te = split_by_dates(sales, "date", split)
    assert len(tr) + len(va) + len(te) == len(sales)
    assert tr["date"].max() < va["date"].min() <= va["date"].max() < te["date"].min()
    # Every date has all of its stores on the same side of the split.
    for part in (tr, va, te):
        assert (part.groupby("date").size() == 2).all()


def test_split_rejects_too_few_dates() -> None:
    with pytest.raises(ValueError, match="Not enough dates"):
        compute_split_dates(pd.Series(pd.date_range("2024-01-01", periods=1)), 0.5, 0.4)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"train": {"test_size_ratio": 0.7, "val_size_ratio": 0.4}}, "must be < 1"),
        ({"features": {"lag_days": [0, 7]}}, "positive integers"),
        ({"model": {"type": "lstm"}}, "model.type"),
        ({"features": {"rolling_stats": ["median"]}}, "rolling_stats"),
    ],
)
def test_invalid_config_is_rejected(
    tmp_path: Path, sales: pd.DataFrame, override: dict, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, sales, **override))


def test_metrics() -> None:
    m = compute_metrics([100, 200, 0], [110, 190, 5])
    assert m["mae"] == pytest.approx(25 / 3)
    assert m["mape"] == pytest.approx(7.5)  # zero actual excluded
    assert m["n"] == 3


@pytest.fixture
def artifact(config_path: Path) -> Artifact:
    return train(config_path)


def test_train_saves_loadable_artifact(artifact: Artifact) -> None:
    loaded = Artifact.load(artifact.config["paths"]["model_artifact"])
    assert loaded.feature_columns == artifact.feature_columns
    assert loaded.val_start < loaded.test_start


def test_evaluate_writes_report_with_baseline(artifact: Artifact) -> None:
    report = evaluate(artifact.config["paths"]["model_artifact"])
    on_disk = json.loads(Path(artifact.config["paths"]["metrics_output"]).read_text())
    assert on_disk["one_step"]["model"]["mae"] == pytest.approx(report["one_step"]["model"]["mae"])
    assert set(report["one_step"]["by_store"]) == {"1", "2"}
    bt = report["backtest"]
    assert bt["n_origins"] > 0
    assert len(bt["mae_by_horizon"]["model"]) == bt["horizon"]
    assert "seasonal_naive" in bt


def test_forecast_shape_and_dates(artifact: Artifact, sales: pd.DataFrame) -> None:
    fc = forecast(sales, artifact, horizon=10)
    assert len(fc) == 10 * 2
    assert not fc["predicted_sales"].isna().any()
    expected = pd.date_range(sales["date"].max() + pd.Timedelta(days=1), periods=10)
    assert sorted(fc["date"].unique()) == list(expected)


def test_first_forecast_day_matches_one_step_prediction(artifact: Artifact, sales: pd.DataFrame) -> None:
    """Recursive forecasting and in-sample prediction must build identical features."""
    cutoff = pd.Timestamp("2023-06-01")
    fc = forecast(sales[sales["date"] < cutoff], artifact, horizon=1)
    one_step = predict(sales, artifact)
    expected = one_step[one_step["date"] == cutoff].sort_values("store_id")
    np.testing.assert_allclose(
        fc.sort_values("store_id")["predicted_sales"].to_numpy(),
        expected["predicted_sales"].to_numpy(),
        rtol=1e-6,
    )


def test_forecast_tolerates_unknown_store(artifact: Artifact, sales: pd.DataFrame) -> None:
    extra = sales[sales["store_id"] == "1"].assign(store_id="99")
    fc = forecast(pd.concat([sales, extra]), artifact, horizon=3)
    assert set(fc["store_id"]) == {"1", "2", "99"}


def test_random_forest_model_type(tmp_path: Path, sales: pd.DataFrame) -> None:
    path = write_config(tmp_path, sales, model={"type": "sklearn", "sklearn": {"n_estimators": 10}})
    art = train(path)
    assert len(forecast(sales, art, horizon=3)) == 6


def test_old_artifact_format_is_rejected(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "old.joblib"
    joblib.dump({"model": None}, path)
    with pytest.raises(ValueError, match="incompatible"):
        Artifact.load(path)
