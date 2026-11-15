"""Evaluation: test-set metrics, a seasonal-naive baseline and a rolling multi-day backtest."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import build_features, complete_rows, load_dataset
from src.model import Artifact
from src.predict import ORIGIN, STEP, history_windows, mapping_with_unknown, recursive_forecast

logger = logging.getLogger(__name__)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """MAE, RMSE and MAPE (MAPE ignores zero actuals; NaN if all are zero)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": mape,
        "n": len(y_true),
    }


def evaluate_one_step(grid: pd.DataFrame, artifact: Artifact) -> dict[str, Any]:
    """One-day-ahead metrics on the test period, using actual history for every feature."""
    cols = artifact.cols
    period = artifact.config["evaluation"]["seasonal_period"]
    mapping = mapping_with_unknown(grid[cols.store], artifact)

    feats = build_features(grid, artifact.spec, cols, mapping)
    feats["_naive"] = feats.groupby(cols.store)[cols.target].shift(period)
    test = complete_rows(feats, [*artifact.feature_columns, "_naive"], cols.target)
    test = test[test[cols.date] >= artifact.test_start]
    if test.empty:
        raise ValueError("No test rows to evaluate")
    test = test.assign(_pred=artifact.predict(test))

    by_store = {
        str(store): {
            "model": compute_metrics(g[cols.target], g["_pred"]),
            "seasonal_naive": compute_metrics(g[cols.target], g["_naive"]),
        }
        for store, g in test.groupby(cols.store)
    }
    return {
        "model": compute_metrics(test[cols.target], test["_pred"]),
        "seasonal_naive": compute_metrics(test[cols.target], test["_naive"]),
        "by_store": by_store,
    }


def backtest(grid: pd.DataFrame, artifact: Artifact) -> dict[str, Any]:
    """Rolling-origin backtest over the test period with recursive multi-day forecasts.

    The model was trained only on dates before the validation period, so every origin
    in the test period is out of sample. The baseline repeats the last observed
    ``seasonal_period`` days before each origin.
    """
    cols = artifact.cols
    ev = artifact.config["evaluation"]
    horizon, step, period = ev["horizon"], ev["backtest_step_days"], ev["seasonal_period"]
    last_date = grid[cols.date].max()

    origins = pd.date_range(artifact.test_start, last_date - pd.Timedelta(days=horizon - 1), freq=f"{step}D")
    if len(origins) == 0:
        logger.warning("Test period shorter than horizon (%d days); skipping backtest", horizon)
        return {"horizon": horizon, "n_origins": 0}

    windows = history_windows(grid, artifact, list(origins))
    fc = recursive_forecast(windows, artifact, horizon, mapping_with_unknown(grid[cols.store], artifact))

    values = grid.set_index([cols.store, cols.date])[cols.target]
    naive_dates = fc[ORIGIN] - pd.Timedelta(days=period) + pd.to_timedelta((fc[STEP] - 1) % period, unit="D")
    res = pd.DataFrame(
        {
            STEP: fc[STEP],
            "actual": values.reindex(pd.MultiIndex.from_arrays([fc[cols.store], fc[cols.date]])).to_numpy(),
            "model": fc[f"predicted_{cols.target}"],
            "naive": values.reindex(pd.MultiIndex.from_arrays([fc[cols.store], naive_dates])).to_numpy(),
        }
    ).dropna()

    abs_err = res.assign(model=(res.actual - res.model).abs(), naive=(res.actual - res.naive).abs())
    mae_by_h = abs_err.groupby(STEP)[["model", "naive"]].mean().round(3)
    return {
        "horizon": horizon,
        "n_origins": len(origins),
        "model": compute_metrics(res["actual"], res["model"]),
        "seasonal_naive": compute_metrics(res["actual"], res["naive"]),
        "mae_by_horizon": {
            "model": mae_by_h["model"].tolist(),
            "seasonal_naive": mae_by_h["naive"].tolist(),
        },
    }


def evaluate(
    artifact_path: str | Path,
    data_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a trained artifact on its test period and write a JSON report."""
    artifact = Artifact.load(artifact_path)
    grid = load_dataset(artifact.config, data_path)

    report = {
        "test_start": str(artifact.test_start.date()),
        "test_end": str(grid[artifact.cols.date].max().date()),
        "one_step": evaluate_one_step(grid, artifact),
        "backtest": backtest(grid, artifact),
    }

    output_path = Path(output_path or artifact.config["paths"]["metrics_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Metrics written to %s", output_path)
    return report


def summarize(report: dict[str, Any]) -> str:
    """Short human-readable summary of an evaluation report."""
    lines = [f"Test period {report['test_start']} to {report['test_end']}"]
    sections = [("One-step (1 day ahead)", report["one_step"])]
    bt = report["backtest"]
    if bt.get("n_origins"):
        sections.append((f"Backtest ({bt['horizon']}-day horizon, {bt['n_origins']} origins)", bt))
    for title, sec in sections:
        m, b = sec["model"], sec["seasonal_naive"]
        lines.append(f"  {title}:")
        lines.append(f"    model          MAE {m['mae']:7.2f}  RMSE {m['rmse']:7.2f}  MAPE {m['mape']:6.2f}%")
        lines.append(f"    seasonal naive MAE {b['mae']:7.2f}  RMSE {b['rmse']:7.2f}  MAPE {b['mape']:6.2f}%")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--artifact", default="models/model.joblib", help="Path to model artifact")
    parser.add_argument("--data", default=None, help="Raw data CSV (default from the artifact's config)")
    parser.add_argument("--output", default=None, help="Metrics JSON path (default from config)")
    args = parser.parse_args()
    print(summarize(evaluate(args.artifact, args.data, args.output)))


if __name__ == "__main__":
    main()
