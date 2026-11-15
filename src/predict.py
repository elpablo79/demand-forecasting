"""Inference: one-step predictions on known history and recursive multi-day forecasts."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import (
    UNKNOWN_STORE,
    build_features,
    fill_missing_dates,
    load_raw_data,
    prepare_raw,
)
from src.model import Artifact

logger = logging.getLogger(__name__)

SERIES = "_series"  # one independent history: a store as of a given forecast origin
ORIGIN = "origin"  # first forecast date of a series
STEP = "horizon"  # 1 = first forecast day; 0 = history


def load_grid(data: pd.DataFrame | str | Path, artifact: Artifact) -> pd.DataFrame:
    """Load a CSV path or DataFrame and fill it to a complete daily grid per store."""
    cols = artifact.cols
    date_format = artifact.config["data"]["date_format"]
    if isinstance(data, (str, Path)):
        raw = load_raw_data(data, cols, date_format)
    else:
        raw = prepare_raw(data, cols, date_format)
    return fill_missing_dates(raw, cols)


def mapping_with_unknown(stores: pd.Series, artifact: Artifact) -> dict[str, int]:
    """Store mapping extended with unseen stores (warned about once, encoded as UNKNOWN_STORE)."""
    unknown = sorted(set(stores.astype(str)) - set(artifact.store_mapping))
    if unknown:
        logger.warning("Stores not seen in training, encoded as %d: %s", UNKNOWN_STORE, unknown)
    return {**artifact.store_mapping, **dict.fromkeys(unknown, UNKNOWN_STORE)}


def predict(data: pd.DataFrame | str | Path, artifact: Artifact) -> pd.DataFrame:
    """One-step-ahead predictions for every date in ``data`` that has enough history.

    Each prediction uses the actual values before that date, so this is useful for
    backfilling and diagnostics. Use :func:`forecast` for dates beyond the data.
    """
    cols = artifact.cols
    grid = load_grid(data, artifact)
    feats = build_features(grid, artifact.spec, cols, mapping_with_unknown(grid[cols.store], artifact))
    usable = feats.dropna(subset=artifact.feature_columns)
    if len(usable) < len(feats):
        logger.info(
            "Skipped %d rows without enough history (need %d prior days per store)",
            len(feats) - len(usable),
            artifact.spec.history_days,
        )
    out = usable[[cols.date, cols.store, cols.target]].reset_index(drop=True)
    out[f"predicted_{cols.target}"] = artifact.predict(usable)
    return out


def history_windows(grid: pd.DataFrame, artifact: Artifact, origins: Sequence[pd.Timestamp]) -> pd.DataFrame:
    """For each origin and store, the daily history just before the origin that features need.

    Days missing from ``grid`` (including stores whose data ends early) are NaN.
    """
    cols, spec = artifact.cols, artifact.spec
    stores = sorted(grid[cols.store].unique())
    values = grid.set_index([cols.store, cols.date])[cols.target]
    parts = []
    for origin in origins:
        days = pd.date_range(origin - pd.Timedelta(days=spec.history_days), origin - pd.Timedelta(days=1))
        idx = pd.MultiIndex.from_product([stores, days], names=[cols.store, cols.date])
        part = values.reindex(idx).reset_index()
        part[ORIGIN] = origin
        parts.append(part)
    windows = pd.concat(parts, ignore_index=True)
    windows[SERIES] = windows[ORIGIN].astype(str) + "|" + windows[cols.store]
    windows[STEP] = 0
    return windows


def recursive_forecast(
    windows: pd.DataFrame, artifact: Artifact, horizon: int, mapping: dict[str, int]
) -> pd.DataFrame:
    """Forecast ``horizon`` days for every series in ``windows`` (from :func:`history_windows`).

    Each day's prediction is written back as that day's value so the next day's lag and
    rolling features can use it. All series are forecast together, one day at a time.
    """
    cols, spec = artifact.cols, artifact.spec
    work = windows
    heads = windows.drop_duplicates(SERIES)[[SERIES, cols.store, ORIGIN]]

    for step in range(1, horizon + 1):
        new = heads.copy()
        new[cols.date] = heads[ORIGIN] + pd.Timedelta(days=step - 1)
        new[STEP] = step
        new[cols.target] = np.nan
        work = pd.concat([work, new], ignore_index=True)
        work = work.sort_values([SERIES, cols.date], kind="stable", ignore_index=True)
        feats = build_features(work, spec, cols, mapping, group_col=SERIES)
        today = feats[feats[STEP] == step]
        work.loc[today.index, cols.target] = artifact.predict(today)

    out = work[work[STEP] > 0][[ORIGIN, cols.date, cols.store, STEP, cols.target]]
    out = out.rename(columns={cols.target: f"predicted_{cols.target}"})
    return out.sort_values([ORIGIN, cols.store, STEP], ignore_index=True)


def forecast(data: pd.DataFrame | str | Path, artifact: Artifact, horizon: int) -> pd.DataFrame:
    """Forecast ``horizon`` days after the last date in ``data`` for every store."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    cols = artifact.cols
    grid = load_grid(data, artifact)
    origin = grid[cols.date].max() + pd.Timedelta(days=1)
    windows = history_windows(grid, artifact, [origin])

    incomplete = sorted(windows.loc[windows[cols.target].isna(), cols.store].unique())
    if incomplete:
        logger.warning(
            "Stores missing some of the last %d days (forecasts less reliable): %s",
            artifact.spec.history_days,
            incomplete,
        )

    fc = recursive_forecast(windows, artifact, horizon, mapping_with_unknown(grid[cols.store], artifact))
    return fc.drop(columns=ORIGIN)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Predict demand with a trained model")
    parser.add_argument("input", help="Input CSV with date, store and target history")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument("--artifact", default="models/model.joblib", help="Model artifact path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--horizon", type=int, help="Forecast N days after the input (default: config)")
    mode.add_argument(
        "--in-sample", action="store_true", help="One-step predictions for dates already in the input"
    )
    args = parser.parse_args()

    artifact = Artifact.load(args.artifact)
    if args.in_sample:
        result = predict(args.input, artifact)
    else:
        horizon = args.horizon or artifact.config["evaluation"]["horizon"]
        result = forecast(args.input, artifact, horizon)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(result), args.output)


if __name__ == "__main__":
    main()
