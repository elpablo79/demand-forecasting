"""Training pipeline for demand forecasting."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.features import (
    Columns,
    FeatureSpec,
    build_features,
    complete_rows,
    feature_columns,
    fit_store_mapping,
    load_dataset,
    save_processed,
)
from src.model import Artifact, fit_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitDates:
    """Train is ``date < val_start``, val is ``val_start <= date < test_start``, test is the rest."""

    val_start: pd.Timestamp
    test_start: pd.Timestamp


def compute_split_dates(dates: pd.Series, test_ratio: float, val_ratio: float) -> SplitDates:
    """Split on unique calendar dates so no date is shared between train, val and test."""
    unique = pd.Series(dates.unique()).sort_values(ignore_index=True)
    n = len(unique)
    n_test = max(1, int(n * test_ratio))
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val
    if n_train < 1:
        raise ValueError(f"Not enough dates ({n}) for the requested split ratios")
    return SplitDates(val_start=unique[n_train], test_start=unique[n_train + n_val])


def split_by_dates(
    df: pd.DataFrame, date_col: str, split: SplitDates
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df[date_col]
    return (
        df[d < split.val_start],
        df[(d >= split.val_start) & (d < split.test_start)],
        df[d >= split.test_start],
    )


def train(config_path: str | Path = "config.yaml", data_path: str | Path | None = None) -> Artifact:
    """Load data, build features, fit on the train split and save the artifact."""
    config = load_config(config_path)
    cols = Columns.from_config(config)
    spec = FeatureSpec.from_config(config)
    paths = config["paths"]

    grid = load_dataset(config, data_path)
    split = compute_split_dates(
        grid[cols.date], config["train"]["test_size_ratio"], config["train"]["val_size_ratio"]
    )

    # Fit the store encoding on training dates only.
    store_mapping = fit_store_mapping(grid.loc[grid[cols.date] < split.val_start, cols.store])

    # Features only look at the past, so building them before splitting does not leak.
    feats = build_features(grid, spec, cols, store_mapping)
    feat_cols = feature_columns(spec, cols)
    usable = complete_rows(feats, feat_cols, cols.target)
    train_df, val_df, test_df = split_by_dates(usable, cols.date, split)
    logger.info(
        "Split rows: train=%d val=%d test=%d (val from %s, test from %s)",
        len(train_df),
        len(val_df),
        len(test_df),
        split.val_start.date(),
        split.test_start.date(),
    )
    if train_df.empty:
        raise ValueError("No training rows left after removing each store's warm-up period")

    model = fit_model(config, train_df, val_df, feat_cols, cols.target)
    best = getattr(model, "best_iteration", None)
    if best is not None:
        logger.info("Early stopping chose %d trees", best + 1)

    artifact = Artifact(
        model=model,
        feature_columns=feat_cols,
        store_mapping=store_mapping,
        config=config,
        val_start=split.val_start,
        test_start=split.test_start,
    )
    artifact.save(paths["model_artifact"])
    logger.info("Model saved to %s", paths["model_artifact"])

    if paths.get("data_processed"):
        save_processed(feats, paths["data_processed"])

    return artifact


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train the demand forecasting model")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--data", default=None, help="Raw data CSV (default from config)")
    args = parser.parse_args()
    train(args.config, args.data)


if __name__ == "__main__":
    main()
