"""Data loading and feature engineering for demand forecasting.

All features for date ``t`` are computed only from target values strictly before
``t``, so they are safe to use for forecasting and when building them on the full
history before a time-based split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

UNKNOWN_STORE = -1


@dataclass(frozen=True)
class Columns:
    date: str = "date"
    store: str = "store_id"
    target: str = "sales"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Columns:
        d = config["data"]
        return cls(date=d["date_column"], store=d["store_column"], target=d["target_column"])


@dataclass(frozen=True)
class FeatureSpec:
    lag_days: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    rolling_stats: tuple[str, ...]
    include_date_features: bool
    holidays: tuple[pd.Timestamp, ...]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeatureSpec:
        f = config["features"]
        return cls(
            lag_days=tuple(sorted(f["lag_days"])),
            rolling_windows=tuple(sorted(f["rolling_windows"])),
            rolling_stats=tuple(f["rolling_stats"]),
            include_date_features=bool(f["include_date_features"]),
            holidays=tuple(pd.to_datetime(h).normalize() for h in f["holidays"]),
        )

    @property
    def history_days(self) -> int:
        """Days of past data needed to compute every feature for one date."""
        return max(*self.lag_days, *self.rolling_windows)


DATE_FEATURES = ["day_of_week", "day_of_month", "month", "day_of_year", "week_of_year", "is_weekend"]


def feature_columns(spec: FeatureSpec, cols: Columns) -> list[str]:
    """Names of the model input columns, in a fixed order."""
    names = [f"{cols.target}_lag_{lag}" for lag in spec.lag_days]
    names += [
        f"{cols.target}_rolling_{stat}_{w}" for w in spec.rolling_windows for stat in spec.rolling_stats
    ]
    names.append(f"{cols.store}_encoded")
    if spec.include_date_features:
        names += DATE_FEATURES
    if spec.holidays:
        names.append("is_holiday")
    return names


def load_raw_data(path: str | Path, cols: Columns, date_format: str | None = None) -> pd.DataFrame:
    """Load raw sales data, validate it and return it sorted by (store, date)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return prepare_raw(df, cols, date_format)


def prepare_raw(df: pd.DataFrame, cols: Columns, date_format: str | None = None) -> pd.DataFrame:
    """Validate columns, parse dates, check for duplicates and sort by (store, date)."""
    missing = {cols.date, cols.store, cols.target} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}. Got: {list(df.columns)}")

    df = df.copy()
    df[cols.date] = pd.to_datetime(df[cols.date], format=date_format).dt.normalize()
    df[cols.store] = df[cols.store].astype(str)
    df[cols.target] = pd.to_numeric(df[cols.target], errors="raise").astype(float)

    dupes = df.duplicated([cols.store, cols.date])
    if dupes.any():
        raise ValueError(f"{int(dupes.sum())} duplicate ({cols.store}, {cols.date}) rows in data")

    return df.sort_values([cols.store, cols.date], kind="stable").reset_index(drop=True)


def fill_missing_dates(df: pd.DataFrame, cols: Columns) -> pd.DataFrame:
    """Reindex each store to a complete daily calendar, leaving missing targets as NaN.

    Lags are computed with row shifts, so without this a gap in the data would make
    "lag 7" silently point to a different day.
    """
    parts = []
    n_added = 0
    for store, g in df.groupby(cols.store, sort=True):
        full = pd.date_range(g[cols.date].min(), g[cols.date].max(), freq="D", name=cols.date)
        g = g.set_index(cols.date).reindex(full)
        n_added += int(g[cols.store].isna().sum())
        g[cols.store] = store
        parts.append(g.reset_index())
    if n_added:
        logger.warning("Filled %d missing (store, date) rows with NaN %s", n_added, cols.target)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values([cols.store, cols.date], kind="stable").reset_index(drop=True)


def fit_store_mapping(stores: pd.Series) -> dict[str, int]:
    """Map each store id (as string) to a stable integer code."""
    return {s: i for i, s in enumerate(sorted(stores.astype(str).unique()))}


def encode_store(stores: pd.Series, mapping: dict[str, int]) -> pd.Series:
    """Encode store ids with ``mapping``; unknown stores get UNKNOWN_STORE with a warning."""
    encoded = stores.astype(str).map(mapping)
    unknown = encoded.isna()
    if unknown.any():
        logger.warning(
            "Stores not seen in training, encoded as %d: %s",
            UNKNOWN_STORE,
            sorted(stores[unknown].astype(str).unique()),
        )
    return encoded.fillna(UNKNOWN_STORE).astype(int)


def build_features(
    df: pd.DataFrame,
    spec: FeatureSpec,
    cols: Columns,
    store_mapping: dict[str, int],
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add all features. ``df`` must be a complete daily grid sorted by (group, date).

    Lags and rolling stats are computed within ``group_col`` (default: the store
    column); forecasting uses a different grouping to run many histories at once.
    Rows are never dropped here; use :func:`complete_rows` to select training rows.
    """
    group_col = group_col or cols.store
    # Collect new columns and attach them in one concat (much faster than one insert each).
    new: dict[str, Any] = {}
    by_group = df.groupby(group_col, sort=False)[cols.target]

    for lag in spec.lag_days:
        new[f"{cols.target}_lag_{lag}"] = by_group.shift(lag)

    shifted = by_group.shift(1).groupby(df[group_col], sort=False)
    stats = list(spec.rolling_stats)
    for w in spec.rolling_windows:
        rolled = shifted.rolling(window=w, min_periods=1).agg(stats)
        rolled = rolled.reset_index(level=0, drop=True)
        for stat in stats:
            new[f"{cols.target}_rolling_{stat}_{w}"] = rolled[stat]

    new[f"{cols.store}_encoded"] = encode_store(df[cols.store], store_mapping)

    d = df[cols.date]
    if spec.include_date_features:
        new["day_of_week"] = d.dt.dayofweek
        new["day_of_month"] = d.dt.day
        new["month"] = d.dt.month
        new["day_of_year"] = d.dt.dayofyear
        new["week_of_year"] = d.dt.isocalendar().week.astype(int)
        new["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    if spec.holidays:
        new["is_holiday"] = d.isin(list(spec.holidays)).astype(int)

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def complete_rows(df: pd.DataFrame, feature_cols: list[str], target: str) -> pd.DataFrame:
    """Rows with a known target and every feature available (drops each store's warm-up period)."""
    return df.dropna(subset=[*feature_cols, target]).reset_index(drop=True)


def load_dataset(config: dict[str, Any], data_path: str | Path | None = None) -> pd.DataFrame:
    """Load raw data from config and fill it to a complete daily grid."""
    cols = Columns.from_config(config)
    path = data_path or config["paths"]["data_raw"]
    raw = load_raw_data(path, cols, config["data"]["date_format"])
    return fill_missing_dates(raw, cols)


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Save processed dataframe as Parquet for reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
