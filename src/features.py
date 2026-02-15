"""Feature engineering for demand forecasting."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_raw_data(
    path: str | Path,
    date_column: str = "date",
    date_format: str | None = None,
) -> pd.DataFrame:
    """Load raw sales data and ensure correct dtypes and order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found. Got: {list(df.columns)}")

    df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    df = df.sort_values([date_column, "store_id"]).reset_index(drop=True)
    return df


def build_lag_features(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """Add lag features per group (e.g. per store)."""
    out = df.copy()
    for lag in lags:
        out[f"{value_col}_lag_{lag}"] = out.groupby(group_col)[value_col].shift(lag)
    return out


def build_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    windows: list[int],
) -> pd.DataFrame:
    """Add rolling mean features per group."""
    out = df.copy()
    for w in windows:
        out[f"{value_col}_rolling_mean_{w}"] = (
            out.groupby(group_col)[value_col]
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        )
    return out


def build_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Add date-based features."""
    out = df.copy()
    d = pd.to_datetime(out[date_column])
    out["day_of_week"] = d.dt.dayofweek
    out["month"] = d.dt.month
    out["year"] = d.dt.year
    out["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    return out


def encode_store(
    df: pd.DataFrame,
    store_column: str,
    encoder: Any | None = None,
) -> tuple[pd.DataFrame, Any]:
    """
    Encode store_id to integer. If encoder is provided, use it (transform);
    otherwise fit a new one. Returns (dataframe with _encoded column, encoder).
    """
    from sklearn.preprocessing import LabelEncoder

    le = encoder if encoder is not None else LabelEncoder()
    col = df[store_column].astype(str)
    if encoder is None:
        encoded = le.fit_transform(col)
    else:
        # Unseen store_ids get -1
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        encoded = col.map(lambda x: mapping.get(x, -1)).values.astype(np.int64)
    out = df.copy()
    out[f"{store_column}_encoded"] = encoded
    return out, le


def build_features(
    df: pd.DataFrame,
    *,
    date_column: str = "date",
    store_column: str = "store_id",
    target_column: str = "sales",
    lag_days: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    include_date_features: bool = True,
    store_encoder: Any | None = None,
) -> tuple[pd.DataFrame, Any]:
    """
    Full feature pipeline: lags, rolling, date features, store encoding.
    Drops rows where lag/rolling create NaNs (initial period per store).
    Returns (feature dataframe, store encoder used or fitted).
    """
    lag_days = lag_days or [1, 7, 14, 28]
    rolling_windows = rolling_windows or [7, 14, 28]

    out = build_lag_features(df, store_column, target_column, lag_days)
    out = build_rolling_features(out, store_column, target_column, rolling_windows)

    if include_date_features:
        out = build_date_features(out, date_column)

    out, store_encoder = encode_store(out, store_column, encoder=store_encoder)

    # Drop rows with NaN from lags (first max(lag) days per store)
    out = out.dropna(subset=[f"{target_column}_lag_{lag_days[0]}"]).reset_index(drop=True)

    return out, store_encoder


def get_feature_columns(
    config: dict[str, Any],
    store_column: str = "store_id",
    target_column: str = "sales",
) -> list[str]:
    """Return list of feature column names used for training (no date, no target)."""
    lag_days = config.get("features", {}).get("lag_days", [1, 7, 14, 28])
    rolling_windows = config.get("features", {}).get("rolling_windows", [7, 14, 28])
    include_date = config.get("features", {}).get("include_date_features", True)

    cols = (
        [f"{target_column}_lag_{l}" for l in lag_days]
        + [f"{target_column}_rolling_mean_{w}" for w in rolling_windows]
        + [f"{store_column}_encoded"]
    )
    if include_date:
        cols += ["day_of_week", "month", "year", "is_weekend"]
    return cols


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Save processed dataframe (e.g. Parquet) for reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
