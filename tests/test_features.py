from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULTS
from src.features import (
    UNKNOWN_STORE,
    Columns,
    FeatureSpec,
    build_features,
    complete_rows,
    feature_columns,
    fill_missing_dates,
    fit_store_mapping,
    prepare_raw,
)

SPEC = FeatureSpec.from_config(DEFAULTS)


def featurize(df: pd.DataFrame, cols: Columns) -> pd.DataFrame:
    grid = fill_missing_dates(prepare_raw(df, cols), cols)
    return build_features(grid, SPEC, cols, fit_store_mapping(grid[cols.store]))


def test_features_do_not_use_same_day_target(sales: pd.DataFrame, cols: Columns) -> None:
    feat_cols = feature_columns(SPEC, cols)
    base = featurize(sales, cols)
    target_date = pd.Timestamp("2023-04-01")
    changed = sales.copy()
    changed.loc[changed["date"] == target_date, "sales"] += 1000
    after = featurize(changed, cols)

    on_day = base["date"] == target_date
    pd.testing.assert_frame_equal(base.loc[on_day, feat_cols], after.loc[on_day, feat_cols])
    # ...but the next day's features do see it.
    next_day = base["date"] == target_date + pd.Timedelta(days=1)
    assert not base.loc[next_day, feat_cols].equals(after.loc[next_day, feat_cols])


def test_lags_count_calendar_days_when_dates_are_missing(sales: pd.DataFrame, cols: Columns) -> None:
    gappy = sales[sales["date"] != pd.Timestamp("2023-03-10")]
    feats = featurize(gappy, cols)
    s1 = feats[feats[cols.store] == "1"].set_index(cols.date)
    day = pd.Timestamp("2023-03-15")
    assert s1.loc[day, "sales_lag_7"] == s1.loc[day - pd.Timedelta(days=7), "sales"]
    assert np.isnan(s1.loc[pd.Timestamp("2023-03-11"), "sales_lag_1"])


def test_complete_rows_drop_whole_warmup_period(sales: pd.DataFrame, cols: Columns) -> None:
    feat_cols = feature_columns(SPEC, cols)
    usable = complete_rows(featurize(sales, cols), feat_cols, cols.target)
    assert not usable[feat_cols].isna().any().any()
    first = usable.groupby(cols.store)[cols.date].min()
    assert (first == sales["date"].min() + pd.Timedelta(days=max(SPEC.lag_days))).all()


def test_lag_order_in_config_does_not_matter(cols: Columns) -> None:
    cfg = {**DEFAULTS, "features": {**DEFAULTS["features"], "lag_days": [28, 1, 7]}}
    assert FeatureSpec.from_config(cfg).lag_days == (1, 7, 28)


def test_custom_column_names_work(sales: pd.DataFrame) -> None:
    cols = Columns(date="day", store="shop", target="units")
    renamed = sales.rename(columns={"date": "day", "store_id": "shop", "sales": "units"})
    feats = featurize(renamed, cols)
    assert {"units_lag_1", "shop_encoded"} <= set(feats.columns)


def test_duplicate_rows_are_rejected(sales: pd.DataFrame, cols: Columns) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        prepare_raw(pd.concat([sales, sales.head(1)]), cols)


def test_missing_columns_are_reported(sales: pd.DataFrame, cols: Columns) -> None:
    with pytest.raises(ValueError, match="store_id"):
        prepare_raw(sales.drop(columns="store_id"), cols)


def test_unknown_store_is_encoded_and_warned(
    sales: pd.DataFrame, cols: Columns, caplog: pytest.LogCaptureFixture
) -> None:
    grid = fill_missing_dates(prepare_raw(sales, cols), cols)
    feats = build_features(grid, SPEC, cols, {"1": 0})
    assert (feats.loc[feats[cols.store] == "2", "store_id_encoded"] == UNKNOWN_STORE).all()
    assert "not seen in training" in caplog.text


def test_holiday_feature(sales: pd.DataFrame, cols: Columns) -> None:
    cfg = {**DEFAULTS, "features": {**DEFAULTS["features"], "holidays": ["2023-01-06"]}}
    spec = FeatureSpec.from_config(cfg)
    grid = fill_missing_dates(prepare_raw(sales, cols), cols)
    feats = build_features(grid, spec, cols, fit_store_mapping(grid[cols.store]))
    assert "is_holiday" in feature_columns(spec, cols)
    assert set(feats.loc[feats["is_holiday"] == 1, cols.date]) == {pd.Timestamp("2023-01-06")}
