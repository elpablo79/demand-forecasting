"""Training pipeline for demand forecasting."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features import (
    build_features,
    get_feature_columns,
    load_raw_data,
    save_processed,
)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML config with sensible defaults."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def time_based_split(
    df: pd.DataFrame,
    date_column: str,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by time: train = first (1 - test - val), val = next val, test = last test.
    """
    df = df.sort_values(date_column).reset_index(drop=True)
    n = len(df)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - test_size - val_size

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]
    return train_df, val_df, test_df


def get_model(config: dict[str, Any]):
    """Instantiate model from config."""
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "xgboost")

    if model_type == "xgboost":
        import xgboost as xgb

        params = model_cfg.get("xgboost", {})
        return xgb.XGBRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 3),
            random_state=params.get("random_state", 42),
        )
    if model_type == "sklearn":
        from sklearn.ensemble import RandomForestRegressor

        params = model_cfg.get("sklearn", {})
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            random_state=params.get("random_state", 42),
        )
    raise ValueError(f"Unknown model type: {model_type}")


def train(
    config_path: str | Path = "config.yaml",
    data_path: str | Path | None = None,
) -> tuple[Any, list[str], Any, dict[str, Any]]:
    """
    Run full training pipeline: load config and data, build features,
    time-based split, train model, save artifact. Returns (model, feature_columns, store_encoder, config).
    """
    config = load_config(config_path)
    paths = config.get("paths", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    feat_cfg = config.get("features", {})

    data_path = data_path or paths.get("data_raw", "data/raw/sales.csv")
    date_col = data_cfg.get("date_column", "date")
    store_col = data_cfg.get("store_column", "store_id")
    target_col = data_cfg.get("target_column", "sales")
    date_fmt = data_cfg.get("date_format") or None

    df = load_raw_data(data_path, date_column=date_col, date_format=date_fmt)
    df, store_encoder = build_features(
        df,
        date_column=date_col,
        store_column=store_col,
        target_column=target_col,
        lag_days=feat_cfg.get("lag_days", [1, 7, 14, 28]),
        rolling_windows=feat_cfg.get("rolling_windows", [7, 14, 28]),
        include_date_features=feat_cfg.get("include_date_features", True),
        store_encoder=None,
    )

    feature_columns = get_feature_columns(config, store_column=store_col, target_column=target_col)
    # Ensure we only use columns that exist
    feature_columns = [c for c in feature_columns if c in df.columns]

    train_df, val_df, test_df = time_based_split(
        df,
        date_column=date_col,
        test_ratio=train_cfg.get("test_size_ratio", 0.2),
        val_ratio=train_cfg.get("val_size_ratio", 0.1),
        random_state=train_cfg.get("random_state"),
    )

    X_train = train_df[feature_columns].astype(np.float32)
    y_train = train_df[target_col]
    X_val = val_df[feature_columns].astype(np.float32)
    y_val = val_df[target_col]

    model = get_model(config)
    fit_kwargs: dict[str, Any] = {}
    if config.get("model", {}).get("type") == "xgboost":
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train, **fit_kwargs)

    artifact_path = paths.get("model_artifact", "models/model.joblib")
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "store_encoder": store_encoder,
        "config": config,
        "date_column": date_col,
        "store_column": store_col,
        "target_column": target_col,
    }
    joblib.dump(artifact, artifact_path)

    processed_path = paths.get("data_processed")
    if processed_path:
        save_processed(df, processed_path)

    return model, feature_columns, store_encoder, config


def main() -> None:
    """CLI entrypoint."""
    train()


if __name__ == "__main__":
    main()
