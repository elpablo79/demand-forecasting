"""Evaluation metrics and reporting for demand forecasting."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE (with guard for zeros)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE: avoid division by zero
    nonzero = y_true != 0
    if nonzero.sum() == 0:
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

    return {"mae": mae, "rmse": rmse, "mape": mape}


def evaluate(
    model_artifact_path: str | Path,
    data_path: str | Path | None = None,
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """
    Load trained artifact, (optionally) load and prepare test data, predict, compute metrics, save report.
    If data_path is None, expects artifact to have been trained and we load test portion from config paths.
    """
    path = Path(model_artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    artifact = joblib.load(path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    config = artifact.get("config", {})
    target_col = artifact.get("target_column", "sales")
    date_col = artifact.get("date_column", "date")
    store_col = artifact.get("store_column", "store_id")

    if data_path is None:
        data_path = config.get("paths", {}).get("data_raw", "data/raw/sales.csv")
    if output_path is None:
        output_path = config.get("paths", {}).get("metrics_output", "reports/metrics.json")

    from src.features import load_raw_data, build_features

    df = load_raw_data(
        data_path,
        date_column=date_col,
        date_format=config.get("data", {}).get("date_format"),
    )
    df, _ = build_features(
        df,
        date_column=date_col,
        store_column=store_col,
        target_column=target_col,
        lag_days=config.get("features", {}).get("lag_days", [1, 7, 14, 28]),
        rolling_windows=config.get("features", {}).get("rolling_windows", [7, 14, 28]),
        include_date_features=config.get("features", {}).get("include_date_features", True),
        store_encoder=artifact.get("store_encoder"),
    )

    # Reuse same time-based split as train to get test set
    from src.train import time_based_split

    train_cfg = config.get("train", {})
    _, _, test_df = time_based_split(
        df,
        date_column=date_col,
        test_ratio=train_cfg.get("test_size_ratio", 0.2),
        val_ratio=train_cfg.get("val_size_ratio", 0.1),
    )

    X_test = test_df[feature_columns].astype(np.float32)
    y_test = test_df[target_col]
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="models/model.joblib", help="Path to model artifact")
    parser.add_argument("--data", default=None, help="Path to raw data (default from config)")
    parser.add_argument("--output", default=None, help="Path to save metrics JSON")
    args = parser.parse_args()
    evaluate(args.artifact, args.data, output_path=args.output)


if __name__ == "__main__":
    main()
