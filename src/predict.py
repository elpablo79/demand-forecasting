"""Inference pipeline for demand forecasting."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load trained model artifact (model + feature_columns + store_encoder + config)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def predict(
    data: pd.DataFrame | str | Path,
    artifact_path: str | Path = "models/model.joblib",
    *,
    return_dataframe: bool = True,
) -> np.ndarray | pd.DataFrame:
    """
    Run prediction on data. `data` can be a DataFrame (with date, store_id, sales)
    or a path to CSV. Uses artifact's encoder and config for feature building.
    Returns predictions; if return_dataframe=True, returns DataFrame with date, store_id, predicted_sales.
    """
    artifact = load_artifact(artifact_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    config = artifact.get("config", {})
    date_col = artifact.get("date_column", "date")
    store_col = artifact.get("store_column", "store_id")
    target_col = artifact.get("target_column", "sales")

    if isinstance(data, (str, Path)):
        from src.features import load_raw_data

        df = load_raw_data(
            data,
            date_column=date_col,
            date_format=config.get("data", {}).get("date_format"),
        )
    else:
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([date_col, store_col]).reset_index(drop=True)

    from src.features import build_features

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

    X = df[feature_columns].astype(np.float32)
    preds = model.predict(X)

    if not return_dataframe:
        return preds

    out = df[[date_col, store_col]].copy()
    out["predicted_sales"] = preds
    if target_col in df.columns:
        out[target_col] = df[target_col].values
    return out


def main() -> None:
    """CLI entrypoint: read CSV, write predictions CSV."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input CSV path (date, store_id, sales)")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path for predictions")
    parser.add_argument("--artifact", default="models/model.joblib", help="Model artifact path")
    args = parser.parse_args()

    result = predict(args.input, args.artifact, return_dataframe=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
