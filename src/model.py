"""Model construction, fitting and the saved artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features import Columns, FeatureSpec

ARTIFACT_VERSION = 2


def get_model(config: dict[str, Any], early_stopping: bool) -> Any:
    """Instantiate the regressor described in config."""
    model_cfg = config["model"]
    if model_cfg["type"] == "xgboost":
        import xgboost as xgb

        params = dict(model_cfg["xgboost"])
        if early_stopping:
            params["early_stopping_rounds"] = config["train"]["early_stopping_rounds"]
        return xgb.XGBRegressor(**params)

    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**model_cfg["sklearn"])


def to_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return df[feature_cols].astype(np.float32)


def fit_model(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> Any:
    """Fit the model; XGBoost uses the validation set for early stopping."""
    use_es = config["model"]["type"] == "xgboost" and len(val_df) > 0
    model = get_model(config, early_stopping=use_es)
    X_train, y_train = to_matrix(train_df, feature_cols), train_df[target]
    if use_es:
        model.fit(
            X_train,
            y_train,
            eval_set=[(to_matrix(val_df, feature_cols), val_df[target])],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)
    return model


@dataclass
class Artifact:
    """Everything needed to reproduce features and predictions after training."""

    model: Any
    feature_columns: list[str]
    store_mapping: dict[str, int]
    config: dict[str, Any]
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    version: int = ARTIFACT_VERSION

    @property
    def cols(self) -> Columns:
        return Columns.from_config(self.config)

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec.from_config(self.config)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(to_matrix(features, self.feature_columns)), dtype=float)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> Artifact:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        artifact = joblib.load(path)
        if not isinstance(artifact, Artifact) or artifact.version != ARTIFACT_VERSION:
            raise ValueError(f"{path} was saved by an incompatible version; retrain the model")
        return artifact
