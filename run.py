"""Run full pipeline: train then evaluate (reproducible)."""

import sys
from pathlib import Path

# Run from project root (where run.py lives) so config and data paths resolve
_PROJECT_ROOT = Path(__file__).resolve().parent
if Path.cwd() != _PROJECT_ROOT:
    sys.path.insert(0, str(_PROJECT_ROOT))
    import os
    os.chdir(_PROJECT_ROOT)

from src.train import train
from src.evaluate import evaluate


def main() -> None:
    config_path = Path("config.yaml")
    artifact_path = Path("models/model.joblib")

    print("Training...")
    train(config_path=config_path)
    print(f"Model saved to {artifact_path}")

    print("Evaluating...")
    metrics = evaluate(artifact_path, data_path=None, config_path=config_path)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
