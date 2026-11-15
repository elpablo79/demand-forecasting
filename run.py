"""Run the full pipeline: train, then evaluate."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Resolve config and data paths relative to the project root, wherever this is run from.
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from src.evaluate import evaluate, summarize  # noqa: E402
from src.train import train  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train and evaluate the demand forecasting model")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    artifact = train(config_path=args.config)
    report = evaluate(artifact.config["paths"]["model_artifact"])
    print(summarize(report))


if __name__ == "__main__":
    main()
