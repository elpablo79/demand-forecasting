# Demand Forecasting (Time Series)

Reproducible pipeline for daily demand forecasting per store, with multi-day forecasts,
a seasonal-naive baseline and a rolling backtest.

## Structure

```
demand-forecasting/
├── config.yaml           # Hyperparameters and paths (defaults in src/config.py)
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # + pytest, ruff, mypy
├── pyproject.toml        # Tool configuration
├── run.py                # Train + evaluate
├── data/raw/             # sales.csv (date, store_id, sales)
├── models/               # Saved model artifact (joblib)
├── reports/              # metrics.json, forecasts
├── src/
│   ├── config.py         # Config loading, defaults, validation
│   ├── features.py       # Loading, gap filling, lag/rolling/calendar features
│   ├── model.py          # Model construction, fitting, Artifact
│   ├── train.py          # Date-based split and training
│   ├── evaluate.py       # Test metrics, baseline, rolling backtest
│   └── predict.py        # One-step predictions and recursive forecasts
└── tests/
```

## Setup

Requires Python 3.11 or above.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt   # or requirements.txt for runtime only
```

## Dataset

A CSV with one row per store per day: `date`, `store_id`, `sales` (names configurable in
`config.yaml`). Missing days are filled in as unknown values so lags always refer to
calendar days; duplicate (store, date) rows are rejected.

## Usage

```bash
python run.py                                   # train + evaluate, prints a summary
python -m src.train                             # train only
python -m src.evaluate                          # evaluate the saved model
python -m src.predict data/raw/sales.csv -o reports/forecast.csv               # next 14 days
python -m src.predict data/raw/sales.csv -o reports/forecast.csv --horizon 30  # next 30 days
python -m src.predict data/raw/sales.csv -o reports/insample.csv --in-sample   # 1-day-ahead on known dates
```

All commands accept `--config` / `--artifact` / `--data` where relevant; see `--help`.

## How it works

- **Features** for day *t* use only data before *t*: lags, rolling mean/std/min/max,
  calendar features (day of week/month/year, week of year, weekend) and optional holidays.
- **Split** is on calendar dates: train, then validation (used for XGBoost early stopping),
  then test. The split dates are stored in the artifact so evaluation is always on the
  same period.
- **Forecasting** is recursive: each predicted day is fed back as history for the next.
- **Evaluation** (`reports/metrics.json`) reports, on the test period:
  - `one_step`: 1-day-ahead accuracy using actual history, overall and per store;
  - `backtest`: rolling-origin multi-day forecasts (every `backtest_step_days`, over
    `horizon` days), including MAE for each day ahead;
  - each alongside a seasonal-naive baseline ("same day last week").

The saved model is trained on dates before the validation period only, so its test
metrics are out of sample. For production forecasts you may want to retrain with
`test_size_ratio`/`val_size_ratio` reduced once you are happy with the evaluation.

## Development

```bash
pytest
ruff check . && ruff format --check .
mypy
```

## License

This project is licensed under the [MIT License](LICENSE).