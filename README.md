# Demand Forecasting (Time Series)

Reproducible pipeline for demand forecasting with time series.

## Structure

```
demand-forecasting/
├── config.yaml       # Hyperparameters and paths
├── requirements.txt
├── run.py            # Runs train + evaluate
├── data/
│   └── raw/          # sales.csv (date, store_id, sales)
├── models/           # Saved model (joblib)
├── reports/          # metrics.json
└── src/
    ├── features.py   # Feature engineering (lag, rolling, date, store encoding)
    ├── train.py      # Training and artifact saving
    ├── evaluate.py   # Metrics (MAE, RMSE, MAPE) and reporting
    └── predict.py    # Inference
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

The CSV must have columns: `date`, `store_id`, `sales`.  
Place your data in `data/raw/sales.csv` (path can be changed in `config.yaml`).

## Usage

**Full pipeline (train + evaluate):**
```bash
python run.py
```

**Training only:**
```bash
python -m src.train
```

**Evaluation only:**
```bash
python -m src.evaluate --artifact models/model.joblib --output reports/metrics.json
```

**Predictions on a CSV:**
```bash
python -m src.predict data/raw/sales.csv -o reports/predictions.csv
```

## Configuration

Edit `config.yaml` to set:
- Data and model paths
- Lag and rolling windows
- Model type (`xgboost` or `sklearn`) and hyperparameters
- Train/val/test ratios (time-based split)

## Reproducibility

- Time-based split (no shuffle): latest periods are used for test and validation.
- `random_state` fixed in config for model and split.
- Artifact saved with feature columns and encoder for consistent inference.
