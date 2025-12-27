# Predicting INR/SGD Rate

## Project
This repository contains code to explore and model the SGD/INR exchange rate using related markets (USD/INR and XAU/INR). It includes feature engineering, quick baselines (logistic / linear regression as well as output plots.

## Data
- Place raw CSVs in `currency_data/` (expected files):
	- `SGD_INR Historical Data.csv`
	- `USD_INR Historical Data.csv`
	- `XAU_INR Historical Data.csv`
- `merge_currency_data(...)` and `feature_engineer_currency_data(...)` produce `merged_currencies.csv` and `processed_currencies.csv` used by the models.

## Requirements
- Python 3.9+ (virtualenv recommended)
- See `requirements.txt` for packages (pandas, scikit-learn, matplotlib, seaborn, numpy)

## Setup
1. Create and activate virtual environment (optional):

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

2. Prepare data and generate processed CSV:

```bash
# edit paths in currencies.py if needed, then run
python3 currencies.py
```

## Usage
- Run models and diagnostics in `ML_models.py`:

```bash
python3 ML_models.py
```

This runs example baselines and saves plots (feature importances, prediction plots) to the repo.

## Notes & best practices
- Beware of data leakage: do NOT include future values (e.g. `Price_Lead1`) as model inputs — they leak the target and produce unrealistically perfect scores. Labels computed with `shift(-1)` are valid targets but must not be used as features.
- Use time-aware train/test splits (no shuffling) for time-series data.
- Add lagged returns, rolling means/std, and interaction terms to improve predictive power.

## Files of interest
- `currencies.py` — data merge / preprocessing / feature engineering
- `ML_models.py` — models, diagnostics, plotting helpers
- `processed_currencies.csv` — cleaned features used by the scripts

## Next steps
- Add cross-validation (time-series split) and hyperparameter tuning
- Try tree ensembles (RandomForest/XGBoost) and richer feature engineering (technical indicators)



