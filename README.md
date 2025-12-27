# Predicting INR/SGD Rate

## Project
This repository implements a methodology inspired by E. Sarmas et al., originally applied to USD/EUR exchange rates, and adapts it to the SGD/INR exchange rate. It includes feature engineering, quick baselines logistic / linear regression as well as output plots.

## Data
- Raw CSVs in `currency_data/`:
	- `SGD_INR Historical Data.csv`
	- `USD_INR Historical Data.csv`
	- `XAU_INR Historical Data.csv`
- Utilising the functions in `getStock.py` to get `merged_currencies.csv` and `processed_currencies.csv` used by the models.

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
python3 ML_models.py
```

## Usage
- Run models and diagnostics in `ML_models.py`:

```bash
python3 ML_models.py
```

This runs example baselines and saves plots (feature importances, prediction plots) to the repo.

## Notes 
- Used time-aware train/test splits (no shuffling) for time-series data.


## Files and Folders of interest
- `Output_images` — folder contains saved output plots of linear and logistic regression
- `ML_models.py` — to run models, diagnostics, plotting helpers
- `processed_currencies.csv` — cleaned features used by the scripts

## Results
- Logistic regression achieved an accuracy of 57% using Youden’s method for threshold selection.
- This represents a slight improvement over the 53% average accuracy reported for USD/EUR prediction in the reference publication.

## Citations
- E. Sarmas et al., "Comparison of Machine Learning Classifiers for Exchange Rate Trend Forecasting," 2022 13th International Conference on Information, Intelligence, Systems & Applications (IISA), Corfu, Greece, 2022, pp. 1-7, doi: 10.1109/IISA56318.2022.9904380.





