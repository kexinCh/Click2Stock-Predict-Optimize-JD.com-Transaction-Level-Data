# Optimization Python Code

This folder contains only the Python code needed for the optimization-side work.

It does not duplicate the reports, figures, datasets, prediction CSV files, or notebooks. Those files already exist in the shared capstone project folder.

## Main scripts

- `optimization_prediction_pipeline.py`: original end-to-end forecasting and optimization pipeline.
- `run_candidate_forecast_benchmark.py`: compares candidate prediction files and runs the optimization model for each forecast.
- `run_policy_model_2x2_benchmark.py`: compares:
  - Best prediction + optimization model
  - Best prediction + baseline policy
  - Lag-1 baseline + optimization model
  - Lag-1 baseline + baseline policy

## Python package

Reusable code is under `src/optimization_prediction/`:

- `config.py`: project paths and dates.
- `data_loading.py`: raw JD data loading.
- `parameter_builder.py`: optimization parameter construction.
- `modeling.py`: forecasting utilities.
- `evaluation.py`: forecast and proxy-cost metrics.
- `optimization_solver.py`: LP / receding-horizon optimization model.
- `baseline_policy.py`: rule-based baseline policy.
- `candidate_predictions.py`: prediction CSV harmonization.
- `lag1_baseline.py`: lag-1 previous-day demand baseline.
- `proposed_predictions.py`: proposed-model prediction helpers.
- `reporting.py`: markdown/report utility functions used by the earlier pipeline.

## Expected external files

The scripts expect to be run from a folder that also contains the project data and prediction files, including:

- `Datasets-20260319T015331Z-3-001/`
- `test_predictions.csv`
- `LightGBMB_prediction.csv`
- `LightGBMC_prediction.csv`
- `RandomForestB_prediction.csv`
- `xgboostC_prediction.csv`

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Commands

```powershell
python run_candidate_forecast_benchmark.py
python run_policy_model_2x2_benchmark.py
```
