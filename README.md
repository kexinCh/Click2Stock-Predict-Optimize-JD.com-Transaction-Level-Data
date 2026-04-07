# Capstone JD Demand Analysis

This project analyzes JD.com order, click, inventory, and network data to build SKU cluster features, warehouse and region demand panels, and forecasting model comparisons.

## Project Layout

- `data/raw/`: source CSV files used as inputs.
- `data/processed/`: generated CSV datasets and intermediate modeling tables.
- `data/processed/byregion/`: region-level clustering and demand panel outputs.
- `data/database/`: SQLite databases used for click and order aggregation.
- `outputs/exploration/`: exploratory charts and exported summary tables.
- `outputs/models/warehouse_demand/`: warehouse-level demand model outputs.
- `outputs/models/region_demand/`: region-level demand model outputs.
- `outputs/models/baseline_comparison/`: best-model vs baseline comparison outputs.
- `notebooks/`: exploratory and report notebooks.
- `docs/`: reference documents and exported HTML.
- root `*.py`: runnable data-processing and modeling scripts.

## Environment Setup

Create or activate the virtual environment, then install dependencies:

```bash
source /mnt/e/capstonejd/.venv/bin/activate
pip install -r requirements.txt
```

## Main Scripts

- `load_jd_data_to_db.py`: loads click, order, and user CSVs into SQLite.
- `compute_click_order_metrics.py`: builds click-to-order metrics from SQLite.
- `user_click_behavior_metrics.py`: aggregates per-user click behavior.
- `compute_sku_order_type_clusters.py`: creates SKU clusters and feature tables.
- `compute_sku_warehouse_train_test_clusters.py`: creates train/test warehouse cluster panels.
- `compute_cluster_warehouse_daily_demand.py`: exports warehouse daily demand grids.
- `build_compare_demand_models.py`: trains and compares warehouse-level models.
- `compare_best_model_vs_baselines.py`: compares the best model to naive baselines.
- `Byregion/compute_cluster_region_daily_demand.py`: builds region-level demand panels.
- `Byregion/build_compare_region_demand_models.py`: trains and compares region-level models.

All updated scripts now default to the organized `data/` and `outputs/` directories, so you can usually run them without passing file paths.

## Typical Workflow

1. Load raw CSVs into SQLite:

```bash
python load_jd_data_to_db.py
```

2. Build click-derived metrics:

```bash
python compute_click_order_metrics.py
python user_click_behavior_metrics.py
```

3. Create warehouse-level clustering outputs:

```bash
python compute_sku_warehouse_train_test_clusters.py --include-clicks
python build_compare_demand_models.py
python compare_best_model_vs_baselines.py
```

4. Create region-level outputs:

```bash
python Byregion/compute_cluster_region_daily_demand.py
python Byregion/build_compare_region_demand_models.py
```

## Notes

- Notebooks were moved into `notebooks/` and their file references were updated to the new layout.
- If you add new data files, keep raw inputs in `data/raw/` and generated files in `data/processed/` or `outputs/`.
- The repository is not currently a git repository, so file organization changes were applied directly in place.
