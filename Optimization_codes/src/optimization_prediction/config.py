from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


K_VALUE = 0.0008
H_VALUE = 0.00055
U_VALUE = 0.07

EXPECTED_START_DATE = pd.Timestamp("2018-03-01")
EXPECTED_END_DATE = pd.Timestamp("2018-03-31")
TRAIN_END_DATE = pd.Timestamp("2018-03-24")
TEST_START_DATE = pd.Timestamp("2018-03-25")
TEST_END_DATE = pd.Timestamp("2018-03-31")

INNER_VALIDATION_START = pd.Timestamp("2018-03-18")
INNER_TRAIN_END = pd.Timestamp("2018-03-17")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    dataset_root: Path
    data_dir: Path
    optimization_dir: Path
    train_demand_path: Path
    test_demand_path: Path
    results_dir: Path
    tables_dir: Path
    figures_dir: Path
    report_md: Path
    report_html: Path
    report_pdf: Path
    previous_overall_metrics_path: Path
    professional_pdf_builder: Path


def build_paths(root: Path) -> ProjectPaths:
    dataset_root = root / "Datasets-20260319T015331Z-3-001"
    data_dir = dataset_root / "Datasets" / "JD (Team 121)"
    optimization_dir = data_dir / "Optimization"
    results_dir = root / "results" / "optimization_prediction"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    return ProjectPaths(
        root=root,
        dataset_root=dataset_root,
        data_dir=data_dir,
        optimization_dir=optimization_dir,
        train_demand_path=dataset_root / "sku_warehouse_train_test_clusters_train_warehouse_daily_demand.csv",
        test_demand_path=dataset_root / "sku_warehouse_train_test_clusters_test_warehouse_daily_demand.csv",
        results_dir=results_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        report_md=root / "optimization_model_report.md",
        report_html=root / "optimization_model_report.html",
        report_pdf=root / "optimization_model_report.pdf",
        previous_overall_metrics_path=root / "results" / "optimization_prediction" / "tables" / "overall_metrics.csv",
        professional_pdf_builder=root / "report_pdf" / "build_professional_report.py",
    )
