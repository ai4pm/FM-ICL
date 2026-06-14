#!/usr/bin/env python
"""Plot TCGA pca100 budget curves with budget-invariant ICL one-pass.

The one-pass ICL method does not run hyperparameter search, so its
run-level AUROC values are repeated across each displayed budget before
using the same aggregation and plotting semantics as plot_hparam_budget_curves.py.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_budget_curves_pca100 as budget_plot

REPO_ROOT = SCRIPT_DIR.parents[1]
METHOD_DIR = Path(
    os.environ.get(
        "FM_ICL_METHOD_RESULT_DIR",
        "/nfs/home/wli66/tabular_transformer_transfer_learning/data_analytics/tcga/budget_curve_method_excels",
    )
)
ONEPASS_METRICS = Path(
    os.environ.get(
        "FM_ICL_BUDGET30S_RUN_METRICS",
        REPO_ROOT / "results" / "tables" / "tcga_hparam_budget30s_icl_classical_run_level_metrics.csv",
    )
)
OUT_FIGURE_DIR = Path(os.environ.get("FM_ICL_FIGURE_DIR", REPO_ROOT / "results" / "figures"))
OUT_TABLE_DIR = Path(os.environ.get("FM_ICL_TABLE_DIR", REPO_ROOT / "results" / "tables"))

TIME_LABELS = budget_plot.TIME_LABELS
COMMON_COLUMNS = ["method", "cancer", "omics", "target", "budget", "search_run", "auroc"]

AFR_METHODS = [
    "ds_tl",
    "icl_one_pass_by_class",
    "elasticnet_min",
    "randomforest_min",
    "xgboost_min",
]
MIXED_METHODS = [
    "ds_tl",
    "icl_one_pass_by_class",
    "elasticnet_mixed",
    "randomforest_mixed",
    "xgboost_mixed",
]

METHOD_LABELS = {
    "ds_tl": "TL",
    "icl_one_pass_by_class": "FM-ICL (DACE)",
    "elasticnet_min": "ElasticNet (AFR)",
    "randomforest_min": "RandomForest (AFR)",
    "xgboost_min": "XGBoost (AFR)",
    "elasticnet_mixed": "ElasticNet (Mix)",
    "randomforest_mixed": "RandomForest (Mix)",
    "xgboost_mixed": "XGBoost (Mix)",
}
METHOD_COLORS = {
    "ds_tl": "#e07b39",
    "icl_one_pass_by_class": "#3a86c8",
    "elasticnet_min": "#e63946",
    "elasticnet_mixed": "#e63946",
    "randomforest_min": "#2a9d5c",
    "randomforest_mixed": "#2a9d5c",
    "xgboost_min": "#9b5de5",
    "xgboost_mixed": "#9b5de5",
}
METHOD_MARKERS = {
    "ds_tl": "o",
    "icl_one_pass_by_class": "s",
    "elasticnet_min": "^",
    "elasticnet_mixed": "^",
    "randomforest_min": "D",
    "randomforest_mixed": "D",
    "xgboost_min": "P",
    "xgboost_mixed": "P",
}


def read_method_rows(method: str) -> pd.DataFrame:
    path = METHOD_DIR / f"budget_curves_raw_pca100_{method}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing method CSV: {path}")
    data = pd.read_csv(path)
    missing = sorted(set(COMMON_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    data = data.loc[data["budget"].astype(str).isin(TIME_LABELS), COMMON_COLUMNS].copy()
    data["method"] = method
    data["search_run"] = pd.to_numeric(data["search_run"], errors="raise").astype(int)
    data["auroc"] = pd.to_numeric(data["auroc"], errors="coerce")
    return data


def read_onepass_rows() -> pd.DataFrame:
    if not ONEPASS_METRICS.is_file():
        raise FileNotFoundError(f"Missing one-pass metric source: {ONEPASS_METRICS}")
    data = pd.read_csv(ONEPASS_METRICS)
    missing = sorted(set(COMMON_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"{ONEPASS_METRICS} is missing required columns: {missing}")
    data = data.loc[
        data["method"].astype(str).eq("icl_one_pass_by_class"),
        ["method", "cancer", "omics", "target", "search_run", "auroc"],
    ].copy()
    data["search_run"] = pd.to_numeric(data["search_run"], errors="raise").astype(int)
    data["auroc"] = pd.to_numeric(data["auroc"], errors="coerce")
    if len(data) != 540:
        raise ValueError(f"Expected 540 one-pass rows before expansion, found {len(data)}")
    if data["auroc"].notna().sum() != 540:
        raise ValueError("Expected all 540 one-pass AUROC values to be non-null")

    frames = []
    for budget in TIME_LABELS:
        copied = data.copy()
        copied["budget"] = budget
        frames.append(copied[COMMON_COLUMNS])
    expanded = pd.concat(frames, ignore_index=True)
    if len(expanded) != 2700:
        raise ValueError(f"Expected 2700 one-pass rows after expansion, found {len(expanded)}")
    return expanded


def build_comparison(methods: list[str]) -> pd.DataFrame:
    frames = [read_method_rows(method) for method in methods if method != "icl_one_pass_by_class"]
    frames.append(read_onepass_rows())
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.loc[combined["method"].isin(methods), COMMON_COLUMNS].copy()
    combined["method"] = pd.Categorical(combined["method"], categories=methods, ordered=True)
    combined = combined.sort_values(["method", "budget", "cancer", "omics", "target", "search_run"])
    combined["method"] = combined["method"].astype(str)
    expected_rows = len(methods) * 18 * 30 * len(TIME_LABELS)
    if len(combined) != expected_rows:
        raise ValueError(f"Expected {expected_rows} combined rows for {methods}, found {len(combined)}")
    return combined


def configure_plot_labels() -> None:
    budget_plot.METHOD_LABELS.update(METHOD_LABELS)
    budget_plot.METHOD_COLORS.update(METHOD_COLORS)
    budget_plot.METHOD_MARKERS.update(METHOD_MARKERS)


def write_plot(stem: str, methods: list[str]) -> None:
    OUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_pdf = OUT_FIGURE_DIR / f"{stem}.pdf"
    combined = build_comparison(methods)
    raw_path = OUT_TABLE_DIR / f"{stem}_combined_raw.csv"
    combined.to_csv(raw_path, index=False)
    print(f"[saved] {raw_path}")
    print(f"[check] {stem}_combined_rows={len(combined)}")

    budget_plot.make_figure(combined, output_pdf, methods=methods)

    for suffix in ("mean_auc_summary", "wins_summary", "mean_rank_summary"):
        summary_path = SCRIPT_DIR / f"{stem}_{suffix}.csv"
        summary = pd.read_csv(summary_path)
        if len(summary) != len(methods) * len(TIME_LABELS):
            raise ValueError(
                f"Expected {len(methods) * len(TIME_LABELS)} rows in {summary_path}, "
                f"found {len(summary)}"
            )
        print(f"[check] {summary_path.name}: rows={len(summary)}")


def main() -> None:
    configure_plot_labels()
    write_plot("budget_curves_pca100_dace_vs_afr", AFR_METHODS)


if __name__ == "__main__":
    main()
