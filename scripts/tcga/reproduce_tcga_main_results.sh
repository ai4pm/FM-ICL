#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${FM_ICL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/fmicl:${PYTHONPATH:-}"

echo "[1/5] Generate budget30s run-level metrics"
python -u analysis/tcga/generate_tcga_hparam_budget30s_metrics_table.py

echo "[2/5] Generate budget30s task tables"
python -u analysis/tcga/generate_tcga_budget30s_task_tables.py

echo "[3/5] Generate AFR budget curve"
bash scripts/tcga/combine_budget_curves_afr_pca100.sh

echo "[4/5] Generate DACE-vs-AFR budget curve"
bash scripts/tcga/combine_budget_curves_dace_vs_afr_pca100.sh

echo "[5/5] Generate pca100 radius-scale summaries and plots"
python -u analysis/tcga/evaluate_tcga_radius_scales_pca100.py
python -u analysis/tcga/plot_tcga_radius_scale_curves_pca100_n32_anchors.py

echo "[done] Results are under results/tables and results/figures"
