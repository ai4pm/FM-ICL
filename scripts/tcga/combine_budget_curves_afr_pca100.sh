#!/bin/bash
#SBATCH --job-name=plot_budget_afr
#SBATCH --output=logs/output_plot_budget_afr_pca100.txt
#SBATCH --error=logs/error_plot_budget_afr_pca100.txt
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --account=ISAAC-UTHSC0057

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${FM_ICL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SCRIPT="$REPO_DIR/analysis/tcga/plot_budget_curves_pca100.py"
METHOD_DIR="${FM_ICL_METHOD_RESULT_DIR:-/nfs/home/wli66/tabular_transformer_transfer_learning/data_analytics/tcga/budget_curve_method_excels}"
CACHE_CSV="$REPO_DIR/results/tables/results_cache_pca100_afr.csv"
COMBINED_XLSX="$REPO_DIR/results/tables/budget_curves_pca100_afr_combined_raw.xlsx"
OUTPUT_PDF="$REPO_DIR/results/figures/budget_curves_pca100_afr.pdf"
CONDA_ENV="${FM_ICL_CONDA_ENV:-py312}"

METHODS=(ds_tl icl_min elasticnet_min randomforest_min xgboost_min)
TARGETS="OS PFI"
YEAR=2
PCA_DIM=100

echo '[setup] loading anaconda3...'
module load anaconda3/2024.06
if [[ -n "${ANACONDA3_SH:-}" && -f "${ANACONDA3_SH}" ]]; then
    source "${ANACONDA3_SH}"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
mkdir -p "$REPO_DIR/results/tables" "$REPO_DIR/results/figures" "$REPO_DIR/scripts/tcga/logs"

echo '[setup] setting PYTHONPATH...'
export PYTHONPATH=${REPO_DIR}:${REPO_DIR}/fmicl:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo '[run] combining AFR method files and plotting...'
python -u ${SCRIPT} \
    --mode plot-from-method-files \
    --methods "${METHODS[@]}" \
    --targets ${TARGETS} \
    --year ${YEAR} \
    --pca-dim ${PCA_DIM} \
    --method-input-xlsx \
        ${METHOD_DIR}/budget_curves_raw_pca100_ds_tl.xlsx \
        ${METHOD_DIR}/budget_curves_raw_pca100_icl_min.xlsx \
        ${METHOD_DIR}/budget_curves_raw_pca100_elasticnet_min.xlsx \
        ${METHOD_DIR}/budget_curves_raw_pca100_randomforest_min.xlsx \
        ${METHOD_DIR}/budget_curves_raw_pca100_xgboost_min.xlsx \
    --combined-output-xlsx ${COMBINED_XLSX} \
    --cache-csv ${CACHE_CSV} \
    --output-path ${OUTPUT_PDF}

echo "[done] output saved to: ${OUTPUT_PDF}"
