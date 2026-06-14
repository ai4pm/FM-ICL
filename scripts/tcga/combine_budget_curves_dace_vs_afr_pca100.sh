#!/bin/bash
#SBATCH --job-name=plot_budget_dace_afr
#SBATCH --output=logs/output_plot_budget_dace_vs_afr_pca100.txt
#SBATCH --error=logs/error_plot_budget_dace_vs_afr_pca100.txt
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --account=ISAAC-UTHSC0057

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${FM_ICL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SCRIPT="$REPO_DIR/analysis/tcga/plot_budget_curves_with_dace_pca100.py"
CONDA_ENV="${FM_ICL_CONDA_ENV:-py312}"

module load anaconda3/2024.06
if [[ -n "${ANACONDA3_SH:-}" && -f "${ANACONDA3_SH}" ]]; then
    source "${ANACONDA3_SH}"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "$REPO_DIR/results/tables" "$REPO_DIR/results/figures" "$REPO_DIR/scripts/tcga/logs"
cd "$REPO_DIR"

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/fmicl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

echo "[run] plotting FM-ICL (DACE) vs AFR baselines under time budgets..."
python -u "$SCRIPT"

echo "[done] output saved under $REPO_DIR/results"
