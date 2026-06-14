#!/bin/bash
#SBATCH --job-name=tcga_dace_mst_p100
#SBATCH --output=logs/output_tcga_dace_mst_pca100_n32_%A_%a.txt
#SBATCH --error=logs/error_tcga_dace_mst_pca100_n32_%A_%a.txt
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -A isaac-uthsc0057
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --gres=gpu:1
#SBATCH --array=0-17%4

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FM_ICL_REPO_ROOT="${FM_ICL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
mkdir -p "${FM_ICL_REPO_ROOT}/experiments/tcga/logs"
module load anaconda3/2024.06
if [[ -n "${ANACONDA3_SH:-}" && -f "${ANACONDA3_SH}" ]]; then
  source "${ANACONDA3_SH}"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FM_ICL_CONDA_ENV:-py312}"
cd "${FM_ICL_REPO_ROOT}"
export PYTHONPATH="${FM_ICL_REPO_ROOT}:${FM_ICL_REPO_ROOT}/fmicl:${PYTHONPATH:-}"

export TCGA_RESULT_SUFFIX="_n32"

CANCERS=(BRCA UCEC KIRC)
OMICS=(mRNA MicroRNA Methylation)
TARGETS=(OS PFI)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
OMICS_COUNT=${#OMICS[@]}
TARGET_COUNT=${#TARGETS[@]}

CANCER_INDEX=$((TASK_ID / (OMICS_COUNT * TARGET_COUNT)))
REMAINDER=$((TASK_ID % (OMICS_COUNT * TARGET_COUNT)))
OMICS_INDEX=$((REMAINDER / TARGET_COUNT))
TARGET_INDEX=$((REMAINDER % TARGET_COUNT))

CANCER_VALUE=${CANCERS[$CANCER_INDEX]}
OMICS_VALUE=${OMICS[$OMICS_INDEX]}
TARGET_VALUE=${TARGETS[$TARGET_INDEX]}

echo "[info] TCGA one-pass fold0 n32 overwrite task=${TASK_ID} cancer=${CANCER_VALUE} omics=${OMICS_VALUE} target=${TARGET_VALUE} pca=100"
echo "[info] TCGA_RESULT_SUFFIX=${TCGA_RESULT_SUFFIX}"

python -u \
  fmicl/main_ancestral_continuum_wrapper_tcga.py \
  --ancestry AFR \
  --cancer-type "${CANCER_VALUE}" \
  --expression-type "${OMICS_VALUE}" \
  --target "${TARGET_VALUE}" \
  --years 2 \
  --modeling_approach Discrete_Stractification_Enhanced_ICL_AI \
  --tcga-omics-feature-source pca \
  --tcga-pca-dim 100 \
  --context-search-mode one_pass_by_class \
  --tabpfn-n-estimators 32 \
  --fold-index-only 0 \
  --overwrite-results
