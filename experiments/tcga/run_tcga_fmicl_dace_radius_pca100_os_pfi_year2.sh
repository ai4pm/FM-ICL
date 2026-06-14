#!/bin/bash
#SBATCH --job-name=tcga_dace_radius_p100
#SBATCH --output=logs/output_tcga_dace_radius_pca100_n32_%a.txt
#SBATCH --error=logs/error_tcga_dace_radius_pca100_n32_%a.txt
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -A isaac-uthsc0057
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --gres=gpu:1
#SBATCH --array=0-33%4

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

export TCGA_RESULT_SUFFIX="${TCGA_RESULT_SUFFIX:-}"
echo "[info] TCGA_RESULT_SUFFIX=${TCGA_RESULT_SUFFIX}"
echo "[info] TCGA FM-ICL (DACE, radius-graph) n_estimators=32: cancers=BRCA,UCEC,KIRC targets=OS,PFI PCA=100"

RADIUS_SCALES=(0.05 0.10 0.15 0.20 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.20 2.40 2.60 2.80 3.00)
PCA_DIMS=(100)
TARGETS=(OS PFI)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PCA_COUNT=${#PCA_DIMS[@]}
TARGET_COUNT=${#TARGETS[@]}
SCALE_STRIDE=$((PCA_COUNT * TARGET_COUNT))
SCALE_INDEX=$((TASK_ID / SCALE_STRIDE))
REMAINDER=$((TASK_ID % SCALE_STRIDE))
TARGET_INDEX=$((REMAINDER / PCA_COUNT))
PCA_INDEX=$((REMAINDER % PCA_COUNT))

RADIUS_SCALE=${RADIUS_SCALES[$SCALE_INDEX]}
PCA_DIM=${PCA_DIMS[$PCA_INDEX]}
TARGET_VALUE=${TARGETS[$TARGET_INDEX]}

echo "[info] task=${TASK_ID} context=radius_by_class radius_scale=${RADIUS_SCALE} target=${TARGET_VALUE} year=2 pca=${PCA_DIM}"

CANCERS=(BRCA UCEC KIRC)
OMICS=(mRNA MicroRNA Methylation)

for cancer in "${CANCERS[@]}"; do
  for omics in "${OMICS[@]}"; do
    python -u \
      fmicl/main_ancestral_continuum_wrapper_tcga.py \
      --ancestry AFR \
      --cancer-type "${cancer}" \
      --expression-type "${omics}" \
      --target "${TARGET_VALUE}" \
      --years 2 \
      --modeling_approach Discrete_Stractification_Enhanced_ICL_AI \
      --tcga-omics-feature-source pca \
      --tcga-pca-dim "${PCA_DIM}" \
      --context-search-mode radius_by_class \
      --distance-metric euclidean \
      --radius-scale "${RADIUS_SCALE}" \
      --tabpfn-n-estimators 32
  done
done
