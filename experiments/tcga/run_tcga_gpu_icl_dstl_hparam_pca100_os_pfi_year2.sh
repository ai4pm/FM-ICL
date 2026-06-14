#!/bin/bash
#SBATCH --job-name=tcga_gpu_hparam_p100
#SBATCH --output=logs/output_gpu_hparam_os_pfi_pca100_%A_%a.txt
#SBATCH --error=logs/error_gpu_hparam_os_pfi_pca100_%A_%a.txt
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --account=ISAAC-UTHSC0057
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%3

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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PCA_DIMS=(100)
CANCERS=(KIRC UCEC BRCA)
METHODS=(icl_min icl_mixed ds_tl)
OMICS=(mRNA MicroRNA Methylation)
TARGETS=(OS PFI)
TIME_LIMIT_MINS=(0.0166666667 0.0833333333 0.5 5 60)
TIME_LABELS=(budget1s budget5s budget30s budget5m budget1h)
REPEATS=30

TASK_ID=$((SLURM_ARRAY_TASK_ID + ${HPARAM_TASK_OFFSET:-0}))
CANCER_COUNT=${#CANCERS[@]}
METHOD_COUNT=${#METHODS[@]}

CANCER_INDEX=$((TASK_ID % CANCER_COUNT))
METHOD_INDEX=$(((TASK_ID / CANCER_COUNT) % METHOD_COUNT))
PCA_INDEX=$((TASK_ID / (CANCER_COUNT * METHOD_COUNT)))

CANCER_VALUE=${CANCERS[$CANCER_INDEX]}
METHOD_VALUE=${METHODS[$METHOD_INDEX]}
PCA_DIM=${PCA_DIMS[$PCA_INDEX]}

echo "[info] hparam gpu task=${TASK_ID} pca=${PCA_DIM} method=${METHOD_VALUE} cancer=${CANCER_VALUE}"

OVERWRITE_ARGS=()
if [[ "${HPARAM_OVERWRITE_EXISTING:-1}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite-existing)
fi

for OMICS_VALUE in "${OMICS[@]}"; do
  for TARGET_VALUE in "${TARGETS[@]}"; do
    for BUDGET_IDX in "${!TIME_LIMIT_MINS[@]}"; do
      TIME_LIMIT_MIN=${TIME_LIMIT_MINS[$BUDGET_IDX]}
      RESULT_SUFFIX="_${TIME_LABELS[$BUDGET_IDX]}"
      for SEARCH_RUN in $(seq 0 $((REPEATS - 1))); do
        NUM_CANDIDATES=10000
        EXTRA_ARGS=()
        if [[ "${METHOD_VALUE}" == icl_* ]]; then
          NUM_CANDIDATES=1
          EXTRA_ARGS+=(--tabpfn-n-estimators 32)
        fi
        python -u fmicl/run_tcga_pca100_os_pfi_random_hparam_search.py \
          --method "${METHOD_VALUE}" \
          --cancer-type "${CANCER_VALUE}" \
          --expression-type "${OMICS_VALUE}" \
          --target "${TARGET_VALUE}" \
          --year 2 \
          --pca-dim "${PCA_DIM}" \
          --search-run "${SEARCH_RUN}" \
          --num-candidates "${NUM_CANDIDATES}" \
          --time-limit-min "${TIME_LIMIT_MIN}" \
          --result-suffix "${RESULT_SUFFIX}" \
          --result-target-group OS_PFI \
          "${OVERWRITE_ARGS[@]}" \
          "${EXTRA_ARGS[@]}"
      done
    done
  done
done
