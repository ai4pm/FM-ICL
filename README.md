# FM-ICL

This repository contains the pca100 TCGA experiments for **FM-ICL**,
short for foundation-model in-context learning.

The main TCGA task grid is:

- cancers: `BRCA`, `UCEC`, `KIRC`
- endpoints: `OS`, `PFI`
- omics: `mRNA`, `MicroRNA`, `Methylation`
- feature dimension: `pca100`

That gives 18 tasks.

## Method Names

Public-facing names use `AFR` rather than `min`.

| Paper/repo label | Internal method key | Current modeling approach |
| --- | --- | --- |
| `FM-ICL (AFR)` | `icl_min` | `Discrete_Stractification_ICL_AI_min` |
| `FM-ICL (Mix)` | `icl_mixed` | `Discrete_Stractification_ICL_AI` |
| `FM-ICL (DACE, MST)` | `icl_one_pass_by_class` | `Discrete_Stractification_Enhanced_ICL_AI` + `one_pass_by_class` |
| `FM-ICL (DACE, radius-graph)` | radius sweep | `Discrete_Stractification_Enhanced_ICL_AI` + `radius_by_class` |

Internal names such as `*_min` are kept only where existing result directories require them.

## Paths

The required pca100 TCGA split data for the 18-task grid is copied into this
repo under `data/tcga`. By default, the experiment scripts read:

- data: `data/tcga`
- results: `/lustre/isaac24/scratch/wli66/tabular_transformer_transfer_learning/Result/TCGA`

Override them with:

```bash
export FM_ICL_DATA_ROOT=/path/to/tcga/data
export FM_ICL_RESULT_ROOT=/path/to/Result/TCGA
export FM_ICL_CONDA_ENV=py312
```

The integrated data includes only:

- `Sample_Splits/raw_split_tcga_pca100d_ctxae20d/{BRCA,UCEC,KIRC}_AFR_{mRNA,MicroRNA,Methylation}_{OS,PFI}_2`
- `ancestral_info/black_ids.pkl`
- `ancestral_info/white_ids.pkl`

It intentionally excludes continuum splits, genotype PCA files, raw omics data,
and the full `ancestral_continuum` cache.

Validate the local data package with:

```bash
python scripts/tcga/validate_required_data.py
```

## Main Scripts

Experiment launchers live in `experiments/tcga/`:

- `run_tcga_classical_hparam_pca100_os_pfi_year2.sh`
- `run_tcga_gpu_icl_dstl_hparam_pca100_os_pfi_year2.sh`
- `run_tcga_fmicl_dace_mst_pca100_os_pfi_year2.sh`
- `run_tcga_fmicl_dace_radius_pca100_os_pfi_year2.sh`

Analysis scripts live in `analysis/tcga/`:

- `generate_tcga_hparam_budget30s_metrics_table.py`
- `generate_tcga_budget30s_task_tables.py`
- `plot_budget_curves_pca100.py`
- `plot_budget_curves_with_dace_pca100.py`
- `plot_dace_delta_by_afr_train.py`
- `evaluate_tcga_radius_scales_pca100.py`
- `plot_tcga_radius_scale_curves_pca100_n32_anchors.py`

Outputs are written under `results/tables/` and `results/figures/`.

## Budget Curves

Only the AFR budget-curve paths are integrated:

- the AFR-only budget curve is written to `results/figures/budget_curves_pca100_afr.pdf`
- the DACE-vs-AFR budget curve is written to `results/figures/budget_curves_pca100_dace_vs_afr.pdf`

The Mix budget-curve outputs are intentionally not included.
