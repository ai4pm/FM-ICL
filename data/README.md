# Data

This folder contains the lean TCGA data package required to run the pca100
FM-ICL experiments in this repository.

`tcga_18_tasks.csv` lists the 18-task grid:

- cancers: `BRCA`, `UCEC`, `KIRC`
- endpoints: `OS`, `PFI`
- omics: `mRNA`, `MicroRNA`, `Methylation`
- year: `2`
- feature dimension: `pca100`

Required local data lives under `data/tcga`:

- `Sample_Splits/raw_split_tcga_pca100d_ctxae20d/`
- `ancestral_info/black_ids.pkl`
- `ancestral_info/white_ids.pkl`

Only the 18 task folders used by this repo are included. The full TCGA split
tree, continuum splits, genotype PCA files, raw omics data, and full
`ancestral_continuum` cache are intentionally not included.

Validate the data package from the repo root with:

```bash
python scripts/tcga/validate_required_data.py
```
