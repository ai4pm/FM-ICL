"""Directory-level prediction summarization helpers."""

from pathlib import Path

from analysis.tcga.generate_tcga_plain_ds_best_pca_by_cancer_omics import summarize_candidate_dir  # noqa: F401


def prediction_files(result_dir: Path) -> list[Path]:
    return sorted(Path(result_dir).glob("prediction_dic_a_run*.pkl"))
