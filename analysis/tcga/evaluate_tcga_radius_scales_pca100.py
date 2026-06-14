from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FMICL_ROOT = REPO_ROOT / "fmicl"
for path in (REPO_ROOT, FMICL_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_tcga_plain_ds_best_pca_by_cancer_omics import (  # noqa: E402
    RESULT_ROOT,
    build_label_lookup,
    summarize_candidate_dir,
)


CANCERS = ("BRCA", "UCEC", "KIRC")
TARGETS = ("OS", "PFI")
OMICS = ("mRNA", "MicroRNA", "Methylation")
PCA_DIMS = (100,)
SCALES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.20, 2.40, 2.60, 2.80, 3.00)
YEAR = 2
OUT_CSV = REPO_ROOT / "results" / "tables" / "tcga_radius_by_class_scales_pca100_summary.csv"


def scale_suffix(scale: float) -> str:
    return str(scale).replace("-", "m").replace(".", "p")


def result_dir(cancer: str, omics: str, target: str, pca_dim: int, scale: float) -> Path:
    stem = (
        f"Discrete_Stractification_Enhanced_ICL_AI_AFR_{cancer}_{omics}_"
        f"{target}_year{YEAR}_pca{pca_dim}_ctxae20"
    )
    return (
        RESULT_ROOT
        / "Discrete_Stractification_Enhanced_ICL_AI"
        / stem
        / f"radius_by_class_{scale_suffix(scale)}"
        / "distance_euclidean"
    )


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    label_lookup = {
        (omics, target): build_label_lookup(omics, target, YEAR)
        for omics in OMICS
        for target in TARGETS
    }

    rows: list[dict[str, object]] = []
    for cancer in CANCERS:
        for target in TARGETS:
            for omics in OMICS:
                for pca_dim in PCA_DIMS:
                    for scale in SCALES:
                        candidate_dir = result_dir(cancer, omics, target, pca_dim, scale)
                        mean_auroc, std_auroc, mean_f1, std_f1, runs_evaluated = summarize_candidate_dir(
                            candidate_dir,
                            label_lookup[(omics, target)],
                        )
                        rows.append(
                            {
                                "cancer_type": cancer,
                                "target": target,
                                "omics_type": omics,
                                "pca_dim": pca_dim,
                                "radius_scale": scale,
                                "mean_auroc": mean_auroc,
                                "std_auroc": std_auroc,
                                "mean_f1": mean_f1,
                                "std_f1": std_f1,
                                "runs_evaluated": runs_evaluated,
                                "prediction_files": len(list(candidate_dir.glob("prediction_dic_a_run*.pkl")))
                                if candidate_dir.is_dir()
                                else 0,
                                "result_dir": str(candidate_dir),
                            }
                        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    complete = df[df["runs_evaluated"].eq(30)].copy()
    best = (
        complete.sort_values(["cancer_type", "target", "mean_auroc"], ascending=[True, True, False])
        .groupby(["cancer_type", "target"], as_index=False)
        .head(1)
        .sort_values(["cancer_type", "target"])
        .reset_index(drop=True)
    )

    print("Best AUROC for radius_by_class, pca100, across radius scales and omics")
    print(
        best[
            [
                "cancer_type",
                "target",
                "omics_type",
                "pca_dim",
                "radius_scale",
                "mean_auroc",
                "std_auroc",
                "mean_f1",
                "std_f1",
                "runs_evaluated",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    missing = df[~df["runs_evaluated"].eq(30)]
    print(f"\nSaved full scan: {OUT_CSV}")
    print(f"Complete rows: {len(complete)}/{len(df)}")
    if not missing.empty:
        print("Incomplete rows by scale:")
        print(missing.groupby("radius_scale")["runs_evaluated"].count().to_string())


if __name__ == "__main__":
    main()
