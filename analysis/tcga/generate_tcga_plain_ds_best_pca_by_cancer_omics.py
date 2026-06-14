from __future__ import annotations

import argparse
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
FMICL_ROOT = REPO_ROOT / "fmicl"
for path in (REPO_ROOT, FMICL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tcga_ancestral_continuum_utils import load_expression_data


DATASET_NAME = "TCGA"
ANCESTRY = "AFR"
RESULT_ROOT = Path(
    os.environ.get(
        "FM_ICL_RESULT_ROOT",
        "/lustre/isaac24/scratch/wli66/tabular_transformer_transfer_learning/Result/TCGA",
    )
)
CACHE_DIR = Path(
    os.environ.get(
        "FM_ICL_LABEL_CACHE_ROOT",
        "/lustre/isaac24/scratch/wli66/tabular_transformer_transfer_learning/Data/TCGA/ancestral_continuum",
    )
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "tables"

TCGA_CANCERS = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
]
TCGA_OMICS = ["mRNA", "MicroRNA", "Methylation"]
DEFAULT_PCA_DIMS = [100]


@dataclass(frozen=True)
class MethodSpec:
    label: str
    folder: str
    dir_pattern: str
    extra_path: str = ""

    def result_dir(
        self,
        cancer: str,
        omics: str,
        target: str,
        year: int,
        pca_dim: int,
        result_suffix: str = "",
    ) -> Path:
        directory = self.dir_pattern.format(
            ancestry=ANCESTRY,
            cancer=cancer,
            omics=omics,
            target=target,
            year=year,
            pca=pca_dim,
        )
        if result_suffix:
            directory = f"{directory}{result_suffix}"
        path = RESULT_ROOT / self.folder / directory
        if self.extra_path:
            path = path / self.extra_path
        return path


METHOD_SPECS = [
    MethodSpec("DS-Enhanced-ICL-batch-by-class", "Discrete_Stractification_Enhanced_ICL_AI", "Discrete_Stractification_Enhanced_ICL_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}_ctxae20", "batch_by_class/distance_euclidean"),
    MethodSpec("DS-Enhanced-ICL-one-pass-by-class", "Discrete_Stractification_Enhanced_ICL_AI", "Discrete_Stractification_Enhanced_ICL_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}_ctxae20", "one_pass_by_class/distance_euclidean"),
    MethodSpec("DS-ICL-maj", "Discrete_Stractification_ICL_AI_maj", "Discrete_Stractification_ICL_AI_maj_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
    MethodSpec("DS-ICL-min", "Discrete_Stractification_ICL_AI_min", "Discrete_Stractification_ICL_AI_min_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
    MethodSpec("DS-ICL-mixed", "Discrete_Stractification_ICL_AI", "Discrete_Stractification_ICL_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
    MethodSpec("DS-TL", "Discrete_Stractification_TL_AI", "Discrete_Stractification_TL_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}", "hyperoption2"),
    MethodSpec("ds-elastic-net", "Discrete_Stractification_ElasticNet_AI", "Discrete_Stractification_ElasticNet_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
    MethodSpec("ds-elastic-net-maj", "Discrete_Stractification_ElasticNet_AI_maj", "Discrete_Stractification_ElasticNet_AI_maj_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
    MethodSpec("ds-elastic-net-min", "Discrete_Stractification_ElasticNet_AI_min", "Discrete_Stractification_ElasticNet_AI_min_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
    MethodSpec("ds-enhanced-elastic-net-batch-by-class", "Discrete_Stractification_Enhanced_ElasticNet_AI", "Discrete_Stractification_Enhanced_ElasticNet_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}_ctxae20", "batch_by_class/distance_euclidean"),
    MethodSpec("ds-enhanced-elastic-net-one-pass-by-class", "Discrete_Stractification_Enhanced_ElasticNet_AI", "Discrete_Stractification_Enhanced_ElasticNet_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}_ctxae20", "one_pass_by_class/distance_euclidean"),
    MethodSpec("ds-enhanced-random-forest-batch-by-class", "Discrete_Stractification_Enhanced_RandomForest_AI", "Discrete_Stractification_Enhanced_RandomForest_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}_ctxae20", "batch_by_class/distance_euclidean"),
    MethodSpec("ds-enhanced-random-forest-one-pass-by-class", "Discrete_Stractification_Enhanced_RandomForest_AI", "Discrete_Stractification_Enhanced_RandomForest_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}_ctxae20", "one_pass_by_class/distance_euclidean"),
    MethodSpec("ds-enhanced-xgboost-batch-by-class", "Discrete_Stractification_Enhanced_XGBoost_AI", "Discrete_Stractification_Enhanced_XGBoost_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}_ctxae20", "batch_by_class/distance_euclidean"),
    MethodSpec("ds-enhanced-xgboost-one-pass-by-class", "Discrete_Stractification_Enhanced_XGBoost_AI", "Discrete_Stractification_Enhanced_XGBoost_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}_ctxae20", "one_pass_by_class/distance_euclidean"),
    MethodSpec("ds-random-forest", "Discrete_Stractification_RandomForest_AI", "Discrete_Stractification_RandomForest_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
    MethodSpec("ds-random-forest-maj", "Discrete_Stractification_RandomForest_AI_maj", "Discrete_Stractification_RandomForest_AI_maj_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
    MethodSpec("ds-random-forest-min", "Discrete_Stractification_RandomForest_AI_min", "Discrete_Stractification_RandomForest_AI_min_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
    MethodSpec("ds-xgboost", "Discrete_Stractification_XGBoost_AI", "Discrete_Stractification_XGBoost_AI_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
    MethodSpec("ds-xgboost-maj", "Discrete_Stractification_XGBoost_AI_maj", "Discrete_Stractification_XGBoost_AI_maj_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
    MethodSpec("ds-xgboost-min", "Discrete_Stractification_XGBoost_AI_min", "Discrete_Stractification_XGBoost_AI_min_{ancestry}_{cancer}_{omics}_{target}_year{year}_pca{pca}"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TCGA plain-DS best-PCA summaries by cancer and omics.")
    parser.add_argument("--target", required=True, choices=["OS", "PFI", "DSS", "DFI"])
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--pca-dims", nargs="+", type=int, default=DEFAULT_PCA_DIMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-text", action="store_true", default=True)
    return parser.parse_args()


def build_label_lookup(omics: str, target: str, year: int) -> dict[str, int]:
    expression_data = load_expression_data(omics, str(CACHE_DIR), year=year, target=target)
    samples = [str(sample_id) for sample_id in expression_data["Samples"]]
    labels = [int(label) for label in expression_data["Y"]]
    return dict(zip(samples, labels))


def load_prediction_metrics(run_path: Path, label_lookup: dict[str, int]) -> dict[str, float] | None:
    with open(run_path, "rb") as handle:
        prediction_dic = pickle.load(handle)

    fold_aurocs: list[float] = []
    fold_f1s: list[float] = []
    for pred_block, sample_block in zip(prediction_dic.get("pred", []), prediction_dic.get("sample_name", [])):
        pred_arr = np.asarray(pred_block, dtype=float)
        if pred_arr.size == 0:
            continue
        sample_ids = [str(sample_id) for sample_id in sample_block]
        limit = min(len(sample_ids), pred_arr.shape[0])
        if limit == 0:
            continue
        sample_ids = sample_ids[:limit]
        pred_arr = pred_arr[:limit]

        keep_idx = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in label_lookup]
        if not keep_idx:
            continue
        y_true_fold = np.asarray([label_lookup[sample_ids[idx]] for idx in keep_idx], dtype=int)
        y_prob_fold = pred_arr[keep_idx]
        if y_prob_fold.ndim == 1:
            y_prob_fold = np.column_stack((1.0 - y_prob_fold, y_prob_fold))
        if len(np.unique(y_true_fold)) < 2:
            continue
        y_pred_fold = y_prob_fold.argmax(axis=1)
        try:
            fold_auc = float(roc_auc_score(y_true_fold, y_prob_fold[:, -1], average="macro"))
            fold_f1 = float(f1_score(y_true_fold, y_pred_fold, average="macro"))
        except Exception:
            continue
        if np.isnan(fold_auc) or np.isnan(fold_f1):
            continue
        fold_aurocs.append(fold_auc)
        fold_f1s.append(fold_f1)

    if not fold_f1s or not fold_aurocs:
        return None

    return {
        "aucroc": float(np.nanmean(fold_aurocs)),
        "f1": float(np.nanmean(fold_f1s)),
        "error": np.nan,
    }


def find_run_paths(result_dir: Path) -> list[Path]:
    run_paths = sorted(result_dir.glob("prediction_dic_a_run*.pkl"))
    if run_paths:
        return run_paths

    fallback_dir = result_dir / "no_label_in_search"
    if fallback_dir.is_dir():
        return sorted(fallback_dir.glob("prediction_dic_a_run*.pkl"))

    return []


def summarize_candidate_dir(result_dir: Path, label_lookup: dict[str, int]) -> tuple[float, float, float, float, int]:
    run_paths = find_run_paths(result_dir)
    if not run_paths:
        return (np.nan, np.nan, np.nan, np.nan, 0)

    aurocs: list[float] = []
    f1s: list[float] = []
    valid_runs = 0
    for run_path in run_paths:
        metrics = load_prediction_metrics(run_path, label_lookup)
        if metrics is None:
            continue
        valid_runs += 1
        aurocs.append(metrics["aucroc"])
        f1s.append(metrics["f1"])

    if valid_runs == 0:
        return (np.nan, np.nan, np.nan, np.nan, 0)

    return (
        float(np.nanmean(aurocs)),
        float(np.nanstd(aurocs)),
        float(np.nanmean(f1s)),
        float(np.nanstd(f1s)),
        valid_runs,
    )


def build_summary_rows(target: str, year: int, pca_dims: list[int]) -> pd.DataFrame:
    label_lookup_by_omics = {
        omics: build_label_lookup(omics, target, year)
        for omics in TCGA_OMICS
    }

    rows: list[dict[str, object]] = []
    for cancer in TCGA_CANCERS:
        for omics in TCGA_OMICS:
            label_lookup = label_lookup_by_omics[omics]
            for method_spec in METHOD_SPECS:
                best_record: dict[str, object] | None = None
                for pca_dim in pca_dims:
                    result_dir = method_spec.result_dir(cancer, omics, target, year, pca_dim)
                    mean_auroc, std_auroc, mean_f1, std_f1, runs_evaluated = summarize_candidate_dir(result_dir, label_lookup)
                    candidate = {
                        "dataset": DATASET_NAME,
                        "cancer_type": cancer,
                        "omics_type": omics,
                        "target": target,
                        "year": year,
                        "method": method_spec.label,
                        "best_pca_dim": pca_dim,
                        "best_mean_auroc": mean_auroc,
                        "best_std_auroc": std_auroc,
                        "best_mean_f1": mean_f1,
                        "best_std_f1": std_f1,
                        "runs_evaluated": runs_evaluated,
                    }
                    if best_record is None:
                        best_record = candidate
                        continue

                    candidate_score = candidate["best_mean_auroc"]
                    best_score = best_record["best_mean_auroc"]
                    if (
                        (not math.isnan(candidate_score) and math.isnan(best_score))
                        or (not math.isnan(candidate_score) and not math.isnan(best_score) and candidate_score > best_score)
                    ):
                        best_record = candidate

                assert best_record is not None
                rows.append(best_record)

    return pd.DataFrame(rows)


def write_ranked_text(df: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for cancer in TCGA_CANCERS:
            cancer_df = df[df["cancer_type"] == cancer]
            if cancer_df.empty:
                continue
            handle.write(f"## {cancer}\n")
            for omics in TCGA_OMICS:
                omics_df = cancer_df[cancer_df["omics_type"] == omics].copy()
                if omics_df.empty:
                    continue
                handle.write(f"### {omics}\n")
                omics_df = omics_df.sort_values(
                    by=["best_mean_auroc", "best_mean_f1", "method"],
                    ascending=[False, False, True],
                    na_position="last",
                )
                for row in omics_df.itertuples(index=False):
                    handle.write(
                        f"{row.method}\t"
                        f"PCA={row.best_pca_dim}\t"
                        f"runs={row.runs_evaluated}\t"
                        f"AUROC={row.best_mean_auroc:.6f}\t"
                        f"AUROC_STD={row.best_std_auroc:.6f}\t"
                        f"F1={row.best_mean_f1:.6f}\t"
                        f"F1_STD={row.best_std_f1:.6f}\n"
                    )
                handle.write("\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = build_summary_rows(args.target, args.year, args.pca_dims)
    stem = f"tcga_{args.target.lower()}_year{args.year}_plain_ds_best_pca_by_cancer_omics"
    csv_path = args.output_dir / f"{stem}.csv"
    txt_path = args.output_dir / f"{stem}.txt"

    summary_df.to_csv(csv_path, index=False)
    if args.write_text:
        write_ranked_text(summary_df, txt_path)

    print(f"wrote {csv_path}")
    if args.write_text:
        print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
