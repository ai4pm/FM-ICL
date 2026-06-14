from __future__ import annotations

import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
FMICL_ROOT = REPO_ROOT / "fmicl"
for path in (REPO_ROOT, FMICL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tcga_ancestral_continuum_utils import load_expression_data  # noqa: E402


RESULT_ROOT = Path(
    os.environ.get(
        "FM_ICL_RESULT_ROOT",
        "/lustre/isaac24/scratch/wli66/tabular_transformer_transfer_learning/Result/TCGA",
    )
)
LABEL_CACHE_ROOT = Path(
    os.environ.get(
        "FM_ICL_LABEL_CACHE_ROOT",
        "/lustre/isaac24/scratch/wli66/tabular_transformer_transfer_learning/Data/TCGA/ancestral_continuum",
    )
)
OUT_DIR = Path(os.environ.get("FM_ICL_TABLE_DIR", REPO_ROOT / "results" / "tables"))

CANCERS = ["BRCA", "UCEC", "KIRC"]
OMICS = ["mRNA", "MicroRNA", "Methylation"]
TARGETS = ["OS", "PFI"]
YEAR = 2
PCA_DIM = 100
BUDGET = "budget30s"
N_RUNS = 30

HPARAM_METHODS = [
    "icl_min",
    "icl_mixed",
    "elasticnet_min",
    "elasticnet_mixed",
    "randomforest_min",
    "randomforest_mixed",
    "xgboost_min",
    "xgboost_mixed",
]
ONE_PASS_METHOD = "icl_one_pass_by_class"
METHODS = [*HPARAM_METHODS, ONE_PASS_METHOD]

RUN_LEVEL_CSV = OUT_DIR / "tcga_hparam_budget30s_icl_classical_run_level_metrics.csv"
SUMMARY_CSV = OUT_DIR / "tcga_hparam_budget30s_icl_classical_summary.csv"
SUMMARY_TXT = OUT_DIR / "tcga_hparam_budget30s_icl_classical_summary.txt"
MISSING_CSV = OUT_DIR / "tcga_hparam_budget30s_icl_classical_missing.csv"


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def label_lookup(omics: str, target: str) -> dict[str, int]:
    expression_data = load_expression_data(omics, str(LABEL_CACHE_ROOT), year=YEAR, target=target)
    return {
        str(sample_id): int(label)
        for sample_id, label in zip(expression_data["Samples"], expression_data["Y"])
    }


def normalize_predictions(pred_block: Any) -> np.ndarray:
    pred_arr = np.asarray(pred_block, dtype=float)
    if pred_arr.ndim == 1:
        pred_arr = np.column_stack((1.0 - pred_arr, pred_arr))
    if pred_arr.ndim != 2 or pred_arr.shape[1] < 2:
        raise ValueError(f"invalid prediction array shape: {pred_arr.shape}")
    row_sums = pred_arr.sum(axis=1, keepdims=True)
    needs_norm = np.isfinite(row_sums[:, 0]) & (np.abs(row_sums[:, 0] - 1.0) > 1e-6)
    if np.any(needs_norm):
        safe_sums = np.where(row_sums == 0, 1.0, row_sums)
        pred_arr = pred_arr / safe_sums
    pred_arr = np.clip(pred_arr[:, :2], 1e-15, 1.0 - 1e-15)
    pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)
    return pred_arr


def metrics_from_payload(payload: dict[str, Any], labels: dict[str, int]) -> dict[str, Any]:
    y_true_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []
    n_blocks = 0

    for pred_block, sample_block in zip(payload.get("pred", []), payload.get("sample_name", [])):
        n_blocks += 1
        pred_arr = normalize_predictions(pred_block)
        sample_ids = [str(sample_id) for sample_id in sample_block]
        limit = min(len(sample_ids), pred_arr.shape[0])
        if limit == 0:
            continue
        sample_ids = sample_ids[:limit]
        pred_arr = pred_arr[:limit]
        keep = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in labels]
        if not keep:
            continue
        y_true_parts.append(np.asarray([labels[sample_ids[idx]] for idx in keep], dtype=int))
        y_prob_parts.append(pred_arr[keep])

    if not y_true_parts:
        raise ValueError("no predicted samples matched labels")

    y_true = np.concatenate(y_true_parts)
    y_prob = np.vstack(y_prob_parts)
    if len(np.unique(y_true)) < 2:
        raise ValueError("single-class labels after matching")

    y_pred = np.argmax(y_prob, axis=1)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob[:, -1])),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "ce": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "n_test_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "n_prediction_blocks": int(n_blocks),
    }


def hparam_prediction_path(method: str, cancer: str, omics: str, target: str, search_run: int) -> Path:
    run_name = f"{method}_AFR_{cancer}_{omics}_{target}_year{YEAR}_pca{PCA_DIM}_{BUDGET}"
    return (
        RESULT_ROOT
        / f"Random_HParam_Search_OS_PFI_year{YEAR}_pca{PCA_DIM}"
        / method
        / run_name
        / f"search_run{search_run}"
        / "best_candidate_test_prediction.pkl"
    )


def one_pass_prediction_path(cancer: str, omics: str, target: str, search_run: int) -> Path:
    run_name = (
        f"Discrete_Stractification_Enhanced_ICL_AI_AFR_{cancer}_{omics}_"
        f"{target}_year{YEAR}_pca{PCA_DIM}_ctxae20_n32"
    )
    return (
        RESULT_ROOT
        / "Discrete_Stractification_Enhanced_ICL_AI"
        / run_name
        / "one_pass_by_class"
        / "distance_euclidean"
        / f"prediction_dic_a_run{search_run}.pkl"
    )


def prediction_path(method: str, cancer: str, omics: str, target: str, search_run: int) -> Path:
    if method == ONE_PASS_METHOD:
        return one_pass_prediction_path(cancer, omics, target, search_run)
    return hparam_prediction_path(method, cancer, omics, target, search_run)


def collect_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    labels_by_task = {(omics, target): label_lookup(omics, target) for omics in OMICS for target in TARGETS}

    for method in METHODS:
        for cancer in CANCERS:
            for omics in OMICS:
                for target in TARGETS:
                    labels = labels_by_task[(omics, target)]
                    for search_run in range(N_RUNS):
                        path = prediction_path(method, cancer, omics, target, search_run)
                        base = {
                            "method": method,
                            "cancer": cancer,
                            "omics": omics,
                            "target": target,
                            "budget": BUDGET,
                            "search_run": search_run,
                            "prediction_path": str(path),
                        }
                        if not path.is_file():
                            missing_rows.append({**base, "reason": "missing prediction file"})
                            continue
                        try:
                            payload = load_pickle(path)
                            metrics = metrics_from_payload(payload, labels)
                        except Exception as exc:
                            missing_rows.append({**base, "reason": str(exc)})
                            continue
                        rows.append(
                            {
                                **base,
                                **metrics,
                                "candidate_index": payload.get("candidate_index"),
                                "candidate_finish_seconds": payload.get("candidate_finish_seconds"),
                            }
                        )

    missing_columns = [
        "method",
        "cancer",
        "omics",
        "target",
        "budget",
        "search_run",
        "prediction_path",
        "reason",
    ]
    return pd.DataFrame(rows), pd.DataFrame(missing_rows, columns=missing_columns)


def summarize(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    grouped = run_df.groupby(["method", "cancer", "omics", "target", "budget"], dropna=False)
    summary = grouped.agg(
        runs_evaluated=("search_run", "count"),
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        ce_mean=("ce", "mean"),
        ce_std=("ce", "std"),
        n_test_samples_mean=("n_test_samples", "mean"),
    ).reset_index()
    return summary.sort_values(["method", "cancer", "omics", "target"]).reset_index(drop=True)


def write_txt(summary: pd.DataFrame, missing: pd.DataFrame) -> None:
    cols = [
        "method",
        "cancer",
        "omics",
        "target",
        "budget",
        "runs_evaluated",
        "auroc_mean",
        "auroc_std",
        "f1_mean",
        "f1_std",
        "ce_mean",
        "ce_std",
    ]
    with SUMMARY_TXT.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(cols) + "\n")
        for _, row in summary[cols].iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))
            handle.write("\t".join(values) + "\n")
        if not missing.empty:
            handle.write("\nMissing or invalid payloads:\n")
            for _, row in missing.iterrows():
                handle.write(
                    f"{row['method']}\t{row['cancer']}\t{row['omics']}\t{row['target']}\t"
                    f"run{row['search_run']}\t{row['reason']}\t{row['prediction_path']}\n"
                )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_df, missing_df = collect_metrics()
    summary_df = summarize(run_df)

    run_df.to_csv(RUN_LEVEL_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    missing_df.to_csv(MISSING_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    write_txt(summary_df, missing_df)

    print(f"[saved] {RUN_LEVEL_CSV}")
    print(f"[saved] {SUMMARY_CSV}")
    print(f"[saved] {SUMMARY_TXT}")
    print(f"[saved] {MISSING_CSV}")
    if not missing_df.empty:
        print(f"[warn] missing_or_invalid={len(missing_df)}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
