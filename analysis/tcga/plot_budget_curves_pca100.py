"""
plot_hparam_budget_curves.py

Reproduces a 3-panel figure (mean ROC AUC, win counts, mean rank) as a
function of time budget, analogous to TabPFN Figure 5, using the TCGA
random hyperparameter search results.

Uncertainty bands are computed across search runs only. For each search_run,
the script first aggregates over the cancer * omics * target tasks, then
plots the mean +/- 95% CI over those search-run-level realizations.

Usage:
    python plot_hparam_budget_curves.py \
        --targets OS PFI \
        --year 2 \
        --pca-dim 400 \
        --cancers KIRC UCEC BRCA \
        --omics mRNA MicroRNA Methylation \
        --output-path budget_curves.pdf
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
FMICL_ROOT = REPO_ROOT / "fmicl"
if str(FMICL_ROOT) not in sys.path:
    sys.path.insert(0, str(FMICL_ROOT))
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

# ── experiment axes ───────────────────────────────────────────────────────────
TCGA_CANCERS = ["KIRC", "UCEC", "BRCA"]
TCGA_OMICS   = ["mRNA", "MicroRNA", "Methylation"]

TIME_LABELS   = ["budget1s", "budget5s", "budget30s", "budget5m", "budget1h"]
TIME_DISPLAY  = ["1s",       "5s",       "30s",        "5min",    "1h"]
TIME_LIMIT_MINS = [1/60, 5/60, 0.5, 5.0, 60.0]
TIME_LIMIT_SECONDS = dict(zip(TIME_LABELS, [limit * 60.0 for limit in TIME_LIMIT_MINS]))

METHODS = ["ds_tl", "icl_mixed", "elasticnet_mixed", "randomforest_mixed", "xgboost_mixed"]
ALL_METHODS = [
    "ds_tl",
    "icl_min",
    "icl_mixed",
    "elasticnet_min",
    "elasticnet_mixed",
    "randomforest_min",
    "randomforest_mixed",
    "xgboost_min",
    "xgboost_mixed",
]
METHOD_LABELS = {
    "ds_tl":            "TL",
    "icl_min":            "FM-ICL (AFR)",
    "icl_mixed":          "FM-ICL (Mix)",
    "elasticnet_min":     "ElasticNet (AFR)",
    "elasticnet_mixed":   "ElasticNet (Mix)",
    "randomforest_min":   "RandomForest (AFR)",
    "randomforest_mixed": "RandomForest (Mix)",
    "xgboost_min":        "XGBoost (AFR)",
    "xgboost_mixed":      "XGBoost (Mix)",
}
METHOD_COLORS = {
    "ds_tl":            "#e07b39",
    "icl_min":            "#3a86c8",
    "icl_mixed":          "#3a86c8",
    "elasticnet_min":     "#e63946",
    "elasticnet_mixed":   "#e63946",
    "randomforest_min":   "#2a9d5c",
    "randomforest_mixed": "#2a9d5c",
    "xgboost_min":        "#9b5de5",
    "xgboost_mixed":      "#9b5de5",
}
METHOD_MARKERS = {
    "ds_tl":            "o",
    "icl_min":            "s",
    "icl_mixed":          "s",
    "elasticnet_min":     "^",
    "elasticnet_mixed":   "^",
    "randomforest_min":   "D",
    "randomforest_mixed": "D",
    "xgboost_min":        "P",
    "xgboost_mixed":      "P",
}

N_REPEATS = 30


# ── plotting style ───────────────────────────────────────────────────────────

def SetPltProp(
    ax,
    xn=None,
    yn=None,
    title=None,
    grid=True,
    bbox_to_anchor=None,
    legend=True,
    pos="upper left",
    borderpad=None,
    ylim=None,
):
    fontsize = 14
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.5)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight("bold")

    if legend:
        if bbox_to_anchor is None:
            ax.legend(
                loc=pos,
                shadow=True,
                prop={"weight": "bold", "size": 10},
                borderpad=borderpad,
            )
        else:
            ax.legend(
                loc=pos,
                shadow=True,
                bbox_to_anchor=bbox_to_anchor,
                prop={"weight": "bold", "size": 10},
                borderpad=borderpad,
            )
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(linewidth="1.5", linestyle="dashed")
    if xn is not None:
        ax.set_xlabel(xn, fontweight="bold", fontsize=fontsize)
    if yn is not None:
        ax.set_ylabel(yn, fontweight="bold", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=fontsize)
    return ax


# ── label lookup ─────────────────────────────────────────────────────────────

def build_label_lookup(omics: str, target: str, year: int) -> dict[str, int]:
    """Load ground-truth labels from the ancestral continuum cache."""
    from tcga_ancestral_continuum_utils import load_expression_data  # type: ignore
    expression_data = load_expression_data(omics, str(CACHE_DIR), year=year, target=target)
    samples = [str(s) for s in expression_data["Samples"]]
    labels  = [int(lb) for lb in expression_data["Y"]]
    return dict(zip(samples, labels))


# ── per-candidate AUC ────────────────────────────────────────────────────────

def load_candidate_payload(candidate_path: Path) -> dict | None:
    try:
        with candidate_path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def candidate_elapsed_seconds(payload: dict) -> float:
    elapsed = payload.get("candidate_elapsed_seconds")
    if elapsed is not None:
        try:
            elapsed_float = float(elapsed)
            if elapsed_float >= 0 and not np.isnan(elapsed_float):
                return elapsed_float
        except Exception:
            pass
    fold_times = []
    for value in payload.get("time", []):
        try:
            value_float = float(value)
        except Exception:
            continue
        if value_float >= 0 and not np.isnan(value_float):
            fold_times.append(value_float)
    return float(np.sum(fold_times)) if fold_times else float("nan")


def candidate_auroc_from_payload(payload: dict, label_lookup: dict[str, int]) -> float | None:
    """Return mean-fold AUROC for one loaded candidate payload, or None on failure."""
    fold_aurocs: list[float] = []
    for pred_block, sample_block in zip(
        payload.get("pred", []), payload.get("sample_name", [])
    ):
        pred_arr = np.asarray(pred_block, dtype=float)
        if pred_arr.size == 0:
            continue
        sample_ids = [str(s) for s in sample_block]
        limit = min(len(sample_ids), pred_arr.shape[0])
        if limit == 0:
            continue
        sample_ids = sample_ids[:limit]
        pred_arr   = pred_arr[:limit]

        keep = [i for i, sid in enumerate(sample_ids) if sid in label_lookup]
        if not keep:
            continue
        y_true = np.array([label_lookup[sample_ids[i]] for i in keep], dtype=int)
        y_prob = pred_arr[keep]
        if y_prob.ndim == 1:
            y_prob = np.column_stack((1.0 - y_prob, y_prob))
        if len(np.unique(y_true)) < 2:
            continue
        try:
            auc = float(roc_auc_score(y_true, y_prob[:, -1], average="macro"))
        except Exception:
            continue
        if not np.isnan(auc):
            fold_aurocs.append(auc)

    if not fold_aurocs:
        return None
    return float(np.nanmean(fold_aurocs))


# ── result directory builder ─────────────────────────────────────────────────

def hparam_search_dir(
    method: str,
    cancer: str,
    omics: str,
    target: str,
    year: int,
    pca_dim: int,
    search_run: int,
    result_suffix: str,
) -> Path:
    run_name = (
        f"{method}_AFR_{cancer}_{omics}_{target}_year{year}_pca{pca_dim}{result_suffix}"
    )
    target_group = "OS_PFI" if target in {"OS", "PFI"} else target
    root_name = f"Random_HParam_Search_{target_group}_year{year}_pca{pca_dim}"
    return RESULT_ROOT / root_name / method / run_name / f"search_run{search_run}"


# ── best-candidate AUC for one (method, cancer, omics, target, budget, seed) ─

def best_auroc_for_run(
    method: str,
    cancer: str,
    omics: str,
    target: str,
    year: int,
    pca_dim: int,
    search_run: int,
    budget_label: str,
    label_lookup: dict[str, int],
) -> dict:
    """
    Among all candidate_XXXXX_prediction.pkl files saved under the given
    budget folder, return the BEST mean-fold AUROC among candidates whose
    reconstructed cumulative finish time stays within the budget.
    """
    result_suffix = f"_{budget_label}"
    run_dir = hparam_search_dir(
        method, cancer, omics, target, year, pca_dim, search_run, result_suffix
    )
    best_prediction_path = run_dir / "best_candidate_test_prediction.pkl"
    if best_prediction_path.is_file():
        payload = load_candidate_payload(best_prediction_path)
        auc = candidate_auroc_from_payload(payload, label_lookup) if payload is not None else None
        return {
            "auroc": auc,
            "selected_candidate_index": payload.get("candidate_index") if payload else None,
            "selected_candidate_elapsed_seconds": payload.get("candidate_elapsed_seconds") if payload else None,
            "selected_candidate_finish_seconds": payload.get("candidate_finish_seconds") if payload else None,
            "budget_seconds": payload.get("budget_seconds", TIME_LIMIT_SECONDS[budget_label]) if payload else TIME_LIMIT_SECONDS[budget_label],
            "n_candidates_total": None,
            "n_candidates_within_budget": None,
            "result_dir": str(run_dir),
        }

    candidate_files = sorted(run_dir.glob("candidate_*_prediction.pkl"))
    if not candidate_files:
        return {
            "auroc": None,
            "selected_candidate_index": None,
            "selected_candidate_elapsed_seconds": None,
            "selected_candidate_finish_seconds": None,
            "budget_seconds": TIME_LIMIT_SECONDS[budget_label],
            "n_candidates_total": 0,
            "n_candidates_within_budget": 0,
            "result_dir": str(run_dir),
        }

    best: float | None = None
    best_candidate_index: int | None = None
    best_candidate_elapsed: float | None = None
    best_candidate_finish: float | None = None
    cumulative_seconds = 0.0
    n_total = 0
    n_within_budget = 0
    budget_seconds = TIME_LIMIT_SECONDS[budget_label]

    for cpath in candidate_files:
        payload = load_candidate_payload(cpath)
        if payload is None:
            continue
        n_total += 1
        elapsed_seconds = candidate_elapsed_seconds(payload)
        if np.isnan(elapsed_seconds):
            continue
        cumulative_seconds += elapsed_seconds
        candidate_index = int(payload.get("candidate_index", n_total - 1))
        if cumulative_seconds > budget_seconds:
            continue
        n_within_budget += 1
        auc = candidate_auroc_from_payload(payload, label_lookup)
        if auc is None:
            continue
        if best is None or auc > best:
            best = auc
            best_candidate_index = candidate_index
            best_candidate_elapsed = elapsed_seconds
            best_candidate_finish = cumulative_seconds

    return {
        "auroc": best,
        "selected_candidate_index": best_candidate_index,
        "selected_candidate_elapsed_seconds": best_candidate_elapsed,
        "selected_candidate_finish_seconds": best_candidate_finish,
        "budget_seconds": budget_seconds,
        "n_candidates_total": n_total,
        "n_candidates_within_budget": n_within_budget,
        "result_dir": str(run_dir),
    }


# ── collect all results into a DataFrame ─────────────────────────────────────

def collect_results(
    cancers: list[str],
    omics_list: list[str],
    targets: list[str],
    year: int,
    pca_dim: int,
    n_repeats: int,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    methods_to_collect = methods or METHODS

    # pre-build label lookups
    label_lookups: dict[tuple[str, str], dict[str, int]] = {}
    for omics in omics_list:
        for target in targets:
            key = (omics, target)
            print(f"  building label lookup: omics={omics} target={target}")
            try:
                label_lookups[key] = build_label_lookup(omics, target, year)
            except Exception as exc:
                warnings.warn(f"Could not build label lookup for {key}: {exc}")
                label_lookups[key] = {}

    total = (
        len(methods_to_collect) * len(cancers) * len(omics_list)
        * len(targets) * len(TIME_LABELS) * n_repeats
    )
    done = 0
    for method in methods_to_collect:
        for cancer in cancers:
            for omics in omics_list:
                for target in targets:
                    ll = label_lookups.get((omics, target), {})
                    for budget_label in TIME_LABELS:
                        for search_run in range(n_repeats):
                            done += 1
                            if done % 200 == 0:
                                print(f"  progress {done}/{total}")
                            result = best_auroc_for_run(
                                method, cancer, omics, target, year,
                                pca_dim, search_run, budget_label, ll,
                            )
                            rows.append(
                                {
                                    "method":      method,
                                    "cancer":      cancer,
                                    "omics":       omics,
                                    "target":      target,
                                    "budget":      budget_label,
                                    "search_run":  search_run,
                                    "auroc":       result["auroc"],
                                    "budget_seconds": result["budget_seconds"],
                                    "selected_candidate_index": result["selected_candidate_index"],
                                    "selected_candidate_elapsed_seconds": result["selected_candidate_elapsed_seconds"],
                                    "selected_candidate_finish_seconds": result["selected_candidate_finish_seconds"],
                                    "n_candidates_total": result["n_candidates_total"],
                                    "n_candidates_within_budget": result["n_candidates_within_budget"],
                                    "strict_cumulative_budget": True,
                                    "result_dir": result["result_dir"],
                                }
                            )

    return pd.DataFrame(rows)


def write_raw_results(df: pd.DataFrame, output_xlsx: Path) -> None:
    """Write raw run-level rows to Excel and a same-stem CSV sidecar."""
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_xlsx, index=False)
    output_csv = output_xlsx.with_suffix(".csv")
    df.to_csv(output_csv, index=False)
    print(f"[saved] {output_xlsx}")
    print(f"[saved] {output_csv}")


def read_method_result_files(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []
    for path in paths:
        if not path.is_file():
            missing.append(path)
            continue
        if path.suffix.lower() in {".xlsx", ".xls"}:
            frames.append(pd.read_excel(path))
        else:
            frames.append(pd.read_csv(path))
    if missing:
        missing_str = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing method result files:\n{missing_str}")
    if not frames:
        raise ValueError("No method result files were provided.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["search_run"] = combined["search_run"].astype(int)
    return combined


# ── aggregation helpers ───────────────────────────────────────────────────────

def summarize_search_run_realizations(
    realization_df: pd.DataFrame,
    value_col: str,
    summary_col: str,
) -> pd.DataFrame:
    """Mean +/- 95% CI across search-run-level realization values."""
    grp = realization_df.dropna(subset=[value_col]).groupby(["method", "budget"])[value_col]
    out = (
        pd.concat(
            [
                grp.mean().rename(summary_col),
                grp.std().rename("std"),
                grp.sem().rename("sem"),
                grp.count().rename("n_search_runs"),
            ],
            axis=1,
        )
        .reset_index()
    )
    out["ci95"] = 1.96 * out["sem"]
    return out


def compute_search_run_realizations(
    df: pd.DataFrame,
    methods: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return one metric value per (method, budget, search_run).

    Each realization aggregates over the task grid
    cancer * omics * target.  Variance/CI can then be computed across
    search runs only, without mixing in task-to-task heterogeneity.
    """
    valid = df.dropna(subset=["auroc"]).copy()

    mean_auc_runs = (
        valid.groupby(["method", "budget", "search_run"])["auroc"]
        .agg(mean_auroc="mean", n_tasks="count")
        .reset_index()
    )

    key_cols = ["cancer", "omics", "target", "search_run", "budget"]
    idx_best = valid.groupby(key_cols)["auroc"].idxmax()
    winners = valid.loc[idx_best, ["method", "budget", "search_run"]].copy()
    wins = (
        winners.groupby(["method", "budget", "search_run"])
        .size()
        .reset_index(name="wins")
    )
    full_index = pd.MultiIndex.from_product(
        [methods, TIME_LABELS, sorted(valid["search_run"].unique())],
        names=["method", "budget", "search_run"],
    )
    wins_runs = (
        wins.set_index(["method", "budget", "search_run"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    ranked = valid.copy()
    ranked["rank"] = ranked.groupby(key_cols)["auroc"].rank(
        ascending=False,
        method="average",
    )
    rank_runs = (
        ranked.groupby(["method", "budget", "search_run"])["rank"]
        .agg(mean_rank="mean", n_ranked_tasks="count")
        .reset_index()
    )

    return mean_auc_runs, wins_runs, rank_runs


# ── plotting ──────────────────────────────────────────────────────────────────

def make_figure(
    df_raw: pd.DataFrame,
    output_path: Path,
    methods: list[str],
    title_suffix: str = "",
) -> None:
    budget_order = TIME_LABELS
    x_ticks      = list(range(len(budget_order)))
    x_labels     = TIME_DISPLAY

    mean_auc_runs, wins_runs, rank_runs = compute_search_run_realizations(df_raw, methods)
    mean_auc = summarize_search_run_realizations(
        mean_auc_runs,
        value_col="mean_auroc",
        summary_col="mean",
    )
    wins = summarize_search_run_realizations(
        wins_runs,
        value_col="wins",
        summary_col="wins",
    )
    ranks = summarize_search_run_realizations(
        rank_runs,
        value_col="mean_rank",
        summary_col="mean_rank",
    )

    realization_prefix = output_path.with_suffix("")
    mean_auc_runs.to_csv(
        realization_prefix.with_name(f"{realization_prefix.name}_mean_auc_by_search_run.csv"),
        index=False,
    )
    wins_runs.to_csv(
        realization_prefix.with_name(f"{realization_prefix.name}_wins_by_search_run.csv"),
        index=False,
    )
    rank_runs.to_csv(
        realization_prefix.with_name(f"{realization_prefix.name}_mean_rank_by_search_run.csv"),
        index=False,
    )
    mean_auc.to_csv(
        realization_prefix.with_name(f"{realization_prefix.name}_mean_auc_summary.csv"),
        index=False,
    )
    wins.to_csv(
        realization_prefix.with_name(f"{realization_prefix.name}_wins_summary.csv"),
        index=False,
    )
    ranks.to_csv(
        realization_prefix.with_name(f"{realization_prefix.name}_mean_rank_summary.csv"),
        index=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.subplots_adjust(bottom=0.28, wspace=0.35)

    for method in methods:
        color  = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        label  = METHOD_LABELS[method]

        # ── panel 1: mean AUC ────────────────────────────────────────────────
        sub = mean_auc[mean_auc["method"] == method].set_index("budget").reindex(budget_order)
        y   = sub["mean"].values.astype(float)
        ci  = sub["ci95"].values.astype(float)
        axes[0].plot(x_ticks, y, color=color, marker=marker, linewidth=2,
                     markersize=7, label=label)
        axes[0].fill_between(
            x_ticks,
            np.where(np.isnan(y - ci), np.nan, y - ci),
            np.where(np.isnan(y + ci), np.nan, y + ci),
            color=color, alpha=0.15,
        )

        # ── panel 2: wins ────────────────────────────────────────────────────
        sub2 = wins[wins["method"] == method].set_index("budget").reindex(budget_order)
        w    = sub2["wins"].values.astype(float)
        ci2  = sub2["ci95"].values.astype(float)
        axes[1].plot(x_ticks, w, color=color, marker=marker, linewidth=2,
                     markersize=7)
        axes[1].fill_between(
            x_ticks,
            np.where(np.isnan(w - ci2), np.nan, w - ci2),
            np.where(np.isnan(w + ci2), np.nan, w + ci2),
            color=color, alpha=0.15,
        )

        # ── panel 3: mean rank ───────────────────────────────────────────────
        sub3 = ranks[ranks["method"] == method].set_index("budget").reindex(budget_order)
        r    = sub3["mean_rank"].values.astype(float)
        ci3  = sub3["ci95"].values.astype(float)
        axes[2].plot(x_ticks, r, color=color, marker=marker, linewidth=2,
                     markersize=7)
        axes[2].fill_between(
            x_ticks,
            np.where(np.isnan(r - ci3), np.nan, r - ci3),
            np.where(np.isnan(r + ci3), np.nan, r + ci3),
            color=color, alpha=0.15,
        )

    # ── axis formatting ───────────────────────────────────────────────────────
    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

    SetPltProp(axes[0], xn="Given Time Budget", yn="Mean ROC AUC", legend=False)
    SetPltProp(axes[1], xn="Given Time Budget", yn="ROC AUC Wins", legend=False)
    SetPltProp(axes[2], xn="Given Time Budget", yn="Mean ROC AUC Rank", legend=False)

    # ── shared legend below all panels ───────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            color=METHOD_COLORS[m],
            marker=METHOD_MARKERS[m],
            linewidth=2,
            markersize=7,
            label=METHOD_LABELS[m],
        )
        for m in methods
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(methods),
        prop={"weight": "bold", "size": 16},
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"[saved] {output_path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ROC AUC vs. time budget for TCGA hparam search results."
    )
    parser.add_argument("--mode", choices=["collect-all", "collect-method", "plot-from-method-files"],
                        default="collect-all")
    parser.add_argument("--targets",   nargs="+", default=["OS", "PFI"],
                        choices=["OS", "PFI", "DSS", "DFI"])
    parser.add_argument("--year",      type=int,  default=2)
    parser.add_argument("--pca-dim",   type=int,  default=100)
    parser.add_argument("--cancers",   nargs="+", default=TCGA_CANCERS)
    parser.add_argument("--omics",     nargs="+", default=TCGA_OMICS)
    parser.add_argument("--n-repeats", type=int,  default=N_REPEATS)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=METHODS,
                        help="Methods to collect/plot, in display order.")
    parser.add_argument("--method", choices=ALL_METHODS, default=None,
                        help="Method to collect when --mode collect-method is used.")
    parser.add_argument("--method-output-xlsx", type=Path, default=None,
                        help="Per-method raw result Excel written by --mode collect-method.")
    parser.add_argument("--method-input-xlsx", nargs="+", type=Path, default=None,
                        help="Per-method Excel/CSV files read by --mode plot-from-method-files.")
    parser.add_argument("--combined-output-xlsx", type=Path, default=None,
                        help="Optional combined raw result Excel written in plot-from-method-files mode.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "budget_curves_pca100_afr.pdf",
    )
    parser.add_argument("--cache-csv",   type=Path, default=None,
                        help="If given, save/load the raw results DataFrame as CSV "
                             "to avoid re-scanning on repeated runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = list(args.methods)

    if args.mode == "collect-method":
        if args.method is None:
            raise ValueError("--method is required for --mode collect-method")
        if args.method_output_xlsx is None:
            raise ValueError("--method-output-xlsx is required for --mode collect-method")
        print(f"[collect] scanning result directories for method={args.method}")
        df_raw = collect_results(
            cancers   = args.cancers,
            omics_list= args.omics,
            targets   = args.targets,
            year      = args.year,
            pca_dim   = args.pca_dim,
            n_repeats = args.n_repeats,
            methods   = [args.method],
        )
        write_raw_results(df_raw, args.method_output_xlsx)
        return

    if args.mode == "plot-from-method-files":
        if not args.method_input_xlsx:
            raise ValueError("--method-input-xlsx is required for --mode plot-from-method-files")
        print("[combine] loading per-method result files")
        df_raw = read_method_result_files(args.method_input_xlsx)
        df_raw = df_raw[df_raw["method"].isin(methods)].copy()
        if args.combined_output_xlsx:
            write_raw_results(df_raw, args.combined_output_xlsx)
        if args.cache_csv:
            args.cache_csv.parent.mkdir(parents=True, exist_ok=True)
            df_raw.to_csv(args.cache_csv, index=False)
            print(f"[cache] saved to {args.cache_csv}")
    elif args.cache_csv and args.cache_csv.is_file():
        print(f"[cache] loading results from {args.cache_csv}")
        df_raw = pd.read_csv(args.cache_csv)
    else:
        print("[collect] scanning result directories …")
        df_raw = collect_results(
            cancers   = args.cancers,
            omics_list= args.omics,
            targets   = args.targets,
            year      = args.year,
            pca_dim   = args.pca_dim,
            n_repeats = args.n_repeats,
            methods   = methods,
        )
        if args.cache_csv:
            args.cache_csv.parent.mkdir(parents=True, exist_ok=True)
            df_raw.to_csv(args.cache_csv, index=False)
            print(f"[cache] saved to {args.cache_csv}")

    suffix = (
        f" | {'+'.join(args.cancers)} | {'+'.join(args.targets)} | year {args.year}"
    )
    df_raw = df_raw[df_raw["method"].isin(methods)].copy()
    make_figure(df_raw, args.output_path, methods=methods, title_suffix=suffix)


if __name__ == "__main__":
    main()
