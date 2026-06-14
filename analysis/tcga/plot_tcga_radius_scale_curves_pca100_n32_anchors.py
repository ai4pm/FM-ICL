from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


OUT_DIR = REPO_ROOT / "results"
RADIUS_SUMMARY_FILES = (
    OUT_DIR / "tables" / "tcga_radius_by_class_scales_pca100_summary.csv",
)
OUT_LONG_CSV = OUT_DIR / "tables" / "tcga_radius_scale_curves_pca100_n32_anchors_long.csv"
OUT_PNG_DIR = OUT_DIR / "figures" / "tcga_radius_scale_curves_pca100_n32_anchors_single_plots"

CANCERS = ("BRCA", "UCEC", "KIRC")
TARGETS = ("OS", "PFI")
OMICS = ("mRNA", "MicroRNA", "Methylation")
PCA_DIMS = (100,)
SMALL_RADIUS_SCALES = (0.05, 0.10, 0.15, 0.20)
ALL_RADIUS_SCALES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.20, 2.40, 2.60, 2.80, 3.00)
YEAR = 2


def SetPltProp(ax, xn=None, yn=None, title=None, grid=True, bbox_to_anchor=None, legend=True, pos="upper left", borderpad=None, ylim=None):
    fontsize = 16
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
            ax.legend(loc=pos, shadow=True, prop={"weight": "bold", "size": 10}, borderpad=borderpad)
        else:
            ax.legend(loc=pos, shadow=True, bbox_to_anchor=bbox_to_anchor, prop={"weight": "bold", "size": 10}, borderpad=borderpad)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(linewidth="1.5", linestyle="dashed")
    if xn is not None:
        ax.set_xlabel(xn, fontweight="bold")
    if yn is not None:
        ax.set_ylabel(yn, fontweight="bold")
    if title is not None:
        ax.set_title(title, fontweight="bold")
    return ax


def scale_suffix(scale: float) -> str:
    return str(scale).replace("-", "m").replace(".", "p")


def radius_result_dir(cancer: str, omics: str, target: str, pca_dim: int, scale: float) -> Path:
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


def icl_result_dir(kind: str, cancer: str, omics: str, target: str, pca_dim: int) -> Path:
    if kind == "icl-min":
        folder = "Discrete_Stractification_ICL_AI_min"
    elif kind == "icl-mixed":
        folder = "Discrete_Stractification_ICL_AI"
    else:
        raise ValueError(kind)
    stem = f"{folder}_AFR_{cancer}_{omics}_{target}_year{YEAR}_pca{pca_dim}_n32"
    return RESULT_ROOT / folder / stem


def metric_row(
    *,
    method: str,
    cancer: str,
    target: str,
    omics: str,
    pca_dim: int,
    x_label: str,
    x_order: int,
    result_dir: Path,
    label_lookup: dict[str, int],
    radius_scale: float | None = None,
) -> dict[str, object]:
    mean_auroc, std_auroc, mean_f1, std_f1, runs_evaluated = summarize_candidate_dir(result_dir, label_lookup)
    return {
        "cancer_type": cancer,
        "target": target,
        "omics_type": omics,
        "pca_dim": pca_dim,
        "method": method,
        "radius_scale": radius_scale,
        "x_label": x_label,
        "x_order": x_order,
        "mean_auroc": mean_auroc,
        "std_auroc": std_auroc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "runs_evaluated": runs_evaluated,
        "result_dir": str(result_dir),
    }


def load_existing_radius_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in RADIUS_SUMMARY_FILES:
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df[
        df["cancer_type"].isin(CANCERS)
        & df["target"].isin(TARGETS)
        & df["omics_type"].isin(OMICS)
        & df["pca_dim"].isin(PCA_DIMS)
        & df["radius_scale"].isin(ALL_RADIUS_SCALES)
        & df["runs_evaluated"].eq(30)
    ].copy()
    df["method"] = "radius_by_class"
    df["x_label"] = df["radius_scale"].map(lambda value: f"{value:g}")
    scale_order = {scale: idx + 1 for idx, scale in enumerate(ALL_RADIUS_SCALES)}
    df["x_order"] = df["radius_scale"].map(scale_order)
    keep = [
        "cancer_type",
        "target",
        "omics_type",
        "pca_dim",
        "method",
        "radius_scale",
        "x_label",
        "x_order",
        "mean_auroc",
        "std_auroc",
        "mean_f1",
        "std_f1",
        "runs_evaluated",
        "result_dir",
    ]
    return df[keep].drop_duplicates(["cancer_type", "target", "omics_type", "pca_dim", "radius_scale"], keep="last")


def build_long_table() -> pd.DataFrame:
    label_lookup = {
        (omics, target): build_label_lookup(omics, target, YEAR)
        for omics in OMICS
        for target in TARGETS
    }

    rows: list[dict[str, object]] = []
    last_order = len(ALL_RADIUS_SCALES) + 1
    for cancer in CANCERS:
        for target in TARGETS:
            for omics in OMICS:
                labels = label_lookup[(omics, target)]
                for pca_dim in PCA_DIMS:
                    rows.append(
                        metric_row(
                            method="icl-min",
                            cancer=cancer,
                            target=target,
                            omics=omics,
                            pca_dim=pca_dim,
                            x_label="FM-ICL (AFR)",
                            x_order=0,
                            result_dir=icl_result_dir("icl-min", cancer, omics, target, pca_dim),
                            label_lookup=labels,
                        )
                    )
                    for scale in SMALL_RADIUS_SCALES:
                        rows.append(
                            metric_row(
                                method="radius_by_class",
                                cancer=cancer,
                                target=target,
                                omics=omics,
                                pca_dim=pca_dim,
                                x_label=f"{scale:g}",
                                x_order=ALL_RADIUS_SCALES.index(scale) + 1,
                                result_dir=radius_result_dir(cancer, omics, target, pca_dim, scale),
                                label_lookup=labels,
                                radius_scale=scale,
                            )
                        )
                    rows.append(
                        metric_row(
                            method="icl-mixed",
                            cancer=cancer,
                            target=target,
                            omics=omics,
                            pca_dim=pca_dim,
                            x_label="FM-ICL (Mix)",
                            x_order=last_order,
                            result_dir=icl_result_dir("icl-mixed", cancer, omics, target, pca_dim),
                            label_lookup=labels,
                        )
                    )

    computed = pd.DataFrame(rows)
    existing_radius = load_existing_radius_rows()
    all_rows = pd.concat([computed, existing_radius], ignore_index=True)
    all_rows = all_rows.drop_duplicates(
        ["cancer_type", "target", "omics_type", "pca_dim", "method", "radius_scale", "x_order"],
        keep="last",
    )
    all_rows = all_rows.sort_values(["cancer_type", "target", "omics_type", "pca_dim", "x_order"]).reset_index(drop=True)
    return all_rows


def plot_single_curve(df: pd.DataFrame, cancer: str, target: str, omics: str, pca_dim: int) -> plt.Figure:
    panel = df[
        df["cancer_type"].eq(cancer)
        & df["target"].eq(target)
        & df["omics_type"].eq(omics)
        & df["pca_dim"].eq(pca_dim)
        & df["runs_evaluated"].eq(30)
    ].sort_values("x_order")
    x_labels = ["FM-ICL (AFR)", *[f"{scale:g}" for scale in ALL_RADIUS_SCALES], "FM-ICL (Mix)"]
    x_orders = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    if not panel.empty:
        ax.errorbar(
            panel["x_order"],
            panel["mean_auroc"],
            yerr=panel["std_auroc"],
            color="#0000ff",
            marker="o",
            markersize=16,
            linewidth=1.8,
            elinewidth=2.0,
            capsize=16,
            capthick=2.0,
        )
        anchors = panel[panel["method"].isin(["icl-min", "icl-mixed"])]
        ax.errorbar(
            anchors["x_order"],
            anchors["mean_auroc"],
            yerr=anchors["std_auroc"],
            linestyle="none",
            marker="o",
            color="#ff0000",
            markersize=16,
            elinewidth=2.0,
            capsize=16,
            capthick=2.0,
            zorder=4,
        )
    ax.tick_params(axis="y", labelleft=True)
    ax.set_xticks(x_orders)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=16)
    for label in ax.get_xticklabels():
        if label.get_text() in {"FM-ICL (AFR)", "FM-ICL (Mix)"}:
            label.set_fontweight("bold")
    SetPltProp(
        ax,
        xn=None,
        yn="AUROC",
        title=None,
        legend=False,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_LONG_CSV.is_file():
        df = pd.read_csv(OUT_LONG_CSV)
        df.loc[df["method"].eq("icl-min"), "x_label"] = "FM-ICL (AFR)"
        df.loc[df["method"].eq("icl-mixed"), "x_label"] = "FM-ICL (Mix)"
    else:
        df = build_long_table()
    df.to_csv(OUT_LONG_CSV, index=False)

    complete = df[df["runs_evaluated"].eq(30)]
    expected = len(CANCERS) * len(TARGETS) * len(OMICS) * len(PCA_DIMS) * (len(ALL_RADIUS_SCALES) + 2)
    print(f"Saved long table: {OUT_LONG_CSV}")
    print(f"Complete plot points: {len(complete)}/{expected}")

    for cancer in CANCERS:
        for target in TARGETS:
            for omics in OMICS:
                for pca_dim in PCA_DIMS:
                    fig = plot_single_curve(df, cancer, target, omics, pca_dim)
                    output_name = f"tcga_radius_scale_curve_{cancer}_{target}_year{YEAR}_{omics}_pca{pca_dim}.png"
                    fig.savefig(OUT_PNG_DIR / output_name, dpi=220, bbox_inches="tight")
                    plt.close(fig)
    print(f"Saved PNG directory: {OUT_PNG_DIR}")

    missing = df[~df["runs_evaluated"].eq(30)]
    if not missing.empty:
        print("\nIncomplete/missing points:")
        print(
            missing[
                ["cancer_type", "target", "omics_type", "pca_dim", "method", "radius_scale", "x_label", "runs_evaluated"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
