from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TABLE_DIR = REPO_ROOT / "results" / "tables"
FIGURE_DIR = REPO_ROOT / "results" / "figures"

TCGA_INPUT = Path(os.environ.get("FM_ICL_DACE_DELTA_INPUT", REPO_ROOT / "data" / "tcga_onepass_delta_by_afr_train.csv"))
COMBINED_OUTPUT = TABLE_DIR / "tcga_dace_delta_by_afr_train_with_external.csv"
FIT_OUTPUT = TABLE_DIR / "tcga_dace_delta_by_afr_train_vs_fmicl_polynomial_fit.csv"
PNG_OUTPUT = FIGURE_DIR / "tcga_dace_delta_by_afr_train_vs_fmicl_polynomial_fit.png"
PDF_OUTPUT = FIGURE_DIR / "tcga_dace_delta_by_afr_train_vs_fmicl_polynomial_fit.pdf"
TOPMED_COMBINED_OUTPUT = TABLE_DIR / "tcga_dace_delta_by_afr_train_with_topmed.csv"
TOPMED_FIT_OUTPUT = TABLE_DIR / "tcga_dace_delta_by_afr_train_with_topmed_linear_fit.csv"
TOPMED_PNG_OUTPUT = FIGURE_DIR / "tcga_dace_delta_by_afr_train_with_topmed_linear_fit.png"
TOPMED_PDF_OUTPUT = FIGURE_DIR / "tcga_dace_delta_by_afr_train_with_topmed_linear_fit.pdf"

LEGACY_ROOT = Path(os.environ.get("FM_ICL_LEGACY_ANALYTICS_ROOT", "/nfs/home/wli66/tabular_transformer_transfer_learning"))
TCGA_CLASSICAL_RUN_LEVEL = Path(os.environ.get("FM_ICL_TCGA_CLASSICAL_RUN_LEVEL", LEGACY_ROOT / "data_analytics/tcga/hparam_pca100_budget30s_icl_comparisons/run_level_recomputed_metrics.csv"))
TCGA_CLASSICAL_FALLBACK_RUN_LEVEL = Path(os.environ.get("FM_ICL_TCGA_CLASSICAL_FALLBACK_RUN_LEVEL", LEGACY_ROOT / "data_analytics/tcga/tcga_hparam_budget30s_icl_classical_run_level_metrics_with_agg.csv"))
TCGA_TASK_BY_METHOD_WORKBOOK = Path(os.environ.get("FM_ICL_TCGA_TASK_BY_METHOD_WORKBOOK", LEGACY_ROOT / "data_analytics/tcga/tcga_hparam_budget30s_task_by_method_tables/tcga_budget30s_task_by_method_tables.xlsx"))
TCGA_TRANSFER_LEARNING_RUN_LEVEL = Path(os.environ.get("FM_ICL_TCGA_TRANSFER_LEARNING_RUN_LEVEL", LEGACY_ROOT / "data_analytics/tcga/budget_curve_method_excels/budget_curves_raw_pca100_ds_tl.csv"))
GENEVA_SUMMARY = Path(os.environ.get("FM_ICL_GENEVA_SUMMARY", LEGACY_ROOT / "data_analytics/geneva/afr_hparam_eval/geneva_afr_hparam_selected_summary.csv"))
ONCOARRAY_SUMMARY = Path(os.environ.get("FM_ICL_ONCOARRAY_SUMMARY", LEGACY_ROOT / "data_analytics/oncoarray/afr_hparam_eval/oncoarray_afr_hparam_selected_summary.csv"))
TOPMED_SUMMARY = Path(os.environ.get("FM_ICL_TOPMED_SUMMARY", LEGACY_ROOT / "data_analytics/topmed/afr_hparam_eval/topmed_afr_hparam_selected_summary.csv"))
POLYNOMIAL_DEGREE = 2
TCGA_BUDGET = "budget30s"
TOPMED_TASK = "TOPMed-COPD"
TOPMED_FEATURE_MODE = "pca400"
TOPMED_AFR_TRAIN = 1771
TOPMED_AFR_VAL = 443

COMPARISON_MIN_LABEL = "FM-ICL (DACE) - compared approaches (AFR)"
COMPARISON_MIXED_LABEL = "FM-ICL (DACE) - compared approaches (Mix)"
TRANSFER_LEARNING_LABEL = "FM-ICL (DACE) - Transfer Learning"
ELASTICNET_MIN_LABEL = "FM-ICL (DACE) - Elastic Net (AFR)"
ELASTICNET_MIXED_LABEL = "FM-ICL (DACE) - Elastic Net (Mix)"
RANDOMFOREST_MIN_LABEL = "FM-ICL (DACE) - Random Forest (AFR)"
RANDOMFOREST_MIXED_LABEL = "FM-ICL (DACE) - Random Forest (Mix)"
XGBOOST_MIN_LABEL = "FM-ICL (DACE) - XGBoost (AFR)"
XGBOOST_MIXED_LABEL = "FM-ICL (DACE) - XGBoost (Mix)"

COMPARISON_STYLES = {
    COMPARISON_MIN_LABEL: {
        "color": "#0000FF",
        "linestyle": "-",
        "delta_column": "delta_onepass_minus_min",
    },
    COMPARISON_MIXED_LABEL: {
        "color": "#FF0000",
        "linestyle": "--",
        "delta_column": "delta_onepass_minus_mixed",
    },
}
BLUE_STYLE = {"color": "#0000FF", "linestyle": "-"}
RED_STYLE = {"color": "#FF0000", "linestyle": "--"}
ADDITIONAL_BASELINES = {
    "ds_tl": {
        "mean_column": "transfer_learning_mean",
        "n_column": "n_transfer_learning",
        "delta_column": "delta_ce_dace_minus_transfer_learning",
    },
    "elasticnet_min": {
        "mean_column": "elasticnet_min_mean",
        "n_column": "n_elasticnet_min",
        "delta_column": "delta_ce_dace_minus_elasticnet_min",
    },
    "elasticnet_mixed": {
        "mean_column": "elasticnet_mixed_mean",
        "n_column": "n_elasticnet_mixed",
        "delta_column": "delta_ce_dace_minus_elasticnet_mixed",
    },
    "randomforest_min": {
        "mean_column": "randomforest_min_mean",
        "n_column": "n_randomforest_min",
        "delta_column": "delta_ce_dace_minus_randomforest_min",
    },
    "randomforest_mixed": {
        "mean_column": "randomforest_mixed_mean",
        "n_column": "n_randomforest_mixed",
        "delta_column": "delta_ce_dace_minus_randomforest_mixed",
    },
    "xgboost_min": {
        "mean_column": "xgboost_min_mean",
        "n_column": "n_xgboost_min",
        "delta_column": "delta_ce_dace_minus_xgboost_min",
    },
    "xgboost_mixed": {
        "mean_column": "xgboost_mixed_mean",
        "n_column": "n_xgboost_mixed",
        "delta_column": "delta_ce_dace_minus_xgboost_mixed",
    },
}
TCGA_WORKBOOK_METHOD_LABELS = {
    "elasticnet_min": "ElasticNet Min",
    "elasticnet_mixed": "ElasticNet Mixed",
    "randomforest_min": "RandomForest Min",
    "randomforest_mixed": "RandomForest Mixed",
    "xgboost_min": "XGBoost Min",
    "xgboost_mixed": "XGBoost Mixed",
}
ADDITIONAL_PLOTS = [
    {
        "stem": "tcga_onepass_delta_by_afr_train_with_external_vs_transfer_learning_polynomial_fit",
        "title": "FM-ICL (DACE) vs Transfer Learning",
        "styles": {
            TRANSFER_LEARNING_LABEL: {
                **BLUE_STYLE,
                "delta_column": "delta_ce_dace_minus_transfer_learning",
            },
        },
    },
    {
        "stem": "tcga_onepass_delta_by_afr_train_with_external_vs_elasticnet_polynomial_fit",
        "title": "FM-ICL (DACE) vs Elastic Net",
        "styles": {
            ELASTICNET_MIN_LABEL: {
                **BLUE_STYLE,
                "delta_column": "delta_ce_dace_minus_elasticnet_min",
            },
            ELASTICNET_MIXED_LABEL: {
                **RED_STYLE,
                "delta_column": "delta_ce_dace_minus_elasticnet_mixed",
            },
        },
    },
    {
        "stem": "tcga_onepass_delta_by_afr_train_with_external_vs_randomforest_polynomial_fit",
        "title": "FM-ICL (DACE) vs Random Forest",
        "styles": {
            RANDOMFOREST_MIN_LABEL: {
                **BLUE_STYLE,
                "delta_column": "delta_ce_dace_minus_randomforest_min",
            },
            RANDOMFOREST_MIXED_LABEL: {
                **RED_STYLE,
                "delta_column": "delta_ce_dace_minus_randomforest_mixed",
            },
        },
    },
    {
        "stem": "tcga_onepass_delta_by_afr_train_with_external_vs_xgboost_polynomial_fit",
        "title": "FM-ICL (DACE) vs XGBoost",
        "styles": {
            XGBOOST_MIN_LABEL: {
                **BLUE_STYLE,
                "delta_column": "delta_ce_dace_minus_xgboost_min",
            },
            XGBOOST_MIXED_LABEL: {
                **RED_STYLE,
                "delta_column": "delta_ce_dace_minus_xgboost_mixed",
            },
        },
    },
]
TOPMED_ADDITIONAL_PLOTS = [
    {
        **plot_spec,
        "stem": str(plot_spec["stem"]).replace("with_external", "with_topmed"),
    }
    for plot_spec in ADDITIONAL_PLOTS
]
TASK_MARKERS = [
    "o",
    "s",
    "^",
    "v",
    "D",
    "P",
    "X",
    "*",
    "h",
    "H",
    "<",
    ">",
    "p",
    "8",
    "d",
]


def task_marker_map(tasks: list[str]) -> dict[str, str]:
    return {task: TASK_MARKERS[idx % len(TASK_MARKERS)] for idx, task in enumerate(tasks)}


def selected_row(df: pd.DataFrame, **conditions: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, expected in conditions.items():
        mask &= df[column].astype(str).eq(str(expected))
    rows = df.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row for {conditions}, found {len(rows)}")
    return rows.iloc[0]


def make_external_row(
    dataset: str,
    task: str,
    feature_mode: str,
    afr_train: int,
    afr_val: int,
    onepass_row: pd.Series,
    min_row: pd.Series,
    mixed_row: pd.Series,
) -> dict[str, object]:
    onepass_mean = float(onepass_row["mean_auroc"])
    min_mean = float(min_row["mean_auroc"])
    mixed_mean = float(mixed_row["mean_auroc"])
    return {
        "dataset": dataset,
        "feature_mode": feature_mode,
        "task": task,
        "cancer": dataset,
        "endpoint": "external",
        "omics": feature_mode,
        "afr_train": afr_train,
        "afr_val": afr_val,
        "onepass_mean": onepass_mean,
        "icl_min_mean": min_mean,
        "icl_mixed_mean": mixed_mean,
        "delta_onepass_minus_min": onepass_mean - min_mean,
        "delta_onepass_minus_mixed": onepass_mean - mixed_mean,
        "n_onepass": int(onepass_row["runs_evaluated"]),
        "n_min": int(min_row["runs_evaluated"]),
        "n_mixed": int(mixed_row["runs_evaluated"]),
    }


def build_external_rows(include_topmed: bool = False) -> pd.DataFrame:
    geneva = pd.read_csv(GENEVA_SUMMARY)
    oncoarray = pd.read_csv(ONCOARRAY_SUMMARY)
    topmed = pd.read_csv(TOPMED_SUMMARY) if include_topmed else None

    rows = []

    geneva_min = selected_row(geneva, feature_mode="pca400_clin57_ctxpca20", raw_method="icl_min")
    geneva_mixed = selected_row(geneva, feature_mode="pca400_clin57_ctxpca20", raw_method="icl_mixed")
    geneva_onepass = selected_row(
        geneva,
        feature_mode="pca400_clin57_ctxae20",
        raw_method="icl_one_pass_by_class_ae20",
    )
    rows.append(
        make_external_row(
            dataset="GENEVA",
            task="GENEVA",
            feature_mode="pca400+clin57+ctxae20",
            afr_train=420,
            afr_val=105,
            onepass_row=geneva_onepass,
            min_row=geneva_min,
            mixed_row=geneva_mixed,
        )
    )

    for feature_mode, label in [
        ("pca300", "OncoArray-Omics"),
        ("pca300_clinagesmoke", "OncoArray-Omics+Clin"),
    ]:
        rows.append(
            make_external_row(
                dataset=label,
                task=label,
                feature_mode=feature_mode,
                afr_train=84,
                afr_val=22,
                onepass_row=selected_row(
                    oncoarray,
                    feature_mode=feature_mode,
                    raw_method="icl_one_pass_by_class",
                ),
                min_row=selected_row(oncoarray, feature_mode=feature_mode, raw_method="icl_min"),
                mixed_row=selected_row(oncoarray, feature_mode=feature_mode, raw_method="icl_mixed"),
            )
        )

    if include_topmed:
        if topmed is None:
            raise ValueError("TOPMed summary was not loaded.")
        rows.append(
            make_external_row(
                dataset="TOPMed",
                task=TOPMED_TASK,
                feature_mode=TOPMED_FEATURE_MODE,
                afr_train=TOPMED_AFR_TRAIN,
                afr_val=TOPMED_AFR_VAL,
                onepass_row=selected_row(topmed, raw_method="icl_onepass_by_class_fold0"),
                min_row=selected_row(topmed, raw_method="icl_min"),
                mixed_row=selected_row(topmed, raw_method="icl_mixed"),
            )
        )

    return pd.DataFrame(rows)


def validate_columns(data: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = sorted(columns.difference(data.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def tcga_task_name(data: pd.DataFrame) -> pd.Series:
    return (
        data["cancer"].astype(str)
        + "-"
        + data["target"].astype(str)
        + "-"
        + data["omics"].astype(str)
    )


def load_tcga_baseline_means() -> dict[tuple[str, str], tuple[float, int]]:
    classical = pd.read_csv(TCGA_CLASSICAL_RUN_LEVEL)
    validate_columns(classical, TCGA_CLASSICAL_RUN_LEVEL, {"method", "task", "budget", "auroc"})
    raw_methods = [method for method in ADDITIONAL_BASELINES if method != "ds_tl"]
    classical = classical.loc[
        classical["method"].isin(raw_methods) & classical["budget"].astype(str).eq(TCGA_BUDGET)
    ].copy()
    classical["auroc"] = pd.to_numeric(classical["auroc"], errors="coerce")

    transfer = pd.read_csv(TCGA_TRANSFER_LEARNING_RUN_LEVEL)
    validate_columns(
        transfer,
        TCGA_TRANSFER_LEARNING_RUN_LEVEL,
        {"method", "cancer", "target", "omics", "budget", "auroc"},
    )
    transfer = transfer.loc[
        transfer["method"].astype(str).eq("ds_tl") & transfer["budget"].astype(str).eq(TCGA_BUDGET)
    ].copy()
    transfer["task"] = tcga_task_name(transfer)
    transfer["auroc"] = pd.to_numeric(transfer["auroc"], errors="coerce")
    transfer["method"] = "ds_tl"

    baselines = pd.concat(
        [
            classical[["task", "method", "auroc"]],
            transfer[["task", "method", "auroc"]],
        ],
        ignore_index=True,
    )
    summary = (
        baselines.dropna(subset=["task", "method", "auroc"])
        .groupby(["task", "method"], observed=True)["auroc"]
        .agg(mean="mean", n="count")
        .reset_index()
    )
    result = {
        (str(row["task"]), str(row["method"])): (float(row["mean"]), int(row["n"]))
        for row in summary.to_dict("records")
    }

    workbook_means = pd.read_excel(TCGA_TASK_BY_METHOD_WORKBOOK, sheet_name="auroc_mean")
    validate_columns(workbook_means, TCGA_TASK_BY_METHOD_WORKBOOK, {"method"})
    workbook_tasks = [column for column in workbook_means.columns if column != "method"]
    for raw_method, workbook_label in TCGA_WORKBOOK_METHOD_LABELS.items():
        rows = workbook_means.loc[workbook_means["method"].astype(str).eq(workbook_label)]
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one workbook row for {workbook_label}, found {len(rows)}"
            )
        row = rows.iloc[0]
        for task in workbook_tasks:
            value = pd.to_numeric(pd.Series([row[task]]), errors="coerce").iloc[0]
            if pd.notna(value):
                result.setdefault((task, raw_method), (float(value), 30))

    fallback = pd.read_csv(TCGA_CLASSICAL_FALLBACK_RUN_LEVEL)
    validate_columns(
        fallback,
        TCGA_CLASSICAL_FALLBACK_RUN_LEVEL,
        {"method", "cancer", "target", "omics", "budget", "auroc"},
    )
    fallback = fallback.loc[
        fallback["method"].isin(raw_methods) & fallback["budget"].astype(str).eq(TCGA_BUDGET)
    ].copy()
    fallback["task"] = tcga_task_name(fallback)
    fallback["auroc"] = pd.to_numeric(fallback["auroc"], errors="coerce")
    fallback_summary = (
        fallback.dropna(subset=["task", "method", "auroc"])
        .groupby(["task", "method"], observed=True)["auroc"]
        .agg(mean="mean", n="count")
        .reset_index()
    )
    for row in fallback_summary.to_dict("records"):
        key = (str(row["task"]), str(row["method"]))
        result.setdefault(key, (float(row["mean"]), int(row["n"])))

    return result


def external_feature_mode(row: pd.Series) -> str:
    task = str(row["task"])
    if task == "GENEVA":
        return "pca400_clin57_ctxpca20"
    if task == "OncoArray-Omics":
        return "pca300"
    if task == "OncoArray-Omics+Clin":
        return "pca300_clinagesmoke"
    if task == TOPMED_TASK:
        return TOPMED_FEATURE_MODE
    raise ValueError(f"Unexpected external task: {task}")


def load_external_baseline_means(include_topmed: bool = False) -> dict[tuple[str, str], tuple[float, int]]:
    summaries = {
        "GENEVA": pd.read_csv(GENEVA_SUMMARY),
        "OncoArray-Omics": pd.read_csv(ONCOARRAY_SUMMARY),
        "OncoArray-Omics+Clin": pd.read_csv(ONCOARRAY_SUMMARY),
    }
    if include_topmed:
        summaries[TOPMED_TASK] = pd.read_csv(TOPMED_SUMMARY)
    result = {}
    for task, data in summaries.items():
        if task == TOPMED_TASK:
            for raw_method in ADDITIONAL_BASELINES:
                baseline_row = selected_row(data, raw_method=raw_method)
                result[(task, raw_method)] = (
                    float(baseline_row["mean_auroc"]),
                    int(baseline_row["runs_evaluated"]),
                )
            continue

        feature_mode = {
            "GENEVA": "pca400_clin57_ctxpca20",
            "OncoArray-Omics": "pca300",
            "OncoArray-Omics+Clin": "pca300_clinagesmoke",
        }[task]
        for raw_method in ADDITIONAL_BASELINES:
            baseline_row = selected_row(data, feature_mode=feature_mode, raw_method=raw_method)
            result[(task, raw_method)] = (
                float(baseline_row["mean_auroc"]),
                int(baseline_row["runs_evaluated"]),
            )
    return result


def add_additional_baselines(combined: pd.DataFrame, include_topmed: bool = False) -> pd.DataFrame:
    combined = combined.copy()
    for baseline in ADDITIONAL_BASELINES.values():
        combined[baseline["mean_column"]] = np.nan
        combined[baseline["n_column"]] = pd.Series(pd.NA, index=combined.index, dtype="Int64")
        combined[baseline["delta_column"]] = np.nan

    tcga_means = load_tcga_baseline_means()
    external_means = load_external_baseline_means(include_topmed=include_topmed)

    for index, row in combined.iterrows():
        task = str(row["task"])
        source = tcga_means if str(row["dataset"]) == "TCGA" else external_means
        for raw_method, baseline in ADDITIONAL_BASELINES.items():
            key = (task, raw_method)
            if key not in source:
                raise ValueError(f"Missing baseline AUROC for {task} / {raw_method}")
            baseline_mean, baseline_n = source[key]
            combined.at[index, baseline["mean_column"]] = baseline_mean
            combined.at[index, baseline["n_column"]] = baseline_n
            combined.at[index, baseline["delta_column"]] = float(row["onepass_mean"]) - baseline_mean

    return combined


def polynomial_fit(
    df: pd.DataFrame,
    delta_column: str,
    comparison: str,
    x_transform_label: str = "log10(afr_train)",
) -> dict[str, object]:
    fit_df = df[["afr_train", delta_column]].dropna()
    x = np.log10(fit_df["afr_train"].to_numpy(dtype=float))
    y = fit_df[delta_column].to_numpy(dtype=float)
    coefficients = np.polyfit(x, y, deg=POLYNOMIAL_DEGREE)
    predicted = np.polyval(coefficients, x)
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else np.nan
    row = {
        "comparison": comparison,
        "x_transform": x_transform_label,
        "fit_type": "polynomial",
        "degree": POLYNOMIAL_DEGREE,
        "r2": r2,
        "n": len(fit_df),
    }
    for power, coefficient in zip(range(POLYNOMIAL_DEGREE, -1, -1), coefficients):
        row[f"coef_x{power}"] = float(coefficient)
    return row


def plot_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    markers_by_task: dict[str, str],
    comparison_styles: dict[str, dict[str, str]],
) -> None:
    for _, row in df.iterrows():
        task = str(row["task"])
        marker = markers_by_task[task]
        x = np.log10(float(row["afr_train"]))
        size = 210
        dataset = str(row["dataset"])
        zorder = 4 if dataset == "TCGA" else 5
        for comparison, style in comparison_styles.items():
            ax.scatter(
                x,
                row[style["delta_column"]],
                s=size,
                marker=marker,
                facecolor=style["color"],
                edgecolor="black",
                linewidth=1.35,
                alpha=0.88,
                zorder=zorder,
            )


def plot_fit_lines(
    ax: plt.Axes,
    fit_rows: list[dict[str, object]],
    x_min: float,
    x_max: float,
    comparison_styles: dict[str, dict[str, str]],
) -> None:
    x_grid = np.linspace(x_min, x_max, 300)
    for row in fit_rows:
        coefficients = [float(row[f"coef_x{power}"]) for power in range(int(row["degree"]), -1, -1)]
        y_grid = np.polyval(coefficients, x_grid)
        style = comparison_styles[str(row["comparison"])]
        ax.plot(
            x_grid,
            y_grid,
            color=style["color"],
            linewidth=3.0,
            linestyle=style["linestyle"],
            label=f"Polynomial fit: {row['comparison']}",
            zorder=2,
        )


def add_legends(
    ax: plt.Axes,
    tasks: list[str],
    markers_by_task: dict[str, str],
    comparison_styles: dict[str, dict[str, str]],
) -> None:
    comparison_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=style["color"],
            linewidth=3.0,
            linestyle=style["linestyle"],
            markerfacecolor=style["color"],
            markeredgecolor="black",
            markersize=13,
            label=label,
        )
        for label, style in comparison_styles.items()
    ]
    task_handles = [
        Line2D(
            [0],
            [0],
            marker=markers_by_task[task],
            color="none",
            markerfacecolor="#D1D5DB",
            markeredgecolor="black",
            markeredgewidth=1.4,
            markersize=13,
            label=task,
        )
        for task in tasks
    ]

    ax.legend(
        handles=comparison_handles + task_handles,
        loc="upper center",
        bbox_to_anchor=(0.0, -0.28, 1.0, 0.1),
        mode="expand",
        borderaxespad=0.0,
        frameon=True,
        title="Task names",
        ncol=4,
        prop=FontProperties(size=14, weight="bold"),
        title_fontproperties=FontProperties(size=17, weight="bold"),
        handlelength=2.4,
        handletextpad=0.8,
        columnspacing=1.2,
        labelspacing=0.8,
    )


def polish_axes(ax: plt.Axes) -> None:
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.2)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight("bold")
    ax.grid(axis="y", linewidth=1.1, linestyle="dashed", alpha=0.7)
    ax.grid(axis="x", linewidth=0.8, linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)


def build_plot(
    combined: pd.DataFrame,
    fit_rows: list[dict[str, object]],
    comparison_styles: dict[str, dict[str, str]],
    title: str,
    x_axis_label: str,
    png_output: Path,
    pdf_output: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(20, 13))
    x_values = np.log10(combined["afr_train"].to_numpy(dtype=float))
    x_pad = max(0.05, 0.06 * (float(x_values.max()) - float(x_values.min())))
    x_min = float(x_values.min() - x_pad)
    x_max = float(x_values.max() + x_pad)
    tasks = list(dict.fromkeys(combined["task"].astype(str)))
    markers_by_task = task_marker_map(tasks)

    plot_fit_lines(ax, fit_rows, x_min=x_min, x_max=x_max, comparison_styles=comparison_styles)
    plot_points(ax, combined, markers_by_task, comparison_styles)

    ax.axhline(0, color="black", linewidth=2.0, linestyle=":", zorder=1)
    if title:
        ax.text(
            0.02,
            0.96,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=24,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=10,
        )
    ax.set_xlabel(x_axis_label, fontsize=21, fontweight="bold")
    ax.set_ylabel("AUROC difference", fontsize=21, fontweight="bold")
    ax.set_xlim(x_min, x_max)
    y_columns = [style["delta_column"] for style in comparison_styles.values()]
    y_min = min(combined[column].min() for column in y_columns)
    y_max = max(combined[column].max() for column in y_columns)
    y_pad = max(0.03, 0.12 * (y_max - y_min))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    polish_axes(ax)

    add_legends(ax, tasks, markers_by_task, comparison_styles)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.46, top=0.98)
    fig.savefig(png_output, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_output, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def build_combined(include_topmed: bool = False) -> pd.DataFrame:
    tcga = pd.read_csv(TCGA_INPUT)
    tcga.insert(0, "dataset", "TCGA")
    tcga.insert(1, "feature_mode", "pca100")

    external = build_external_rows(include_topmed=include_topmed)
    combined = pd.concat([tcga, external], ignore_index=True, sort=False)
    return add_additional_baselines(combined, include_topmed=include_topmed)


def write_output_set(
    combined: pd.DataFrame,
    combined_output: Path,
    fit_output: Path,
    png_output: Path,
    pdf_output: Path,
    additional_plots: list[dict[str, object]],
) -> None:
    combined.to_csv(combined_output, index=False)
    fit_rows = [
        polynomial_fit(
            combined,
            "delta_onepass_minus_min",
            COMPARISON_MIN_LABEL,
            x_transform_label="log10(afr_context_sample_size)",
        ),
        polynomial_fit(
            combined,
            "delta_onepass_minus_mixed",
            COMPARISON_MIXED_LABEL,
            x_transform_label="log10(afr_context_sample_size)",
        ),
    ]
    pd.DataFrame(fit_rows).to_csv(fit_output, index=False)

    build_plot(
        combined,
        fit_rows,
        comparison_styles=COMPARISON_STYLES,
        title="FM-ICL (DACE) vs FM-ICL baselines",
        x_axis_label="AFR context sample size (log scale)",
        png_output=png_output,
        pdf_output=pdf_output,
    )

    for plot_spec in additional_plots:
        comparison_styles = plot_spec["styles"]
        extra_fit_rows = [
            polynomial_fit(combined, style["delta_column"], comparison)
            for comparison, style in comparison_styles.items()
        ]
        fit_output = TABLE_DIR / f"{plot_spec['stem']}.csv"
        png_output = FIGURE_DIR / f"{plot_spec['stem']}.png"
        pdf_output = FIGURE_DIR / f"{plot_spec['stem']}.pdf"
        pd.DataFrame(extra_fit_rows).to_csv(fit_output, index=False)
        build_plot(
            combined,
            extra_fit_rows,
            comparison_styles=comparison_styles,
            title=str(plot_spec["title"]),
            x_axis_label="AFR training sample size (log scale)",
            png_output=png_output,
            pdf_output=pdf_output,
        )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    combined = build_combined(include_topmed=False)
    write_output_set(
        combined,
        combined_output=COMBINED_OUTPUT,
        fit_output=FIT_OUTPUT,
        png_output=PNG_OUTPUT,
        pdf_output=PDF_OUTPUT,
        additional_plots=ADDITIONAL_PLOTS,
    )


if __name__ == "__main__":
    main()
