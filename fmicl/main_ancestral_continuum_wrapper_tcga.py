from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FMICL_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, FMICL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tcga_ancestral_continuum_networks import (
    develop_Cont_Enhanced_ICL_AI_raw_sample_split,
    develop_DS_Enhanced_ICL_AI_raw_sample_split,
    develop_DS_ElasticNetSelected_ICL_AI_raw_sample_split,
    develop_DS_ElasticNetSelected_Enhanced_ICL_AI_raw_sample_split,
    develop_DS_ICL_AI_raw_sample_split,
    develop_DS_TL_AI_for_raw_sample_split,
    develop_merge_ancestry_omics_ICL_AI_raw_sample_split,
)
from tcga_classical_networks import (
    develop_DS_ElasticNet_AI_raw_sample_split,
    develop_DS_Enhanced_ElasticNet_AI_raw_sample_split,
    develop_DS_Enhanced_RandomForest_AI_raw_sample_split,
    develop_DS_Enhanced_XGBoost_AI_raw_sample_split,
    develop_DS_RandomForest_AI_raw_sample_split,
    develop_DS_XGBoost_AI_raw_sample_split,
    develop_merge_ancestry_omics_ElasticNet_AI_raw_sample_split,
    develop_merge_ancestry_omics_RandomForest_AI_raw_sample_split,
    develop_merge_ancestry_omics_XGBoost_AI_raw_sample_split,
)


DATA_ROOT = Path(
    os.environ.get(
        "FM_ICL_DATA_ROOT",
        REPO_ROOT / "data" / "tcga",
    )
)
RESULT_ROOT = Path(
    os.environ.get(
        "FM_ICL_RESULT_ROOT",
        "/lustre/isaac24/scratch/wli66/tabular_transformer_transfer_learning/Result/TCGA",
    )
)
RESULT_SUFFIX = os.environ.get("TCGA_RESULT_SUFFIX", "").strip()

DISCRETE_MODES = {
    "Discrete_Stractification_ICL_AI": ("icl", "mix"),
    "Discrete_Stractification_ICL_AI_min": ("icl", "min"),
    "Discrete_Stractification_ICL_AI_maj": ("icl", "maj"),
    "Discrete_Stractification_ElasticNet_AI": ("elasticnet", "mix"),
    "Discrete_Stractification_ElasticNet_AI_min": ("elasticnet", "min"),
    "Discrete_Stractification_ElasticNet_AI_maj": ("elasticnet", "maj"),
    "Discrete_Stractification_RandomForest_AI": ("randomforest", "mix"),
    "Discrete_Stractification_RandomForest_AI_min": ("randomforest", "min"),
    "Discrete_Stractification_RandomForest_AI_maj": ("randomforest", "maj"),
    "Discrete_Stractification_XGBoost_AI": ("xgboost", "mix"),
    "Discrete_Stractification_XGBoost_AI_min": ("xgboost", "min"),
    "Discrete_Stractification_XGBoost_AI_maj": ("xgboost", "maj"),
}

ELASTICNET_SELECTED_ICL_MODES = {
    "Discrete_Stractification_ElasticNetSelected_ICL_AI": "mix",
    "Discrete_Stractification_ElasticNetSelected_ICL_AI_min": "min",
    "Discrete_Stractification_ElasticNetSelected_ICL_AI_maj": "maj",
}

ELASTICNET_SELECTED_ENHANCED_ICL_MODES = {
    "Discrete_Stractification_ElasticNetSelected_Enhanced_ICL_AI": "mix",
    "Discrete_Stractification_ElasticNetSelected_Enhanced_ICL_AI_min": "min",
}

MERGE_MODES = {
    "merge_ancestry_omics_ICL_AI": ("icl", "mix", False),
    "merge_ancestry_omics_ICL_AI_min": ("icl", "min", False),
    "merge_ancestry_omics_ICL_AI_maj": ("icl", "maj", False),
    "merge_ancestry_omics_weight_use_ICL_AI": ("icl", "mix", True),
    "merge_ancestry_omics_weight_use_ICL_AI_min": ("icl", "min", True),
    "merge_ancestry_omics_weight_use_ICL_AI_maj": ("icl", "maj", True),
    "merge_ancestry_omics_ElasticNet_AI": ("elasticnet", "mix", False),
    "merge_ancestry_omics_RandomForest_AI": ("randomforest", "mix", False),
    "merge_ancestry_omics_XGBoost_AI": ("xgboost", "mix", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TCGA-only wrapper for ancestral continuum jobs.")
    parser.add_argument("--data_type", default="TCGA")
    parser.add_argument("--target", choices=["OS", "PFI", "DSS", "DFI"], required=True)
    parser.add_argument("--expression_type", "--expression-type", dest="expression_type", choices=["mRNA", "MicroRNA", "Methylation"], required=True)
    parser.add_argument("--modeling_approach", required=True)
    parser.add_argument("--years", type=int, required=True)
    parser.add_argument("--ancestry", choices=["AFR", "EUR"], required=True)
    parser.add_argument("--cancer_type", "--cancer-type", dest="cancer_type", required=True)
    parser.add_argument("--tcga_omics_feature_source", "--tcga-omics-feature-source", dest="tcga_omics_feature_source", choices=["raw", "autoencoder", "pca"], default="raw")
    parser.add_argument("--tcga_autoencoder_dim", "--tcga-autoencoder-dim", "--latent-dim", dest="tcga_autoencoder_dim", type=int, default=20)
    parser.add_argument("--tcga_pca_dim", "--tcga-pca-dim", "--pca-dim", dest="tcga_pca_dim", type=int, default=400)
    parser.add_argument("--tcga_context_autoencoder_dim", "--tcga-context-autoencoder-dim", "--context-autoencoder-dim", dest="tcga_context_autoencoder_dim", type=int, default=20)
    parser.add_argument("--experimental_runs", "--experimental-runs", dest="experimental_runs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold_index_only", "--fold-index-only", dest="fold_index_only", type=int, default=None)
    parser.add_argument("--tabpfn_n_estimators", "--tabpfn-n-estimators", dest="tabpfn_n_estimators", type=int, default=32)
    parser.add_argument("--tabpfn_softmax_temperature", "--tabpfn-softmax-temperature", dest="tabpfn_softmax_temperature", type=float, default=0.9)
    parser.add_argument("--tabpfn_balance_probabilities", "--tabpfn-balance-probabilities", dest="tabpfn_balance_probabilities", action="store_true")
    parser.add_argument("--distance_metric", "--distance-metric", dest="distance_metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--context_search_mode", "--context-search-mode", dest="context_search_mode", choices=["batch", "one_pass", "radius", "batch_by_class", "one_pass_by_class", "radius_by_class", "batch_knn", "knn_by_class"], default="one_pass")
    parser.add_argument(
        "--context_search_aggregation",
        "--context-search-aggregation",
        dest="context_search_aggregation",
        type=str,
        default=None,
        help="Optional marker for aggregated context search (e.g. 'aggregated_per_omics').",
    )
    parser.add_argument("--radius_scale", "--radius-scale", dest="radius_scale", type=float, default=1.0)
    parser.add_argument("--knn_k", "--knn-k", dest="knn_k", type=int, default=10)
    parser.add_argument("--hyper_params_option", "--hyper-params-option", dest="hyper_params_option", type=int, default=2)
    parser.add_argument("--overwrite_results", "--overwrite-results", dest="overwrite_results", action="store_true")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def _canonicalize_ids(raw_ids) -> set[str | int]:
    selected_ids: set[str | int] = set()
    for sample_id in raw_ids:
        selected_ids.add(sample_id)
        sample_id_str = str(sample_id).strip()
        selected_ids.add(sample_id_str)
        try:
            selected_ids.add(int(sample_id_str))
        except Exception:
            pass
    return selected_ids


def load_selected_ids(ancestry: str) -> set[str | int]:
    filename = "black_ids.pkl" if ancestry == "AFR" else "white_ids.pkl"
    with open(DATA_ROOT / "ancestral_info" / filename, "rb") as handle:
        raw_ids = pickle.load(handle)
    return _canonicalize_ids(raw_ids)


def task_name(args: argparse.Namespace) -> str:
    return f"{args.cancer_type}_{args.ancestry}_{args.expression_type}_{args.target}_{args.years}"


def feature_suffix(args: argparse.Namespace) -> str:
    if args.tcga_omics_feature_source == "raw":
        return ""
    if args.tcga_omics_feature_source == "autoencoder":
        return "_autoencoder"
    return f"_pca{args.tcga_pca_dim}"


def split_root(args: argparse.Namespace, continuum: bool = False) -> Path:
    split_prefix = "continuum_split" if continuum else "raw_split"
    base_dir = DATA_ROOT / "Sample_Splits"
    if args.tcga_omics_feature_source == "raw":
        if continuum:
            raise ValueError("Continuum splits are only available for TCGA autoencoder or PCA omics features.")
        return base_dir / split_prefix
    if args.tcga_omics_feature_source == "autoencoder":
        return base_dir / f"{split_prefix}_tcga_ae{args.tcga_autoencoder_dim}d"
    return base_dir / f"{split_prefix}_tcga_pca{args.tcga_pca_dim}d_ctxae{args.tcga_context_autoencoder_dim}d"


def raw_split_dir(args: argparse.Namespace) -> Path:
    return split_root(args, continuum=False) / task_name(args)


def continuum_split_dir(args: argparse.Namespace) -> Path:
    return split_root(args, continuum=True) / task_name(args)


def result_dir(args: argparse.Namespace) -> Path:
    base = f"{args.ancestry}_{args.cancer_type}_{args.expression_type}_{args.target}_year{args.years}{feature_suffix(args)}"
    context_base = f"{base}_ctxae{args.tcga_context_autoencoder_dim}"
    approach = args.modeling_approach
    if approach in ELASTICNET_SELECTED_ICL_MODES:
        mode = ELASTICNET_SELECTED_ICL_MODES[approach]
        folder = {
            "mix": "Discrete_Stractification_ElasticNetSelected_ICL_AI",
            "min": "Discrete_Stractification_ElasticNetSelected_ICL_AI_min",
            "maj": "Discrete_Stractification_ElasticNetSelected_ICL_AI_maj",
        }[mode]
        return RESULT_ROOT / folder / f"{folder}_{base}{RESULT_SUFFIX}"
    if approach in ELASTICNET_SELECTED_ENHANCED_ICL_MODES:
        mode = ELASTICNET_SELECTED_ENHANCED_ICL_MODES[approach]
        folder = {
            "mix": "Discrete_Stractification_ElasticNetSelected_Enhanced_ICL_AI",
            "min": "Discrete_Stractification_ElasticNetSelected_Enhanced_ICL_AI_min",
        }[mode]
        return (
            RESULT_ROOT
            / folder
            / f"{folder}_{context_base}{RESULT_SUFFIX}"
            / args.context_search_mode
            / f"distance_{args.distance_metric}"
        )
    if approach in DISCRETE_MODES:
        family, mode = DISCRETE_MODES[approach]
        if family == "icl":
            folder = {"mix": "Discrete_Stractification_ICL_AI", "min": "Discrete_Stractification_ICL_AI_min", "maj": "Discrete_Stractification_ICL_AI_maj"}[mode]
            return RESULT_ROOT / folder / f"{folder}_{base}{RESULT_SUFFIX}"
        folder = {
            ("elasticnet", "mix"): "Discrete_Stractification_ElasticNet_AI",
            ("elasticnet", "min"): "Discrete_Stractification_ElasticNet_AI_min",
            ("elasticnet", "maj"): "Discrete_Stractification_ElasticNet_AI_maj",
            ("randomforest", "mix"): "Discrete_Stractification_RandomForest_AI",
            ("randomforest", "min"): "Discrete_Stractification_RandomForest_AI_min",
            ("randomforest", "maj"): "Discrete_Stractification_RandomForest_AI_maj",
            ("xgboost", "mix"): "Discrete_Stractification_XGBoost_AI",
            ("xgboost", "min"): "Discrete_Stractification_XGBoost_AI_min",
            ("xgboost", "maj"): "Discrete_Stractification_XGBoost_AI_maj",
        }[(family, mode)]
        return RESULT_ROOT / folder / f"{folder}_{base}{RESULT_SUFFIX}"
    if approach == "Discrete_Stractification_Enhanced_ICL_AI":
        base_dir = RESULT_ROOT / "Discrete_Stractification_Enhanced_ICL_AI" / f"Discrete_Stractification_Enhanced_ICL_AI_{context_base}{RESULT_SUFFIX}"
        if args.context_search_mode == "radius_by_class":
            scale_suffix = str(args.radius_scale).replace("-", "m").replace(".", "p")
            return base_dir / f"radius_by_class_{scale_suffix}" / f"distance_{args.distance_metric}"
        if args.context_search_mode == "knn_by_class":
            return base_dir / f"knn_by_class_k{args.knn_k}" / f"distance_{args.distance_metric}"
        # if aggregation marker is provided, append _agg to the context-search folder name
        mode_folder = args.context_search_mode
        if getattr(args, "context_search_aggregation", None) and str(args.context_search_aggregation).startswith("aggregated"):
            mode_folder = f"{mode_folder}_agg"
        base_dir = base_dir / mode_folder / f"distance_{args.distance_metric}"
        if args.context_search_mode == "radius":
            base_dir = base_dir / f"radius_scale_{str(args.radius_scale).replace('-', 'm').replace('.', 'p')}"
        return base_dir / "no_label_in_search" if args.context_search_mode in {"batch", "one_pass", "radius", "batch_knn"} else base_dir
    if approach == "Discrete_Stractification_Enhanced_ElasticNet_AI":
        return RESULT_ROOT / "Discrete_Stractification_Enhanced_ElasticNet_AI" / f"Discrete_Stractification_Enhanced_ElasticNet_AI_{context_base}{RESULT_SUFFIX}" / args.context_search_mode / f"distance_{args.distance_metric}"
    if approach == "Discrete_Stractification_Enhanced_RandomForest_AI":
        return RESULT_ROOT / "Discrete_Stractification_Enhanced_RandomForest_AI" / f"Discrete_Stractification_Enhanced_RandomForest_AI_{context_base}{RESULT_SUFFIX}" / args.context_search_mode / f"distance_{args.distance_metric}"
    if approach == "Discrete_Stractification_Enhanced_XGBoost_AI":
        return RESULT_ROOT / "Discrete_Stractification_Enhanced_XGBoost_AI" / f"Discrete_Stractification_Enhanced_XGBoost_AI_{context_base}{RESULT_SUFFIX}" / args.context_search_mode / f"distance_{args.distance_metric}"
    if approach == "Discrete_Stractification_TL_AI":
        return RESULT_ROOT / "Discrete_Stractification_TL_AI" / f"Discrete_Stractification_TL_AI_{base}{RESULT_SUFFIX}" / f"hyperoption{args.hyper_params_option}"
    if approach == "cont_Stractification_Enhanced_ICL_AI":
        base_dir = RESULT_ROOT / "cont_Stractification_Enhanced_ICL_AI" / f"cont_Stractification_Enhanced_ICL_AI_{base}{RESULT_SUFFIX}" / args.context_search_mode / f"distance_{args.distance_metric}"
        return base_dir / "no_label_in_search" if args.context_search_mode in {"batch", "one_pass", "radius", "batch_knn"} else base_dir
    if approach in MERGE_MODES:
        family, mode, weight_use = MERGE_MODES[approach]
        if family == "icl":
            folder = "merge_ancestry_omics_weight_use_ICL_AI" if weight_use else "merge_ancestry_omics_ICL_AI"
            if mode == "min":
                folder += "_min"
            elif mode == "maj":
                folder += "_maj"
            merge_base = f"{args.ancestry}_{args.cancer_type}_{args.expression_type}_{args.target}_year{args.years}"
            return RESULT_ROOT / folder / f"{folder}_{merge_base}{RESULT_SUFFIX}"
        folder = {"elasticnet": "merge_ancestry_omics_ElasticNet_AI", "randomforest": "merge_ancestry_omics_RandomForest_AI", "xgboost": "merge_ancestry_omics_XGBoost_AI"}[family]
        merge_base = f"{args.ancestry}_{args.cancer_type}_{args.expression_type}_{args.target}_year{args.years}"
        return RESULT_ROOT / folder / f"{folder}_{merge_base}{RESULT_SUFFIX}"
    raise ValueError(f"Unsupported TCGA modeling_approach: {approach}")


def discrete_test_split(fold_payload, ancestry: str):
    train_data, minority_test_data, majority_test_data = fold_payload
    return train_data, minority_test_data if ancestry == "AFR" else majority_test_data


def continuum_test_split(fold_payload: dict, ancestry: str) -> dict:
    if ancestry == "AFR":
        return {"test_query": fold_payload["test_white_query"], "test_context": fold_payload["test_white_context"]}
    return {"test_query": fold_payload["test_black_query"], "test_context": fold_payload["test_black_context"]}


def prediction_dict() -> dict[str, list]:
    return {"pred": [], "time": [], "sample_name": []}


def apply_discrete_model(args: argparse.Namespace, train_data, test_data, output: dict[str, list], selected_ids: set[str | int]) -> None:
    approach = args.modeling_approach
    if approach in ELASTICNET_SELECTED_ICL_MODES:
        develop_DS_ElasticNetSelected_ICL_AI_raw_sample_split(
            train_data,
            test_data,
            output,
            selected_id=selected_ids,
            mode=ELASTICNET_SELECTED_ICL_MODES[approach],
            tabpfn_n_estimators=args.tabpfn_n_estimators,
            tabpfn_softmax_temperature=args.tabpfn_softmax_temperature,
            tabpfn_balance_probabilities=args.tabpfn_balance_probabilities,
        )
        return
    if approach in ELASTICNET_SELECTED_ENHANCED_ICL_MODES:
        develop_DS_ElasticNetSelected_Enhanced_ICL_AI_raw_sample_split(
            train_data,
            test_data,
            output,
            selected_id=selected_ids,
            mode=ELASTICNET_SELECTED_ENHANCED_ICL_MODES[approach],
            context_search_mode=args.context_search_mode,
            distance_metric=args.distance_metric,
            tabpfn_n_estimators=args.tabpfn_n_estimators,
            tabpfn_softmax_temperature=args.tabpfn_softmax_temperature,
            tabpfn_balance_probabilities=args.tabpfn_balance_probabilities,
        )
        return
    if approach in DISCRETE_MODES:
        family, mode = DISCRETE_MODES[approach]
        if family == "icl":
            develop_DS_ICL_AI_raw_sample_split(train_data, test_data, output, selected_id=selected_ids, mode=mode, tabpfn_n_estimators=args.tabpfn_n_estimators, tabpfn_softmax_temperature=args.tabpfn_softmax_temperature, tabpfn_balance_probabilities=args.tabpfn_balance_probabilities)
            return
        fn: Callable = {"elasticnet": develop_DS_ElasticNet_AI_raw_sample_split, "randomforest": develop_DS_RandomForest_AI_raw_sample_split, "xgboost": develop_DS_XGBoost_AI_raw_sample_split}[family]
        fn(train_data, test_data, output, selected_id=selected_ids, mode=mode)
        return
    if approach == "Discrete_Stractification_Enhanced_ICL_AI":
        develop_DS_Enhanced_ICL_AI_raw_sample_split(
            train_data,
            test_data,
            output,
            selected_id=selected_ids,
            context_search_mode=args.context_search_mode,
            distance_metric=args.distance_metric,
            tabpfn_n_estimators=args.tabpfn_n_estimators,
            tabpfn_softmax_temperature=args.tabpfn_softmax_temperature,
            tabpfn_balance_probabilities=args.tabpfn_balance_probabilities,
            radius_scale=args.radius_scale,
            knn_k=args.knn_k,
        )
        return
    if approach == "Discrete_Stractification_Enhanced_ElasticNet_AI":
        develop_DS_Enhanced_ElasticNet_AI_raw_sample_split(train_data, test_data, output, selected_id=selected_ids, context_search_mode=args.context_search_mode, distance_metric=args.distance_metric)
        return
    if approach == "Discrete_Stractification_Enhanced_RandomForest_AI":
        develop_DS_Enhanced_RandomForest_AI_raw_sample_split(train_data, test_data, output, selected_id=selected_ids, context_search_mode=args.context_search_mode, distance_metric=args.distance_metric)
        return
    if approach == "Discrete_Stractification_Enhanced_XGBoost_AI":
        develop_DS_Enhanced_XGBoost_AI_raw_sample_split(train_data, test_data, output, selected_id=selected_ids, context_search_mode=args.context_search_mode, distance_metric=args.distance_metric)
        return
    if approach == "Discrete_Stractification_TL_AI":
        develop_DS_TL_AI_for_raw_sample_split(train_data, test_data, output, selected_id=selected_ids, hyper_params_option=args.hyper_params_option, architecture="nn")
        return
    if approach in MERGE_MODES:
        family, mode, weight_use = MERGE_MODES[approach]
        if family == "icl":
            develop_merge_ancestry_omics_ICL_AI_raw_sample_split(train_data, test_data, output, weight_use=weight_use, selected_id=selected_ids, mode=mode)
            return
        fn = {"elasticnet": develop_merge_ancestry_omics_ElasticNet_AI_raw_sample_split, "randomforest": develop_merge_ancestry_omics_RandomForest_AI_raw_sample_split, "xgboost": develop_merge_ancestry_omics_XGBoost_AI_raw_sample_split}[family]
        fn(train_data, test_data, output)
        return
    raise ValueError(f"Unsupported discrete approach: {approach}")


def run_discrete_job(args: argparse.Namespace, selected_ids: set[str | int]) -> None:
    load_dir = raw_split_dir(args)
    save_dir = result_dir(args)
    if not load_dir.exists():
        print(f"skip missing TCGA split directory: {load_dir}")
        return
    if args.overwrite_results and save_dir.exists():
        if not save_dir.resolve().is_relative_to(RESULT_ROOT.resolve()):
            raise ValueError(f"Refusing to overwrite result directory outside RESULT_ROOT: {save_dir}")
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for run_seed in range(args.experimental_runs):
        split_path = load_dir / f"{args.folds}_folds_sample_split_{run_seed}.pkl"
        if not split_path.is_file():
            print(f"skip missing split file: {split_path}")
            continue
        out_path = save_dir / f"prediction_dic_a_run{run_seed}.pkl"
        if out_path.is_file():
            continue
        with open(split_path, "rb") as handle:
            split_many_folds = pickle.load(handle)
        output = prediction_dict()
        set_global_seed(run_seed)
        fold_count = len(split_many_folds) if not isinstance(split_many_folds, dict) else len(split_many_folds.keys())
        if args.fold_index_only is None:
            fold_indices = range(fold_count)
        else:
            if args.fold_index_only < 0 or args.fold_index_only >= fold_count:
                raise ValueError(f"Requested fold_index_only={args.fold_index_only}, but split has {fold_count} folds.")
            fold_indices = [args.fold_index_only]
        for fold_idx in fold_indices:
            train_data, test_data = discrete_test_split(split_many_folds[fold_idx], args.ancestry)
            apply_discrete_model(args, train_data, test_data, output, selected_ids)
        with open(out_path, "wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_continuum_job(args: argparse.Namespace) -> None:
    load_dir = continuum_split_dir(args)
    save_dir = result_dir(args)
    if not load_dir.exists():
        print(f"skip missing TCGA continuum split directory: {load_dir}")
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    for run_seed in range(args.experimental_runs):
        split_path = load_dir / f"{args.folds}_folds_continuum_split_{run_seed}.pkl"
        if not split_path.is_file():
            print(f"skip missing continuum split file: {split_path}")
            continue
        out_path = save_dir / f"prediction_dic_a_run{run_seed}.pkl"
        if out_path.is_file():
            continue
        with open(split_path, "rb") as handle:
            split_many_folds = pickle.load(handle)
        output = prediction_dict()
        set_global_seed(run_seed)
        for fold_idx in range(args.folds):
            dev_data = split_many_folds[fold_idx]
            test_data = continuum_test_split(dev_data, args.ancestry)
            develop_Cont_Enhanced_ICL_AI_raw_sample_split(dev_data, test_data, output, selected_id=None, context_search_mode=args.context_search_mode, distance_metric=args.distance_metric)
        with open(out_path, "wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    if args.data_type != "TCGA":
        raise ValueError(f"This wrapper only supports data_type=TCGA, got {args.data_type!r}")
    if args.modeling_approach == "cont_Stractification_Enhanced_ICL_AI":
        run_continuum_job(args)
        return
    selected_ids = load_selected_ids(args.ancestry)
    run_discrete_job(args, selected_ids)


if __name__ == "__main__":
    main()
