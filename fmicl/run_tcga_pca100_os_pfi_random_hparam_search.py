from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FMICL_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, FMICL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

TCGA_CANCERS = [
    "BRCA", "LUAD", "UCEC", "CESC", "GBM",
    "HNSC", "KIRC", "KIRP", "ACC", "BLCA",
    "CHOL", "COAD", "DLBC", "ESCA", "KICH",
    "LAML", "LIHC", "LUSC", "MESO", "OV",
    "PAAD", "PCPG", "PRAD", "READ", "SARC",
    "SKCM", "STAD", "TGCT", "THCA", "THYM",
    "UCS", "UVM", "LGG",
]
TCGA_OMICS = ["mRNA", "MicroRNA", "Methylation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time-capped random hyperparameter search for TCGA pca100 OS/PFI survival tasks."
    )
    parser.add_argument(
        "--method",
        choices=[
            "ds_tl",
            "icl_min",
            "icl_mixed",
            "icl_knn_by_class_ce",
            "icl_knn_by_class_auroc",
            "icl_one_pass_by_class_hs_ce",
            "elasticnet_min",
            "elasticnet_mixed",
            "randomforest_min",
            "randomforest_mixed",
            "xgboost_min",
            "xgboost_mixed",
        ],
        required=True,
    )
    parser.add_argument("--cancer-type", choices=TCGA_CANCERS, required=True)
    parser.add_argument("--expression-type", choices=TCGA_OMICS, required=True)
    parser.add_argument("--target", choices=["OS", "PFI", "DSS", "DFI"], default="OS")
    parser.add_argument("--year", type=int, default=2)
    parser.add_argument("--search-run", type=int, required=True, help="Random-search repeat id. Also selects split seed by default.")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=10000)
    parser.add_argument("--time-limit-min", type=float, default=60.0)
    parser.add_argument("--pca-dim", type=int, default=100)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--ancestry", choices=["AFR"], default="AFR")
    parser.add_argument("--result-suffix", default=os.environ.get("TCGA_HPARAM_RESULT_SUFFIX", ""))
    parser.add_argument("--classical-n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--tabpfn-n-estimators", type=int, default=32)
    parser.add_argument("--tabpfn-softmax-temperature", type=float, default=0.9)
    parser.add_argument("--tabpfn-balance-probabilities", action="store_true")
    parser.add_argument("--knn-ks", default="1,3,5,10", help="Comma-separated K candidates for KNN-by-class ICL hparam methods.")
    parser.add_argument(
        "--result-target-group",
        default=None,
        help="Optional endpoint group used in the result root name, e.g. OS_PFI.",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def canonicalize_ids(raw_ids) -> set[str | int]:
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


def load_selected_ids() -> set[str | int]:
    with open(DATA_ROOT / "ancestral_info" / "black_ids.pkl", "rb") as handle:
        return canonicalize_ids(pickle.load(handle))


def split_path(args: argparse.Namespace) -> Path:
    split_seed = args.search_run if args.split_seed is None else args.split_seed
    split_dir = (
        DATA_ROOT
        / "Sample_Splits"
        / f"raw_split_tcga_pca{args.pca_dim}d_ctxae20d"
        / f"{args.cancer_type}_{args.ancestry}_{args.expression_type}_{args.target}_{args.year}"
    )
    return split_dir / f"{args.folds}_folds_sample_split_{split_seed}.pkl"


def result_dir(args: argparse.Namespace) -> Path:
    suffix = args.result_suffix.strip()
    run_name = (
        f"{args.method}_{args.ancestry}_{args.cancer_type}_{args.expression_type}"
        f"_{args.target}_year{args.year}_pca{args.pca_dim}{suffix}"
    )
    target_group = (args.result_target_group or args.target).strip()
    root_name = f"Random_HParam_Search_{target_group}_year{args.year}_pca{args.pca_dim}"
    return RESULT_ROOT / root_name / args.method / run_name / f"search_run{args.search_run}"


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


def log_uniform_or_zero(rng: random.Random, low: float, high: float, zero_prob: float = 0.15) -> float:
    if rng.random() < zero_prob:
        return 0.0
    return log_uniform(rng, low, high)


def random_candidates(method: str, num_candidates: int, seed: int, knn_ks: list[int] | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    family = method.replace("_mixed", "_min")
    if method in {"icl_knn_by_class_ce", "icl_knn_by_class_auroc"}:
        return [{"knn_k": k} for k in (knn_ks or parse_knn_ks("1,3,5,10"))]
    if method == "icl_one_pass_by_class_hs_ce":
        return [
            {"strategy": "min"},
            {"strategy": "one_pass_by_class"},
            {"strategy": "mix"},
        ]
    for _ in range(num_candidates):
        if family == "elasticnet_min":
            candidates.append(
                {
                    "C": log_uniform(rng, 1e-3, 1e2),
                    "l1_ratio": rng.uniform(0.0, 1.0),
                    "class_weight": rng.choice(["balanced", None]),
                    "max_iter": 5000,
                }
            )
        elif family == "randomforest_min":
            candidates.append(
                {
                    "n_estimators": rng.choice([100, 200, 300, 500, 800, 1000]),
                    "max_depth": rng.choice([None, 3, 5, 8, 10, 15, 20]),
                    "min_samples_leaf": rng.choice([1, 2, 3, 5, 10]),
                    "min_samples_split": rng.choice([2, 4, 8, 16]),
                    "max_features": rng.choice(["sqrt", "log2", None, 0.25, 0.5, 0.75]),
                    "class_weight": rng.choice(["balanced", "balanced_subsample", None]),
                }
            )
        elif family == "xgboost_min":
            candidates.append(
                {
                    "n_estimators": rng.choice([100, 200, 300, 400, 600, 800]),
                    "max_depth": rng.choice([2, 3, 4, 5, 6, 8]),
                    "learning_rate": log_uniform(rng, 0.005, 0.2),
                    "subsample": rng.uniform(0.5, 1.0),
                    "colsample_bytree": rng.uniform(0.5, 1.0),
                    "min_child_weight": log_uniform(rng, 0.1, 10.0),
                    "gamma": log_uniform_or_zero(rng, 1e-4, 5.0),
                    "reg_alpha": log_uniform_or_zero(rng, 1e-5, 10.0),
                    "reg_lambda": log_uniform(rng, 1e-3, 100.0),
                }
            )
        elif family == "ds_tl":
            candidates.append(
                {
                    "batch_size": rng.choice([8, 16, 32, 64]),
                    "learning_rate": log_uniform(rng, 1e-4, 3e-2),
                    "L1_reg": log_uniform_or_zero(rng, 1e-7, 1e-3),
                    "L2_reg": log_uniform_or_zero(rng, 1e-8, 1e-4),
                    "base_epochs": 150,
                    "finetune_epochs": 50,
                }
            )
        elif family == "icl_min":
            candidates.append(
                {
                    "n_estimators": 32,
                    "softmax_temperature": 0.9,
                    "balance_probabilities": False,
                }
            )
            break
        else:
            raise ValueError(f"Unsupported method: {method}")
    return candidates


def parse_knn_ks(raw: str) -> list[int]:
    ks = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not ks:
        raise ValueError("--knn-ks must contain at least one integer.")
    if any(k < 1 for k in ks):
        raise ValueError("--knn-ks values must all be positive.")
    return ks


def prediction_dict() -> dict[str, list]:
    return {"pred": [], "time": [], "sample_name": []}


def sample_mode_for_method(method: str) -> str:
    return "mix" if method.endswith("_mixed") else "min"


def model_family(method: str) -> str:
    return method.replace("_mixed", "_min")


def _ensure_binary_predict_proba(model, train_X, train_y, test_X):
    model.fit(train_X, train_y)
    pred_prob = model.predict_proba(test_X)
    classes = np.asarray(model.classes_).reshape(-1)
    if len(classes) == 1:
        if classes[0] == 0:
            return np.column_stack((np.ones(len(test_X)), np.zeros(len(test_X))))
        return np.column_stack((np.zeros(len(test_X)), np.ones(len(test_X))))
    if pred_prob.shape[1] != 2:
        raise ValueError(f"Expected binary predict_proba with 2 columns, got {pred_prob.shape}.")
    if list(classes) == [0, 1]:
        return pred_prob
    class_to_col = {int(cls): idx for idx, cls in enumerate(classes)}
    return np.column_stack((pred_prob[:, class_to_col[0]], pred_prob[:, class_to_col[1]]))


def _prepare_discrete_predictive_arrays(train_data, test_data, selected_id=None, mode="mix"):
    selected_id = set(selected_id) if selected_id is not None else None
    train_X = []
    train_y = []
    test_X = []
    for key, val in train_data.items():
        if mode == "maj" and selected_id is not None and key in selected_id:
            continue
        if mode == "min" and selected_id is not None and key not in selected_id:
            continue
        predictive_feat = np.asarray(val[0], dtype=np.float32).reshape(-1)
        label = int(np.asarray(val[2]).reshape(-1)[0])
        train_X.append(predictive_feat)
        train_y.append(label)

    test_sample_id = list(test_data.keys())
    for key in test_sample_id:
        predictive_feat = np.asarray(test_data[key][0], dtype=np.float32).reshape(-1)
        test_X.append(predictive_feat)

    feat_dim = train_X[0].shape[-1] if train_X else (test_X[0].shape[-1] if test_X else 0)
    train_X = np.asarray(train_X, dtype=np.float32).reshape((-1, feat_dim)) if train_X else np.zeros((0, feat_dim), dtype=np.float32)
    train_y = np.asarray(train_y, dtype=int)
    test_X = np.asarray(test_X, dtype=np.float32).reshape((-1, feat_dim)) if test_X else np.zeros((0, feat_dim), dtype=np.float32)
    return train_X, train_y, test_X, test_sample_id


def build_model(method: str, params: dict[str, Any], n_jobs: int):
    family = model_family(method)
    if family == "elasticnet_min":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        C=float(params["C"]),
                        l1_ratio=float(params["l1_ratio"]),
                        class_weight=params["class_weight"],
                        max_iter=int(params["max_iter"]),
                        random_state=0,
                    ),
                ),
            ]
        )
    if family == "randomforest_min":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=params["max_depth"],
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        min_samples_split=int(params["min_samples_split"]),
                        max_features=params["max_features"],
                        class_weight=params["class_weight"],
                        random_state=0,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
    if family == "xgboost_min":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is required for xgboost.") from exc
        if XGBClassifier is None:
            raise ImportError("xgboost is required for xgboost.")
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=int(params["max_depth"]),
                        learning_rate=float(params["learning_rate"]),
                        subsample=float(params["subsample"]),
                        colsample_bytree=float(params["colsample_bytree"]),
                        min_child_weight=float(params["min_child_weight"]),
                        gamma=float(params["gamma"]),
                        reg_alpha=float(params["reg_alpha"]),
                        reg_lambda=float(params["reg_lambda"]),
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=0,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
    raise ValueError(f"No classical model builder for {method}")


def append_classical_predictions(method: str, params: dict[str, Any], train_data, test_data, selected_ids, output, n_jobs: int) -> None:
    train_X, train_y, test_X, test_sample_id = _prepare_discrete_predictive_arrays(
        train_data,
        test_data,
        selected_id=selected_ids,
        mode=sample_mode_for_method(method),
    )
    model = build_model(method, params, n_jobs)
    start = time.perf_counter()
    if len(test_X) > 0:
        pred_prob = _ensure_binary_predict_proba(model, train_X, train_y, test_X)
        elapsed = time.perf_counter() - start
        output["pred"].append(np.asarray(pred_prob, dtype=float).reshape((-1, 2)))
        output["sample_name"].append(test_sample_id)
        output["time"].append(elapsed)
    else:
        output["pred"].append(np.zeros((0, 2)))
        output["sample_name"].append(test_sample_id)
        output["time"].append(-1)


def append_icl_predictions(args: argparse.Namespace, params: dict[str, Any], train_data, test_data, selected_ids, output) -> None:
    from tabpfn import TabPFNClassifier

    train_X, train_y, test_X, test_sample_id = _prepare_discrete_predictive_arrays(
        train_data,
        test_data,
        selected_id=selected_ids,
        mode=sample_mode_for_method(args.method),
    )
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    start = time.perf_counter()
    model = TabPFNClassifier(
        n_estimators=int(params.get("n_estimators", args.tabpfn_n_estimators)),
        softmax_temperature=float(params.get("softmax_temperature", args.tabpfn_softmax_temperature)),
        balance_probabilities=bool(params.get("balance_probabilities", args.tabpfn_balance_probabilities)),
    )
    if len(test_X) > 0:
        model.fit(train_X, train_y)
        pred_prob = model.predict_proba(test_X)
        elapsed = time.perf_counter() - start
        classes = model.classes_
        if len(classes) == 1:
            if classes[0] == 0:
                pred_prob = np.column_stack((np.ones(len(pred_prob)), np.zeros(len(pred_prob))))
            else:
                pred_prob = np.column_stack((np.zeros(len(pred_prob)), np.ones(len(pred_prob))))
        output["pred"].append(np.asarray(pred_prob, dtype=float).reshape((-1, 2)))
        output["sample_name"].append(test_sample_id)
        output["time"].append(elapsed)
    else:
        output["pred"].append(np.zeros((0, 2)))
        output["sample_name"].append(test_sample_id)
        output["time"].append(-1)


def _context_arrays(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    sample_ids = list(data.keys())
    features = []
    labels = []
    contexts = []
    for sample_id in sample_ids:
        value = data[sample_id]
        features.append(np.asarray(value[0], dtype=np.float32).reshape(-1))
        labels.append(int(np.asarray(value[2]).reshape(-1)[0]))
        contexts.append(np.asarray(value[3] if len(value) > 3 else value[0], dtype=np.float32).reshape(-1))
    feat_dim = features[0].shape[-1] if features else 0
    context_dim = contexts[0].shape[-1] if contexts else 0
    return (
        np.asarray(features, dtype=np.float32).reshape((-1, feat_dim)) if features else np.zeros((0, feat_dim), dtype=np.float32),
        np.asarray(labels, dtype=int),
        np.asarray(contexts, dtype=np.float32).reshape((-1, context_dim)) if contexts else np.zeros((0, context_dim), dtype=np.float32),
        sample_ids,
    )


def knn_by_class_source_indices(source_context, source_y, target_context, target_y, knn_k: int, distance_metric: str = "euclidean") -> np.ndarray:
    from tcga_ancestral_continuum_networks import _normalize_augmented_feature_blocks, knn_graph_search_context_cls

    normalized = _normalize_augmented_feature_blocks(
        [
            ("source", source_context, source_y),
            ("target", target_context, target_y),
        ]
    )
    source_context = normalized["source"]
    target_context = normalized["target"]
    selected = []
    for cls in np.unique(np.asarray(target_y).reshape(-1)):
        source_mask = np.asarray(source_y).reshape(-1) == cls
        target_mask = np.asarray(target_y).reshape(-1) == cls
        if not np.any(source_mask) or not np.any(target_mask):
            continue
        local = knn_graph_search_context_cls(
            source_context[source_mask],
            np.asarray(source_y)[source_mask],
            target_context[target_mask],
            np.asarray(target_y)[target_mask],
            k=int(knn_k),
            distance_metric=distance_metric,
        )
        selected.extend(np.where(source_mask)[0][np.asarray(local, dtype=int)].tolist())
    return np.asarray(sorted(set(selected)), dtype=int)


def one_pass_by_class_source_indices(source_context, source_y, target_context, target_y, distance_metric: str = "euclidean") -> np.ndarray:
    from tcga_ancestral_continuum_networks import _normalize_augmented_feature_blocks, search_context_cls_by_class

    normalized = _normalize_augmented_feature_blocks(
        [
            ("source", source_context, source_y),
            ("target", target_context, target_y),
        ]
    )
    source_context = normalized["source"]
    target_context = normalized["target"]
    selected = search_context_cls_by_class(
        source_context,
        source_y,
        target_context,
        target_y,
        distance_metric=distance_metric,
    )
    return np.asarray(sorted(set(np.asarray(selected, dtype=int).tolist())), dtype=int)


def evaluate_tabpfn_dict(args: argparse.Namespace, train_data, holdout_data) -> tuple[dict[str, list], dict[str, Any]]:
    from tabpfn import TabPFNClassifier

    output = prediction_dict()
    train_X, train_y, _train_context, _train_ids = _context_arrays(train_data)
    holdout_X, _holdout_y, _holdout_context, sample_ids = _context_arrays(holdout_data)

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    holdout_X = scaler.transform(holdout_X) if len(holdout_X) > 0 else holdout_X

    start = time.perf_counter()
    if len(holdout_X) > 0:
        model = TabPFNClassifier(
            n_estimators=args.tabpfn_n_estimators,
            softmax_temperature=args.tabpfn_softmax_temperature,
            balance_probabilities=args.tabpfn_balance_probabilities,
        )
        model.fit(train_X, train_y)
        pred_prob = model.predict_proba(holdout_X)
        classes = np.asarray(model.classes_).reshape(-1)
        if len(classes) == 1:
            if classes[0] == 0:
                pred_prob = np.column_stack((np.ones(len(holdout_X)), np.zeros(len(holdout_X))))
            else:
                pred_prob = np.column_stack((np.zeros(len(holdout_X)), np.ones(len(holdout_X))))
        elapsed = time.perf_counter() - start
    else:
        pred_prob = np.zeros((0, 2))
        elapsed = -1

    output["pred"].append(np.asarray(pred_prob, dtype=float).reshape((-1, 2)))
    output["sample_name"].append(sample_ids)
    output["time"].append(elapsed)
    return output, {
        "n_train_samples": int(len(train_data)),
        "n_selected_source": None,
    }


def evaluate_one_pass_hs_icl(args: argparse.Namespace, strategy: str, source_data, target_train_data, holdout_data) -> tuple[dict[str, list], dict[str, Any]]:
    if strategy == "min":
        return evaluate_tabpfn_dict(args, target_train_data, holdout_data)
    if strategy == "mix":
        return evaluate_tabpfn_dict(args, merge_dicts(source_data, target_train_data), holdout_data)
    if strategy != "one_pass_by_class":
        raise ValueError(f"Unsupported one-pass HS strategy: {strategy}")

    source_X, source_y, source_context, source_ids = _context_arrays(source_data)
    target_X, target_y, target_context, target_ids = _context_arrays(target_train_data)
    source_indices = one_pass_by_class_source_indices(source_context, source_y, target_context, target_y)
    selected_source_sample_ids = [source_ids[int(i)] for i in source_indices.tolist()]
    selected_source_data = {source_ids[int(i)]: source_data[source_ids[int(i)]] for i in source_indices.tolist()}
    train_data = merge_dicts(selected_source_data, target_train_data)
    output, extra = evaluate_tabpfn_dict(args, train_data, holdout_data)
    extra.update(
        {
            "strategy": strategy,
            "n_source_pool": int(len(source_data)),
            "n_target_context": int(len(target_ids)),
            "n_selected_source": int(len(source_indices)),
            "selected_source_sample_ids": selected_source_sample_ids,
        }
    )
    return output, extra


def evaluate_knn_by_class_icl(args: argparse.Namespace, knn_k: int, source_data, target_train_data, holdout_data) -> tuple[dict[str, list], dict[str, Any]]:
    from tabpfn import TabPFNClassifier

    output = prediction_dict()
    source_X, source_y, source_context, source_ids = _context_arrays(source_data)
    target_X, target_y, target_context, _target_ids = _context_arrays(target_train_data)
    holdout_X, _holdout_y, _holdout_context, sample_ids = _context_arrays(holdout_data)
    source_indices = knn_by_class_source_indices(source_context, source_y, target_context, target_y, int(knn_k))
    selected_source_sample_ids = [source_ids[int(i)] for i in source_indices.tolist()]
    if len(source_indices) > 0:
        ext_X = np.vstack((source_X[source_indices], target_X))
        ext_y = np.concatenate((source_y[source_indices], target_y)).ravel()
    else:
        ext_X = target_X
        ext_y = target_y.ravel()

    scaler = StandardScaler()
    ext_X = scaler.fit_transform(ext_X)
    holdout_X = scaler.transform(holdout_X) if len(holdout_X) > 0 else holdout_X

    start = time.perf_counter()
    if len(holdout_X) > 0:
        model = TabPFNClassifier(
            n_estimators=args.tabpfn_n_estimators,
            softmax_temperature=args.tabpfn_softmax_temperature,
            balance_probabilities=args.tabpfn_balance_probabilities,
        )
        model.fit(ext_X, ext_y)
        pred_prob = model.predict_proba(holdout_X)
        classes = np.asarray(model.classes_).reshape(-1)
        if len(classes) == 1:
            if classes[0] == 0:
                pred_prob = np.column_stack((np.ones(len(holdout_X)), np.zeros(len(holdout_X))))
            else:
                pred_prob = np.column_stack((np.zeros(len(holdout_X)), np.ones(len(holdout_X))))
        elapsed = time.perf_counter() - start
    else:
        pred_prob = np.zeros((0, 2))
        elapsed = -1

    output["pred"].append(np.asarray(pred_prob, dtype=float).reshape((-1, 2)))
    output["sample_name"].append(sample_ids)
    output["time"].append(elapsed)
    return output, {
        "knn_k": int(knn_k),
        "n_selected_source": int(len(source_indices)),
        "selected_source_sample_ids": selected_source_sample_ids,
    }


def append_ds_tl_predictions(params: dict[str, Any], train_data, test_data, selected_ids, output) -> None:
    import torch
    from advanced_transfer_learning.Multimodal_NN.Multimodal_nn_utils import HParams as nn_HParams
    from advanced_transfer_learning.Multimodal_NN.Multimodal_nn_utils import _probs_from_logits_binary
    from advanced_transfer_learning.Multimodal_NN.Multimodal_nn_utils import train_binary_mlp as trainer

    train_src_X = []
    train_src_y = []
    train_tgt_X = []
    train_tgt_y = []
    test_X = []

    for key, val in train_data.items():
        predictive_feat = np.asarray(val[0], dtype=np.float32).reshape(-1)
        label = int(np.asarray(val[2]).reshape(-1)[0])
        if key in selected_ids:
            train_tgt_X.append(predictive_feat)
            train_tgt_y.append(label)
        else:
            train_src_X.append(predictive_feat)
            train_src_y.append(label)

    test_sample_id = list(test_data.keys())
    for key in test_sample_id:
        test_X.append(np.asarray(test_data[key][0], dtype=np.float32).reshape(-1))

    if not train_src_X or not train_tgt_X:
        output["pred"].append(np.zeros((0, 2)))
        output["sample_name"].append(test_sample_id)
        output["time"].append(-1)
        return

    feat_dim = train_src_X[0].shape[-1]
    train_src_X = np.asarray(train_src_X, dtype=np.float32).reshape((-1, feat_dim))
    train_tgt_X = np.asarray(train_tgt_X, dtype=np.float32).reshape((-1, feat_dim))
    train_src_y = np.asarray(train_src_y, dtype=int)
    train_tgt_y = np.asarray(train_tgt_y, dtype=int)
    test_X = np.asarray(test_X, dtype=np.float32).reshape((-1, feat_dim)) if test_X else np.zeros((0, feat_dim), dtype=np.float32)

    scaler = StandardScaler()
    scaler.fit(np.vstack((train_src_X, train_tgt_X)))
    train_src_X = scaler.transform(train_src_X)
    train_tgt_X = scaler.transform(train_tgt_X)
    test_X = scaler.transform(test_X)

    base_hparams = nn_HParams(
        n_epochs=int(params["base_epochs"]),
        learning_rate=float(params["learning_rate"]),
        dropout=0.0,
        batch_size=int(params["batch_size"]),
        L1_reg=float(params["L1_reg"]),
        L2_reg=float(params["L2_reg"]),
    )
    finetune_hparams = nn_HParams(
        n_epochs=int(params["finetune_epochs"]),
        learning_rate=float(params["learning_rate"]),
        dropout=0.0,
        batch_size=int(params["batch_size"]),
        L1_reg=float(params["L1_reg"]),
        L2_reg=float(params["L2_reg"]),
    )

    start = time.perf_counter()
    base_model, _ = trainer(train_src_X, train_src_y, h=base_hparams)
    tuned_model, _ = trainer(train_tgt_X, train_tgt_y, model=base_model, h=finetune_hparams)
    tuned_model.eval()
    device = next(tuned_model.parameters()).device
    with torch.no_grad():
        logits = tuned_model(torch.tensor(test_X, dtype=torch.float32).to(device)).squeeze(1).cpu().numpy()
    pred_prob = _probs_from_logits_binary(logits)
    elapsed = time.perf_counter() - start

    output["pred"].append(np.asarray(pred_prob, dtype=float).reshape((-1, 2)))
    output["sample_name"].append(test_sample_id)
    output["time"].append(elapsed)


def _record_to_feature_label(value) -> tuple[np.ndarray, int]:
    predictive_feat = np.asarray(value[0], dtype=np.float32).reshape(-1)
    label = int(np.asarray(value[2]).reshape(-1)[0])
    return predictive_feat, label


def _arrays_from_records(data: dict) -> tuple[np.ndarray, np.ndarray, list]:
    sample_ids = list(data.keys())
    features = []
    labels = []
    for sample_id in sample_ids:
        feat, label = _record_to_feature_label(data[sample_id])
        features.append(feat)
        labels.append(label)
    if not features:
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=int), sample_ids
    feat_dim = features[0].shape[-1]
    return (
        np.asarray(features, dtype=np.float32).reshape((-1, feat_dim)),
        np.asarray(labels, dtype=int),
        sample_ids,
    )


def _split_by_selected_ids(data: dict, selected_ids: set[str | int]) -> tuple[dict, dict]:
    minority = {}
    majority = {}
    for key, value in data.items():
        if key in selected_ids:
            minority[key] = value
        else:
            majority[key] = value
    return majority, minority


def _predict_ds_tl_model(model, scaler: StandardScaler, test_data) -> dict[str, list]:
    import torch
    from advanced_transfer_learning.Multimodal_NN.Multimodal_nn_utils import _probs_from_logits_binary

    output = prediction_dict()
    test_X, _test_y, test_sample_id = _arrays_from_records(test_data)
    if test_X.shape[0] == 0:
        output["pred"].append(np.zeros((0, 2)))
        output["sample_name"].append(test_sample_id)
        output["time"].append(-1)
        return output

    test_X = scaler.transform(test_X)
    start = time.perf_counter()
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(test_X, dtype=torch.float32).to(device)).squeeze(1).cpu().numpy()
    pred_prob = _probs_from_logits_binary(logits)
    elapsed = time.perf_counter() - start

    output["pred"].append(np.asarray(pred_prob, dtype=float).reshape((-1, 2)))
    output["sample_name"].append(test_sample_id)
    output["time"].append(elapsed)
    return output


def fit_ds_tl_candidate(
    params: dict[str, Any],
    majority_train_data,
    majority_validation_data,
    minority_train_data,
    minority_validation_data,
):
    from advanced_transfer_learning.Multimodal_NN.Multimodal_nn_utils import HParams as nn_HParams
    from advanced_transfer_learning.Multimodal_NN.Multimodal_nn_utils import train_binary_mlp as trainer

    maj_train_X, maj_train_y, _ = _arrays_from_records(majority_train_data)
    maj_val_X, maj_val_y, _ = _arrays_from_records(majority_validation_data)
    min_train_X, min_train_y, _ = _arrays_from_records(minority_train_data)
    min_val_X, min_val_y, _ = _arrays_from_records(minority_validation_data)

    if (
        maj_train_X.shape[0] == 0
        or maj_val_X.shape[0] == 0
        or min_train_X.shape[0] == 0
        or min_val_X.shape[0] == 0
    ):
        return None

    scaler = StandardScaler()
    scaler.fit(np.vstack((maj_train_X, min_train_X)))
    maj_train_X = scaler.transform(maj_train_X)
    maj_val_X = scaler.transform(maj_val_X)
    min_train_X = scaler.transform(min_train_X)
    min_val_X = scaler.transform(min_val_X)

    base_hparams = nn_HParams(
        n_epochs=int(params["base_epochs"]),
        learning_rate=float(params["learning_rate"]),
        dropout=0.0,
        batch_size=int(params["batch_size"]),
        L1_reg=float(params["L1_reg"]),
        L2_reg=float(params["L2_reg"]),
    )
    finetune_hparams = nn_HParams(
        n_epochs=int(params["finetune_epochs"]),
        learning_rate=float(params["learning_rate"]),
        dropout=0.0,
        batch_size=int(params["batch_size"]),
        L1_reg=float(params["L1_reg"]),
        L2_reg=float(params["L2_reg"]),
    )

    base_model, pretrain_val_ce = trainer(
        maj_train_X,
        maj_train_y,
        h=base_hparams,
        X_val=maj_val_X,
        y_val=maj_val_y,
    )
    tuned_model, finetune_val_ce = trainer(
        min_train_X,
        min_train_y,
        model=base_model,
        h=finetune_hparams,
        X_val=min_val_X,
        y_val=min_val_y,
    )
    return {
        "model": tuned_model,
        "scaler": scaler,
        "pretrain_validation_cross_entropy": float(pretrain_val_ce),
        "validation_cross_entropy": float(finetune_val_ce),
    }


def evaluate_holdout(args: argparse.Namespace, params: dict[str, Any], train_data, holdout_data, selected_ids) -> dict[str, list]:
    """Train on one split and predict one holdout split."""
    output = prediction_dict()
    if model_family(args.method) == "icl_min":
        append_icl_predictions(args, params, train_data, holdout_data, selected_ids, output)
    elif args.method == "ds_tl":
        append_ds_tl_predictions(params, train_data, holdout_data, selected_ids, output)
    else:
        append_classical_predictions(args.method, params, train_data, holdout_data, selected_ids, output, args.classical_n_jobs)
    return output


def labels_for_samples(data_block, sample_ids: list) -> np.ndarray:
    labels = []
    for sample_id in sample_ids:
        value = data_block[sample_id]
        labels.append(int(np.asarray(value[2]).reshape(-1)[0]))
    return np.asarray(labels, dtype=int)


def validation_cross_entropy(output: dict[str, list], validation_data) -> float | None:
    if not output.get("pred") or not output.get("sample_name"):
        return None
    pred_arr = np.asarray(output["pred"][0], dtype=float)
    sample_ids = [sid for sid in output["sample_name"][0] if sid in validation_data]
    limit = min(len(sample_ids), pred_arr.shape[0])
    if limit == 0:
        return None
    sample_ids = sample_ids[:limit]
    pred_arr = pred_arr[:limit]
    if pred_arr.ndim == 1:
        pred_arr = np.column_stack((1.0 - pred_arr, pred_arr))
    if pred_arr.shape[1] == 1:
        pred_arr = np.column_stack((1.0 - pred_arr[:, 0], pred_arr[:, 0]))
    y_true = labels_for_samples(validation_data, sample_ids)
    try:
        return float(log_loss(y_true, pred_arr[:, -1], labels=[0, 1]))
    except Exception:
        return None


def validation_auroc(output: dict[str, list], validation_data) -> float | None:
    if not output.get("pred") or not output.get("sample_name"):
        return None
    pred_arr = np.asarray(output["pred"][0], dtype=float)
    sample_ids = [sid for sid in output["sample_name"][0] if sid in validation_data]
    limit = min(len(sample_ids), pred_arr.shape[0])
    if limit == 0:
        return None
    sample_ids = sample_ids[:limit]
    pred_arr = pred_arr[:limit]
    if pred_arr.ndim == 1:
        pred_arr = np.column_stack((1.0 - pred_arr, pred_arr))
    if pred_arr.shape[1] == 1:
        pred_arr = np.column_stack((1.0 - pred_arr[:, 0], pred_arr[:, 0]))
    y_true = labels_for_samples(validation_data, sample_ids)
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, pred_arr[:, -1]))
    except Exception:
        return None


def split_record_dict(data: dict, sample_ids: list) -> dict:
    return {sample_id: data[sample_id] for sample_id in sample_ids}


def merge_dicts(*parts: dict) -> dict:
    merged = {}
    for part in parts:
        merged.update(part)
    return merged


def _safe_train_validation_split(data: dict, split_seed: int, validation_fraction: float = 0.2):
    sample_ids = list(data.keys())
    if len(sample_ids) < 2:
        raise ValueError("Need at least two samples to split train/validation.")
    labels = np.asarray([int(np.asarray(data[sample_id][2]).reshape(-1)[0]) for sample_id in sample_ids], dtype=int)
    stratify = labels if len(np.unique(labels)) > 1 and np.bincount(labels).min() >= 2 else None
    train_ids, validation_ids = train_test_split(
        sample_ids,
        test_size=validation_fraction,
        random_state=split_seed,
        stratify=stratify,
    )
    return split_record_dict(data, train_ids), split_record_dict(data, validation_ids)


def get_train_validation_test(split_many_folds, split_seed: int, method: str, selected_ids: set[str | int]):
    """Build a no-leak train/validation/test split from fold 0.

    Validation is held out from fold 0 train samples, and fold 0 minority
    holdout remains the final test set. Mixed methods can use fold 0 majority
    holdout as additional training/context data; min methods do not.
    """
    if 0 not in split_many_folds:
        raise ValueError("Expected fold 0 in split_many_folds.")
    fold0_train_data, fold0_minority_test_data, fold0_majority_test_data = split_many_folds[0]
    fold_train_majority, fold_train_minority = _split_by_selected_ids(fold0_train_data, selected_ids)
    minority_train_data, minority_validation_data = _safe_train_validation_split(
        fold_train_minority,
        split_seed=split_seed,
        validation_fraction=0.2,
    )

    test_overlap = set(fold0_train_data).intersection(fold0_minority_test_data)
    if test_overlap:
        raise ValueError(f"Fold 0 train overlaps final minority test set: {len(test_overlap)}")

    if method == "ds_tl":
        majority_pool = dict(fold_train_majority)
        majority_pool.update(fold0_majority_test_data)
        majority_train_data, majority_validation_data = _safe_train_validation_split(
            majority_pool,
            split_seed=split_seed,
            validation_fraction=0.2,
        )
        train_data = dict(majority_train_data)
        train_data.update(minority_train_data)
        return {
            "train_data": train_data,
            "validation_data": minority_validation_data,
            "test_data": fold0_minority_test_data,
            "majority_train_data": majority_train_data,
            "majority_validation_data": majority_validation_data,
            "minority_train_data": minority_train_data,
            "minority_validation_data": minority_validation_data,
            "final_train_data": None,
            "metadata": {
                "split_source": "split_many_folds[0]",
                "pretrain_validation": "majority samples from fold0_train_data + fold0_majority_test_data",
                "finetune_validation": "minority samples from fold0_train_data",
                "test": "fold0_minority_test_data",
            },
        }

    if method in {"icl_knn_by_class_ce", "icl_knn_by_class_auroc", "icl_one_pass_by_class_hs_ce"}:
        selection_metric = "validation_auroc" if method == "icl_knn_by_class_auroc" else "validation_cross_entropy"
        source_train_data = dict(fold_train_majority)
        source_train_data.update(fold0_majority_test_data)
        target_train_data = dict(minority_train_data)
        target_final_train_data = dict(minority_train_data)
        target_final_train_data.update(minority_validation_data)
        return {
            "train_data": merge_dicts(source_train_data, target_train_data),
            "validation_data": minority_validation_data,
            "test_data": fold0_minority_test_data,
            "source_train_data": source_train_data,
            "target_train_data": target_train_data,
            "target_final_train_data": target_final_train_data,
            "final_train_data": merge_dicts(source_train_data, target_final_train_data),
            "metadata": {
                "split_source": "split_many_folds[0]",
                "validation": "minority samples from fold0_train_data",
                "test": "fold0_minority_test_data",
                "context_search": "KNN by class in normalized ctxae20 using source_train_data and AFR target train/validation as appropriate",
                "source_training_adds": "fold0_majority_test_data",
                "selection_metric": selection_metric,
                "candidate_strategies": ["min", "one_pass_by_class", "mix"] if method == "icl_one_pass_by_class_hs_ce" else None,
            },
        }

    train_data = dict(fold_train_majority)
    train_data.update(minority_train_data)
    if sample_mode_for_method(method) == "mix":
        overlap = set(train_data).intersection(fold0_majority_test_data)
        if overlap:
            raise ValueError(f"Unexpected overlap between training data and majority holdout: {len(overlap)}")
        train_data.update(fold0_majority_test_data)

    validation_data = minority_validation_data
    final_train_data = dict(train_data)
    final_train_data.update(validation_data)
    return {
        "train_data": train_data,
        "validation_data": validation_data,
        "test_data": fold0_minority_test_data,
        "final_train_data": final_train_data,
        "metadata": {
            "split_source": "split_many_folds[0]",
            "validation": "minority samples from fold0_train_data",
            "test": "fold0_minority_test_data",
            "mixed_training_adds": "fold0_majority_test_data" if sample_mode_for_method(method) == "mix" else None,
        },
    }


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    args = parse_args()
    split_seed = args.search_run if args.split_seed is None else args.split_seed
    set_global_seed(100000 + args.search_run)

    spath = split_path(args)
    if not spath.is_file():
        print(f"[skip] missing split file: {spath}")
        return
    with spath.open("rb") as handle:
        split_many_folds = pickle.load(handle)

    selected_ids = load_selected_ids()
    out_dir = result_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_prediction_path = out_dir / "best_candidate_test_prediction.pkl"
    if best_prediction_path.is_file() and not args.overwrite_existing:
        print(f"[skip] existing best test prediction: {best_prediction_path}")
        return

    seed_key = f"{args.method}|{args.cancer_type}|{args.expression_type}|{args.search_run}|{split_seed}"
    candidate_seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    knn_ks = parse_knn_ks(args.knn_ks)
    selection_metric = "validation_auroc" if args.method == "icl_knn_by_class_auroc" else "validation_cross_entropy"
    candidates = random_candidates(args.method, args.num_candidates, candidate_seed, knn_ks=knn_ks)
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(
            {
                "method": args.method,
                "cancer_type": args.cancer_type,
                "expression_type": args.expression_type,
                "ancestry": args.ancestry,
                "target": args.target,
                "years": args.year,
                "pca_dim": args.pca_dim,
                "folds": args.folds,
                "search_run": args.search_run,
                "split_seed": split_seed,
                "split_path": str(spath),
                "num_candidates": len(candidates),
                "time_limit_min": args.time_limit_min,
                "knn_ks": knn_ks if args.method in {"icl_knn_by_class_ce", "icl_knn_by_class_auroc"} else None,
                "candidate_strategies": [candidate.get("strategy") for candidate in candidates]
                if args.method == "icl_one_pass_by_class_hs_ce"
                else None,
                "selection_metric": selection_metric,
            },
            handle,
            indent=2,
        )

    split_bundle = get_train_validation_test(split_many_folds, split_seed, args.method, selected_ids)
    train_data = split_bundle["train_data"]
    validation_data = split_bundle["validation_data"]
    test_data = split_bundle["test_data"]

    start_wall = time.monotonic()
    budget_seconds = float(args.time_limit_min) * 60.0
    summary_path = out_dir / "candidate_summary.jsonl"
    best: dict[str, Any] | None = None
    evaluated = 0
    over_budget_candidate: dict[str, Any] | None = None

    with summary_path.open("w") as summary_handle:
        for candidate_index, params in enumerate(candidates):
            candidate_start_seconds = time.monotonic() - start_wall
            if candidate_start_seconds >= budget_seconds:
                print(f"[stop] reached {candidate_start_seconds / 60.0:.2f} min before candidate {candidate_index}")
                break

            print(
                f"[run] {args.method} {args.cancer_type} {args.expression_type} "
                f"search_run={args.search_run} candidate={candidate_index} elapsed_min={candidate_start_seconds / 60.0:.2f}",
                flush=True,
            )
            candidate_start = time.monotonic()
            if args.method == "ds_tl":
                ds_fit = fit_ds_tl_candidate(
                    params,
                    split_bundle["majority_train_data"],
                    split_bundle["majority_validation_data"],
                    split_bundle["minority_train_data"],
                    split_bundle["minority_validation_data"],
                )
                validation_output = {"time": []}
                extra = {}
            elif args.method in {"icl_knn_by_class_ce", "icl_knn_by_class_auroc"}:
                ds_fit = None
                validation_output, extra = evaluate_knn_by_class_icl(
                    args,
                    int(params["knn_k"]),
                    split_bundle["source_train_data"],
                    split_bundle["target_train_data"],
                    validation_data,
                )
            elif args.method == "icl_one_pass_by_class_hs_ce":
                ds_fit = None
                validation_output, extra = evaluate_one_pass_hs_icl(
                    args,
                    str(params["strategy"]),
                    split_bundle["source_train_data"],
                    split_bundle["target_train_data"],
                    validation_data,
                )
            else:
                ds_fit = None
                validation_output = evaluate_holdout(args, params, train_data, validation_data, selected_ids)
                extra = {}
            candidate_elapsed = time.monotonic() - candidate_start
            candidate_finish_seconds = time.monotonic() - start_wall
            if args.method == "ds_tl":
                val_ce = ds_fit["validation_cross_entropy"] if ds_fit is not None else None
                pretrain_val_ce = ds_fit["pretrain_validation_cross_entropy"] if ds_fit is not None else None
                val_auroc = None
            else:
                val_ce = validation_cross_entropy(validation_output, validation_data)
                val_auroc = validation_auroc(validation_output, validation_data)
                pretrain_val_ce = None
            candidate_score = val_auroc if selection_metric == "validation_auroc" else val_ce
            eligible = candidate_start_seconds < budget_seconds and candidate_score is not None
            summary_row = {
                "candidate_index": candidate_index,
                "params": params,
                "validation_cross_entropy": val_ce,
                "validation_auroc": val_auroc,
                "pretrain_validation_cross_entropy": pretrain_val_ce,
                "candidate_elapsed_seconds": candidate_elapsed,
                "candidate_start_seconds": candidate_start_seconds,
                "candidate_finish_seconds": candidate_finish_seconds,
                "eligible_within_budget": eligible,
                "validation_fold_time_seconds": validation_output["time"],
                "extra": extra,
            }
            summary_handle.write(json.dumps(summary_row, default=json_default) + "\n")
            summary_handle.flush()
            evaluated += 1

            if eligible and (
                best is None
                or (
                    selection_metric == "validation_auroc"
                    and candidate_score > best["selection_score"]
                )
                or (
                    selection_metric == "validation_cross_entropy"
                    and candidate_score < best["selection_score"]
                )
            ):
                best = {
                    "candidate_index": candidate_index,
                    "params": params,
                    "validation_cross_entropy": val_ce,
                    "validation_auroc": val_auroc,
                    "selection_score": candidate_score,
                    "pretrain_validation_cross_entropy": pretrain_val_ce,
                    "candidate_elapsed_seconds": candidate_elapsed,
                    "candidate_start_seconds": candidate_start_seconds,
                    "candidate_finish_seconds": candidate_finish_seconds,
                    "ds_tl_model": ds_fit["model"] if ds_fit is not None else None,
                    "ds_tl_scaler": ds_fit["scaler"] if ds_fit is not None else None,
                }

            if candidate_finish_seconds > budget_seconds:
                over_budget_candidate = summary_row
                print(
                    f"[budget] candidate {candidate_index} finished after budget "
                    f"({candidate_finish_seconds:.2f}s > {budget_seconds:.2f}s); using best started-within-budget candidate",
                    flush=True,
                )
                break

    if best is None:
        print(f"[done] no eligible candidate found out_dir={out_dir}")
        return

    print(
        f"[select] candidate={best['candidate_index']} "
        f"{selection_metric}={best['selection_score']:.6f}",
        flush=True,
    )
    test_start = time.monotonic()
    if args.method == "ds_tl":
        test_output = _predict_ds_tl_model(best["ds_tl_model"], best["ds_tl_scaler"], test_data)
        test_extra = {}
    elif args.method in {"icl_knn_by_class_ce", "icl_knn_by_class_auroc"}:
        test_output, test_extra = evaluate_knn_by_class_icl(
            args,
            int(best["params"]["knn_k"]),
            split_bundle["source_train_data"],
            split_bundle["target_final_train_data"],
            test_data,
        )
    elif args.method == "icl_one_pass_by_class_hs_ce":
        test_output, test_extra = evaluate_one_pass_hs_icl(
            args,
            str(best["params"]["strategy"]),
            split_bundle["source_train_data"],
            split_bundle["target_final_train_data"],
            test_data,
        )
    else:
        test_output = evaluate_holdout(args, best["params"], split_bundle["final_train_data"], test_data, selected_ids)
        test_extra = {}
    test_elapsed = time.monotonic() - test_start
    payload = {
        "pred": test_output["pred"],
        "time": test_output["time"],
        "sample_name": test_output["sample_name"],
        "params": best["params"],
        "candidate_index": best["candidate_index"],
        "search_run": args.search_run,
        "split_seed": split_seed,
        "validation_cross_entropy": best["validation_cross_entropy"],
        "validation_auroc": best.get("validation_auroc"),
        "pretrain_validation_cross_entropy": best.get("pretrain_validation_cross_entropy"),
        "candidate_elapsed_seconds": best["candidate_elapsed_seconds"],
        "candidate_start_seconds": best["candidate_start_seconds"],
        "candidate_finish_seconds": best["candidate_finish_seconds"],
        "budget_seconds": budget_seconds,
        "test_elapsed_seconds": test_elapsed,
        "selection_metric": selection_metric,
        "selection_score": best["selection_score"],
        "fold_role": split_bundle["metadata"],
        "test_extra": test_extra,
        "over_budget_candidate": over_budget_candidate,
    }
    with best_prediction_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[done] selected_candidate={best['candidate_index']} "
        f"test_prediction={best_prediction_path} out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
