import time

import numpy as np
import sklearn.utils.validation as sklearn_validation

if not hasattr(sklearn_validation, "_is_pandas_df") and hasattr(
    sklearn_validation, "is_pandas_df"
):
    sklearn_validation._is_pandas_df = sklearn_validation.is_pandas_df

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _missing_advanced_transfer_learning(*_args, **_kwargs):
    raise ImportError(
        "Missing advanced_transfer_learning.TabPFN.tabpfn_utils. "
        "Enhanced context-search models require that package to be available on PYTHONPATH."
    )


try:
    from advanced_transfer_learning.TabPFN.tabpfn_utils import (
        batched_search_context_cls,
        batched_search_context_cls_by_class,
        knn_graph_search_context_cls,
        radius_search_context_cls_with_param_search,
        scaled_radius_candidates,
        search_context_cls,
        search_context_cls_by_class,
    )
except ImportError:
    batched_search_context_cls = _missing_advanced_transfer_learning
    batched_search_context_cls_by_class = _missing_advanced_transfer_learning
    knn_graph_search_context_cls = _missing_advanced_transfer_learning
    radius_search_context_cls_with_param_search = _missing_advanced_transfer_learning
    scaled_radius_candidates = _missing_advanced_transfer_learning
    search_context_cls = _missing_advanced_transfer_learning
    search_context_cls_by_class = _missing_advanced_transfer_learning


def _normalize_augmented_feature_blocks(block_map):
    feature_blocks = []
    block_specs = []

    for name, X, y in block_map:
        X = np.asarray(X)

        if X.size == 0 or len(X) == 0:
            feat_dim = X.shape[1] if X.ndim == 2 else 0
            block_specs.append((name, 0, feat_dim))
            continue

        if X.ndim == 1:
            X = X.reshape(1, -1)

        feature_blocks.append(X)
        block_specs.append((name, len(X), X.shape[1]))

    if not feature_blocks:
        return {}

    combined_aug = np.vstack(feature_blocks)
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_aug)

    normalized_map = {}
    start = 0
    for name, length, feat_dim in block_specs:
        if length == 0:
            normalized_map[name] = np.empty((0, feat_dim), dtype=np.float32)
            continue
        normalized_map[name] = combined_scaled[start:start + length].astype(np.float32, copy=False)
        start += length
    return normalized_map


def _ensure_binary_predict_proba(model, train_X, train_y, test_X):
    train_y = np.asarray(train_y).reshape(-1)
    unique_classes = np.unique(train_y)

    if len(unique_classes) == 1:
        if int(unique_classes[0]) == 0:
            return np.hstack(
                (
                    np.ones((len(test_X), 1), dtype=float),
                    np.zeros((len(test_X), 1), dtype=float),
                )
            )
        return np.hstack(
            (
                np.zeros((len(test_X), 1), dtype=float),
                np.ones((len(test_X), 1), dtype=float),
            )
        )

    model.fit(train_X, train_y)
    pred_prob = model.predict_proba(test_X)
    classes = np.asarray(model.classes_).reshape(-1)

    if len(classes) == 1:
        if classes[0] == 0:
            return np.hstack(
                (
                    np.ones((len(test_X), 1), dtype=float),
                    np.zeros((len(test_X), 1), dtype=float),
                )
            )
        return np.hstack(
            (
                np.zeros((len(test_X), 1), dtype=float),
                np.ones((len(test_X), 1), dtype=float),
            )
        )

    if pred_prob.shape[1] != 2:
        raise ValueError(f"Expected binary predict_proba output with 2 columns, got {pred_prob.shape}.")

    if list(classes) == [0, 1]:
        return pred_prob

    class_to_col = {int(cls): idx for idx, cls in enumerate(classes)}
    return np.column_stack((pred_prob[:, class_to_col[0]], pred_prob[:, class_to_col[1]]))


def _prepare_discrete_predictive_arrays(train_data, test_data, selected_id=None, mode="mix"):
    if selected_id is not None:
        selected_id = set(selected_id)

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

    if not train_X:
        if mode != "mix":
            # Some sparse folds can lose all retained samples under maj/min filtering.
            # Fall back to the full discrete training set so the run can complete.
            return _prepare_discrete_predictive_arrays(
                train_data,
                test_data,
                selected_id=None,
                mode="mix",
            )
        raise ValueError("No training samples available after discrete stratification filtering.")

    feat_dim = train_X[0].shape[-1]
    train_X = np.asarray(train_X, dtype=np.float32).reshape((-1, feat_dim))
    train_y = np.asarray(train_y, dtype=int).reshape(-1)

    if test_X:
        test_X = np.asarray(test_X, dtype=np.float32).reshape((-1, feat_dim))
    else:
        test_X = np.zeros((0, feat_dim), dtype=np.float32)

    return train_X, train_y, test_X, test_sample_id


def _prepare_merge_ancestry_omics_arrays(train_data, test_data):
    train_X = []
    train_y = []
    test_X = []

    for _, val in train_data.items():
        predictive_feat = np.asarray(val[0], dtype=np.float32).reshape(-1)
        ancestry_feat = np.asarray(val[1], dtype=np.float32).reshape(-1)
        label = int(np.asarray(val[2]).reshape(-1)[0])
        train_X.append(np.concatenate((predictive_feat, ancestry_feat)))
        train_y.append(label)

    test_sample_id = list(test_data.keys())
    for key in test_sample_id:
        predictive_feat = np.asarray(test_data[key][0], dtype=np.float32).reshape(-1)
        ancestry_feat = np.asarray(test_data[key][1], dtype=np.float32).reshape(-1)
        test_X.append(np.concatenate((predictive_feat, ancestry_feat)))

    if not train_X:
        raise ValueError("No training samples available for merged omics-plus-genetics modeling.")

    feat_dim = train_X[0].shape[-1]
    train_X = np.asarray(train_X, dtype=np.float32).reshape((-1, feat_dim))
    train_y = np.asarray(train_y, dtype=int).reshape(-1)

    if test_X:
        test_X = np.asarray(test_X, dtype=np.float32).reshape((-1, feat_dim))
    else:
        test_X = np.zeros((0, feat_dim), dtype=np.float32)

    return train_X, train_y, test_X, test_sample_id


def _prepare_sum_ancestry_omics_arrays(train_data, test_data):
    train_omics = []
    train_genetics = []
    train_y = []
    test_omics = []
    test_genetics = []

    for _, val in train_data.items():
        predictive_feat = np.asarray(val[0], dtype=np.float32).reshape(-1)
        ancestry_feat = np.asarray(val[1], dtype=np.float32).reshape(-1)
        label = int(np.asarray(val[2]).reshape(-1)[0])
        if predictive_feat.shape[-1] != ancestry_feat.shape[-1]:
            raise ValueError(
                "Summed ancestry-omics modeling requires omics and genetics to share the same dimensionality, "
                f"got {predictive_feat.shape[-1]} and {ancestry_feat.shape[-1]}."
            )
        train_omics.append(predictive_feat)
        train_genetics.append(ancestry_feat)
        train_y.append(label)

    test_sample_id = list(test_data.keys())
    for key in test_sample_id:
        predictive_feat = np.asarray(test_data[key][0], dtype=np.float32).reshape(-1)
        ancestry_feat = np.asarray(test_data[key][1], dtype=np.float32).reshape(-1)
        if predictive_feat.shape[-1] != ancestry_feat.shape[-1]:
            raise ValueError(
                "Summed ancestry-omics modeling requires omics and genetics to share the same dimensionality, "
                f"got {predictive_feat.shape[-1]} and {ancestry_feat.shape[-1]}."
            )
        test_omics.append(predictive_feat)
        test_genetics.append(ancestry_feat)

    if not train_omics:
        raise ValueError("No training samples available for summed ancestry-plus-omics modeling.")

    feat_dim = train_omics[0].shape[-1]
    train_omics = np.asarray(train_omics, dtype=np.float32).reshape((-1, feat_dim))
    train_genetics = np.asarray(train_genetics, dtype=np.float32).reshape((-1, feat_dim))
    train_y = np.asarray(train_y, dtype=int).reshape(-1)

    if test_omics:
        test_omics = np.asarray(test_omics, dtype=np.float32).reshape((-1, feat_dim))
        test_genetics = np.asarray(test_genetics, dtype=np.float32).reshape((-1, feat_dim))
    else:
        test_omics = np.zeros((0, feat_dim), dtype=np.float32)
        test_genetics = np.zeros((0, feat_dim), dtype=np.float32)

    omics_scaler = StandardScaler()
    genetics_scaler = StandardScaler()
    train_omics = omics_scaler.fit_transform(train_omics)
    train_genetics = genetics_scaler.fit_transform(train_genetics)
    test_omics = omics_scaler.transform(test_omics)
    test_genetics = genetics_scaler.transform(test_genetics)

    train_X = train_omics + train_genetics
    test_X = test_omics + test_genetics
    return train_X.astype(np.float32), train_y, test_X.astype(np.float32), test_sample_id


def _append_classical_predictions(model, train_X, train_y, test_X, test_sample_id, prediction_dic):
    start = time.perf_counter()
    if len(test_X) > 0:
        pred_prob = _ensure_binary_predict_proba(model, train_X, train_y, test_X)
        end = time.perf_counter()
        prediction_dic["pred"].append(np.asarray(pred_prob, dtype=float).reshape((-1, 2)))
        prediction_dic["sample_name"].append(test_sample_id)
        prediction_dic["time"].append(end - start)
    else:
        prediction_dic["pred"].append(np.zeros((0, 2)))
        prediction_dic["sample_name"].append(test_sample_id)
        prediction_dic["time"].append(-1)


def _prepare_enhanced_discrete_predictive_arrays(train_data, test_data, selected_id):
    selected_id = set(selected_id)

    train_src_X = []
    train_src_y = []
    train_src_context_X = []
    train_tgt_X = []
    train_tgt_y = []
    train_tgt_context_X = []
    test_X = []

    for key, val in train_data.items():
        predictive_feat = np.asarray(val[0], dtype=np.float32).reshape(-1)
        context_feat = np.asarray(val[3] if len(val) > 3 else val[0], dtype=np.float32).reshape(-1)
        label = int(np.asarray(val[2]).reshape(-1)[0])

        if key in selected_id:
            train_tgt_X.append(predictive_feat)
            train_tgt_y.append(label)
            train_tgt_context_X.append(context_feat)
        else:
            train_src_X.append(predictive_feat)
            train_src_y.append(label)
            train_src_context_X.append(context_feat)

    test_sample_id = list(test_data.keys())
    for key in test_sample_id:
        predictive_feat = np.asarray(test_data[key][0], dtype=np.float32).reshape(-1)
        test_X.append(predictive_feat)

    if not train_tgt_X:
        raise ValueError("No minority training samples available for enhanced discrete stratification.")
    if not train_src_X:
        raise ValueError("No majority training samples available for enhanced discrete stratification.")

    pred_dim = train_tgt_X[0].shape[-1]
    train_src_X = np.asarray(train_src_X, dtype=np.float32).reshape((-1, pred_dim))
    train_tgt_X = np.asarray(train_tgt_X, dtype=np.float32).reshape((-1, pred_dim))
    train_src_y = np.asarray(train_src_y, dtype=int).reshape(-1)
    train_tgt_y = np.asarray(train_tgt_y, dtype=int).reshape(-1)
    train_src_context_X = np.asarray(train_src_context_X, dtype=np.float32).reshape((len(train_src_context_X), -1))
    train_tgt_context_X = np.asarray(train_tgt_context_X, dtype=np.float32).reshape((len(train_tgt_context_X), -1))

    if test_X:
        test_X = np.asarray(test_X, dtype=np.float32).reshape((-1, pred_dim))
    else:
        test_X = np.zeros((0, pred_dim), dtype=np.float32)

    return (
        train_src_X,
        train_src_y,
        train_src_context_X,
        train_tgt_X,
        train_tgt_y,
        train_tgt_context_X,
        test_X,
        test_sample_id,
    )


def _build_enhanced_discrete_training_set(
    train_data,
    selected_id,
    *,
    context_search_mode="one_pass",
    distance_metric="euclidean",
):
    (
        train_src_X,
        train_src_y,
        train_src_context_X,
        train_tgt_X,
        train_tgt_y,
        train_tgt_context_X,
        _,
        _,
    ) = _prepare_enhanced_discrete_predictive_arrays(
        train_data,
        {},
        selected_id=selected_id,
    )

    normalized_search_map = _normalize_augmented_feature_blocks(
        [
            ("train_src", train_src_context_X, train_src_y),
            ("train_tgt", train_tgt_context_X, train_tgt_y),
        ]
    )
    train_src_context_X = normalized_search_map["train_src"]
    train_tgt_context_X = normalized_search_map["train_tgt"]

    knn_graph_k = 10
    if context_search_mode == "batch":
        batch_size = max(1, len(train_tgt_X))
        similar_ind = batched_search_context_cls(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            distance_metric=distance_metric,
            batch_size=batch_size,
        )
    elif context_search_mode == "batch_by_class":
        batch_size = max(1, len(train_tgt_X))
        similar_ind = batched_search_context_cls_by_class(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            distance_metric=distance_metric,
            batch_size=batch_size,
        )
    elif context_search_mode == "one_pass":
        similar_ind = search_context_cls(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            distance_metric=distance_metric,
        )
    elif context_search_mode == "one_pass_by_class":
        similar_ind = search_context_cls_by_class(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            distance_metric=distance_metric,
        )
    elif context_search_mode == "batch_knn":
        similar_ind = knn_graph_search_context_cls(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            k=knn_graph_k,
            distance_metric=distance_metric,
        )
    elif context_search_mode == "radius":
        radius = scaled_radius_candidates(train_src_context_X.shape[-1])
        similar_ind = radius_search_context_cls_with_param_search(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            radii=radius,
            distance_metric=distance_metric,
        )
    else:
        raise ValueError(
            f"Unsupported context_search_mode='{context_search_mode}'. "
            "Supported modes are 'one_pass', 'one_pass_by_class', 'batch', 'batch_by_class', 'batch_knn', and 'radius'."
        )

    ext_X = np.vstack((train_src_X[similar_ind], train_tgt_X))
    ext_y = np.concatenate((train_src_y[similar_ind], train_tgt_y)).ravel()
    return ext_X, ext_y


def _develop_ds_enhanced_classical_raw_sample_split(
    model,
    train_data,
    test_data,
    prediction_dic,
    *,
    selected_id,
    context_search_mode="one_pass",
    distance_metric="euclidean",
):
    (
        _train_src_X,
        _train_src_y,
        _train_src_context_X,
        _train_tgt_X,
        _train_tgt_y,
        _train_tgt_context_X,
        test_X,
        test_sample_id,
    ) = _prepare_enhanced_discrete_predictive_arrays(
        train_data,
        test_data,
        selected_id=selected_id,
    )

    ext_X, ext_y = _build_enhanced_discrete_training_set(
        train_data,
        selected_id,
        context_search_mode=context_search_mode,
        distance_metric=distance_metric,
    )
    _append_classical_predictions(model, ext_X, ext_y, test_X, test_sample_id, prediction_dic)


def develop_DS_ElasticNet_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
    selected_id=None,
    mode="mix",
):
    train_X, train_y, test_X, test_sample_id = _prepare_discrete_predictive_arrays(
        train_data,
        test_data,
        selected_id=selected_id,
        mode=mode,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=0,
                ),
            ),
        ]
    )
    _append_classical_predictions(model, train_X, train_y, test_X, test_sample_id, prediction_dic)


def develop_merge_ancestry_omics_ElasticNet_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
):
    train_X, train_y, test_X, test_sample_id = _prepare_merge_ancestry_omics_arrays(
        train_data,
        test_data,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=0,
                ),
            ),
        ]
    )
    _append_classical_predictions(model, train_X, train_y, test_X, test_sample_id, prediction_dic)


def develop_sum_ancestry_omics_ElasticNet_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
):
    train_X, train_y, test_X, test_sample_id = _prepare_sum_ancestry_omics_arrays(
        train_data,
        test_data,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=0,
                ),
            ),
        ]
    )
    _append_classical_predictions(model, train_X, train_y, test_X, test_sample_id, prediction_dic)


def develop_DS_RandomForest_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
    selected_id=None,
    mode="mix",
):
    train_X, train_y, test_X, test_sample_id = _prepare_discrete_predictive_arrays(
        train_data,
        test_data,
        selected_id=selected_id,
        mode=mode,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    _append_classical_predictions(model, train_X, train_y, test_X, test_sample_id, prediction_dic)


def develop_merge_ancestry_omics_RandomForest_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
):
    train_X, train_y, test_X, test_sample_id = _prepare_merge_ancestry_omics_arrays(
        train_data,
        test_data,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    _append_classical_predictions(model, train_X, train_y, test_X, test_sample_id, prediction_dic)


def develop_sum_ancestry_omics_RandomForest_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
):
    train_X, train_y, test_X, test_sample_id = _prepare_sum_ancestry_omics_arrays(
        train_data,
        test_data,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    _append_classical_predictions(model, train_X, train_y, test_X, test_sample_id, prediction_dic)


def develop_DS_XGBoost_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
    selected_id=None,
    mode="mix",
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for Discrete_Stractification_XGBoost_AI but is not installed in the active environment."
        ) from exc

    train_X, train_y, test_X, test_sample_id = _prepare_discrete_predictive_arrays(
        train_data,
        test_data,
        selected_id=selected_id,
        mode=mode,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=0,
                    n_jobs=8,
                ),
            ),
        ]
    )
    _append_classical_predictions(model, train_X, train_y, test_X, test_sample_id, prediction_dic)


def develop_merge_ancestry_omics_XGBoost_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for merge_ancestry_omics_XGBoost_AI but is not installed in the active environment."
        ) from exc

    train_X, train_y, test_X, test_sample_id = _prepare_merge_ancestry_omics_arrays(
        train_data,
        test_data,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=0,
                    n_jobs=8,
                ),
            ),
        ]
    )
    _append_classical_predictions(model, train_X, train_y, test_X, test_sample_id, prediction_dic)


def develop_sum_ancestry_omics_XGBoost_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for sum_ancestry_omics_XGBoost_AI but is not installed in the active environment."
        ) from exc

    train_X, train_y, test_X, test_sample_id = _prepare_sum_ancestry_omics_arrays(
        train_data,
        test_data,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=0,
                    n_jobs=8,
                ),
            ),
        ]
    )
    _append_classical_predictions(model, train_X, train_y, test_X, test_sample_id, prediction_dic)


def develop_DS_Enhanced_ElasticNet_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
    selected_id,
    context_search_mode="one_pass",
    distance_metric="euclidean",
):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=0,
                ),
            ),
        ]
    )
    _develop_ds_enhanced_classical_raw_sample_split(
        model,
        train_data,
        test_data,
        prediction_dic,
        selected_id=selected_id,
        context_search_mode=context_search_mode,
        distance_metric=distance_metric,
    )


def develop_DS_Enhanced_RandomForest_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
    selected_id,
    context_search_mode="one_pass",
    distance_metric="euclidean",
):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    _develop_ds_enhanced_classical_raw_sample_split(
        model,
        train_data,
        test_data,
        prediction_dic,
        selected_id=selected_id,
        context_search_mode=context_search_mode,
        distance_metric=distance_metric,
    )


def develop_DS_Enhanced_XGBoost_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
    selected_id,
    context_search_mode="one_pass",
    distance_metric="euclidean",
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for Discrete_Stractification_Enhanced_XGBoost_AI but is not installed in the active environment."
        ) from exc

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=0,
                    n_jobs=8,
                ),
            ),
        ]
    )
    _develop_ds_enhanced_classical_raw_sample_split(
        model,
        train_data,
        test_data,
        prediction_dic,
        selected_id=selected_id,
        context_search_mode=context_search_mode,
        distance_metric=distance_metric,
    )
