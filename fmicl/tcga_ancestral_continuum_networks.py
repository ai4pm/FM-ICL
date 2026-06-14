import importlib


import numpy as np
import sklearn.utils.validation as sklearn_validation

if not hasattr(sklearn_validation, "_is_pandas_df") and hasattr(
    sklearn_validation, "is_pandas_df"
):
    sklearn_validation._is_pandas_df = sklearn_validation.is_pandas_df

from tabpfn import TabPFNClassifier
import torch
import time
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tcga_training_utils import hyperparam_search_cv


def _missing_advanced_transfer_learning(*_args, **_kwargs):
    raise ImportError(
        "Missing advanced_transfer_learning dependencies. "
        "ICL and transfer-learning models require the advanced_transfer_learning package tree to be available."
    )


class _MissingAdvancedTransferLearningHParams:
    def __init__(self, *_args, **_kwargs):
        _missing_advanced_transfer_learning()


_NN_BACKEND_IMPORT_ERROR = None


def _load_nn_backend():
    global _NN_BACKEND_IMPORT_ERROR
    global _probs_from_logits_binary, nn_HParams, develop_TL_NN_models, train_nn, train_nn_ddp

    try:
        module = importlib.import_module(
            "advanced_transfer_learning.Multimodal_NN.Multimodal_nn_utils"
        )
    except Exception as exc:
        _NN_BACKEND_IMPORT_ERROR = exc
        _probs_from_logits_binary = _missing_advanced_transfer_learning
        nn_HParams = _MissingAdvancedTransferLearningHParams
        develop_TL_NN_models = _missing_advanced_transfer_learning
        train_nn = _missing_advanced_transfer_learning
        train_nn_ddp = _missing_advanced_transfer_learning
        return False

    _NN_BACKEND_IMPORT_ERROR = None
    _probs_from_logits_binary = module._probs_from_logits_binary
    nn_HParams = module.HParams
    develop_TL_NN_models = module.develop_multimodal_base_models
    train_nn = module.train_binary_mlp
    train_nn_ddp = module.ddp_train_nn
    return True


def _require_nn_backend():
    if _load_nn_backend():
        return
    raise ImportError(
        "Failed to import advanced_transfer_learning.Multimodal_NN.Multimodal_nn_utils"
    ) from _NN_BACKEND_IMPORT_ERROR


def _resolve_tl_trainer(architecture):
    _require_nn_backend()
    trainers = {
        "nn": train_nn,
        "nn_ddp": train_nn_ddp,
    }
    if architecture not in trainers:
        print(
            f"Requested TL architecture '{architecture}' is unavailable in this workspace; "
            "falling back to 'nn'."
        )
        architecture = "nn"
    return architecture, trainers[architecture]


try:
    from advanced_transfer_learning.TabPFN.tabpfn_utils import _fit_tabpfn, batched_search_context_cls, batched_search_context_cls_by_class, search_context_cls, search_context_cls_by_class, knn_graph_search_context_cls, radius_search_context_cls, radius_search_context_cls_with_param_search, cont_radius_search_context_cls_with_param_search, cont_radius_search_context_cls_with_param_search_by_class, scaled_radius_candidates, develop_multimodal_base_models as develop_ICL_AI_models
except ImportError:
    _fit_tabpfn = _missing_advanced_transfer_learning
    batched_search_context_cls = _missing_advanced_transfer_learning
    batched_search_context_cls_by_class = _missing_advanced_transfer_learning
    search_context_cls = _missing_advanced_transfer_learning
    search_context_cls_by_class = _missing_advanced_transfer_learning
    knn_graph_search_context_cls = _missing_advanced_transfer_learning
    radius_search_context_cls = _missing_advanced_transfer_learning
    radius_search_context_cls_with_param_search = _missing_advanced_transfer_learning
    cont_radius_search_context_cls_with_param_search = _missing_advanced_transfer_learning
    cont_radius_search_context_cls_with_param_search_by_class = _missing_advanced_transfer_learning
    scaled_radius_candidates = _missing_advanced_transfer_learning
    develop_ICL_AI_models = _missing_advanced_transfer_learning

_probs_from_logits_binary = _missing_advanced_transfer_learning
nn_HParams = _MissingAdvancedTransferLearningHParams
develop_TL_NN_models = _missing_advanced_transfer_learning
train_nn = _missing_advanced_transfer_learning
train_nn_ddp = _missing_advanced_transfer_learning
_load_nn_backend()

try:
    from advanced_transfer_learning.resnet_finetune.resnet_utils import HParams as resnet_HParams, train_one_model as train_resnet, develop_multimodal_base_models as develop_TL_resnet_models
except ImportError:
    resnet_HParams = _MissingAdvancedTransferLearningHParams
    train_resnet = _missing_advanced_transfer_learning
    develop_TL_resnet_models = _missing_advanced_transfer_learning

try:
    from advanced_transfer_learning.transformer_finetune.transformer_utils import HParams as transformer_HParams, train_one_model as train_transformer, develop_multimodal_base_models as develop_TL_transformer_models
except ImportError:
    transformer_HParams = _MissingAdvancedTransferLearningHParams
    train_transformer = _missing_advanced_transfer_learning
    develop_TL_transformer_models = _missing_advanced_transfer_learning

# =================================
# module for various AI prediction approaches 
# =================================
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

        normalized_dim = X.shape[1]
        feature_blocks.append(X)
        block_specs.append((name, len(X), normalized_dim))

    if not feature_blocks:
        return {}

    combined_aug = np.vstack(feature_blocks)
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_aug)

    normalized_map = {}
    start = 0
    for name, length, normalized_dim in block_specs:
        if length == 0:
            normalized_map[name] = np.empty((0, normalized_dim), dtype=np.float32)
            continue
        normalized_map[name] = combined_scaled[start:start + length].astype(np.float32, copy=False)
        start += length
    return normalized_map


def _ensure_binary_predict_proba(model, train_X, train_y, test_X):
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
        raise ValueError("No training samples available after discrete stratification filtering.")

    feat_dim = train_X[0].shape[-1]
    train_X = np.asarray(train_X, dtype=np.float32).reshape((-1, feat_dim))
    train_y = np.asarray(train_y, dtype=int).reshape(-1)

    if test_X:
        test_X = np.asarray(test_X, dtype=np.float32).reshape((-1, feat_dim))
    else:
        test_X = np.zeros((0, feat_dim), dtype=np.float32)

    return train_X, train_y, test_X, test_sample_id


def _prepare_discrete_enhanced_icl_arrays(train_data, test_data, selected_id):
    selected_id = set(selected_id)

    train_src_X = []
    train_src_y = []
    train_src_context_X = []
    train_src_ids = []
    train_tgt_X = []
    train_tgt_y = []
    train_tgt_context_X = []
    train_tgt_ids = []
    test_X = []

    for key, val in train_data.items():
        predictive_feat = np.asarray(val[0], dtype=np.float32).reshape(-1)
        context_feat = np.asarray(val[3] if len(val) > 3 else val[0], dtype=np.float32).reshape(-1)
        label = int(np.asarray(val[2]).reshape(-1)[0])

        if key in selected_id:
            train_tgt_X.append(predictive_feat)
            train_tgt_y.append(label)
            train_tgt_context_X.append(context_feat)
            train_tgt_ids.append(key)
        else:
            train_src_X.append(predictive_feat)
            train_src_y.append(label)
            train_src_context_X.append(context_feat)
            train_src_ids.append(key)

    if not train_src_X:
        raise ValueError("No majority training samples available for enhanced discrete stratification ICL.")
    if not train_tgt_X:
        raise ValueError("No minority training samples available for enhanced discrete stratification ICL.")

    feat_dim = train_src_X[0].shape[-1]
    context_dim = train_src_context_X[0].shape[-1]

    train_src_X = np.asarray(train_src_X, dtype=np.float32).reshape((-1, feat_dim))
    train_tgt_X = np.asarray(train_tgt_X, dtype=np.float32).reshape((-1, feat_dim))
    train_src_y = np.asarray(train_src_y, dtype=int).reshape(-1)
    train_tgt_y = np.asarray(train_tgt_y, dtype=int).reshape(-1)
    train_src_context_X = np.asarray(train_src_context_X, dtype=np.float32).reshape((-1, context_dim))
    train_tgt_context_X = np.asarray(train_tgt_context_X, dtype=np.float32).reshape((-1, context_dim))

    test_sample_id = list(test_data.keys())
    for key in test_sample_id:
        predictive_feat = np.asarray(test_data[key][0], dtype=np.float32).reshape(-1)
        test_X.append(predictive_feat)

    if test_X:
        test_X = np.asarray(test_X, dtype=np.float32).reshape((-1, feat_dim))
    else:
        test_X = np.zeros((0, feat_dim), dtype=np.float32)

    return (
        train_src_X,
        train_src_y,
        train_src_context_X,
        train_tgt_X,
        train_tgt_y,
        train_tgt_context_X,
        test_X,
        test_sample_id,
        train_src_ids,
        train_tgt_ids,
    )


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


def develop_DS_ElasticNet_AI_raw_sample_split(train_data, test_data, prediction_dic):
    train_X, train_y, test_X, test_sample_id = _prepare_discrete_predictive_arrays(
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


def develop_DS_RandomForest_AI_raw_sample_split(train_data, test_data, prediction_dic):
    train_X, train_y, test_X, test_sample_id = _prepare_discrete_predictive_arrays(
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


def develop_DS_XGBoost_AI_raw_sample_split(train_data, test_data, prediction_dic):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for Discrete_Stractification_XGBoost_AI but is not installed in the active environment."
        ) from exc

    train_X, train_y, test_X, test_sample_id = _prepare_discrete_predictive_arrays(
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


def develop_Discrete_Stractification_TL_AI(all_tr_sample_names, all_tr, ancestral_data_tr, test_data, prediction_dic, selected_id, hyper_params_option = 1, architecture='nn'):
    # Repro
   
    # model specificiation  
    x_te = test_data[0]; 

    # get test samples 
    test_sample_names = []; test_ind = []
    for i, sn in enumerate(ancestral_data_tr):
        if sn in selected_id:
            test_sample_names.append(sn)
            test_ind.append(i)
    test_ind = np.array(test_ind, dtype=int)

    domain_train_sample_names =[]
    for s in all_tr_sample_names:
        if s in selected_id and not s in test_sample_names:
            domain_train_sample_names.append(s)

    # sample_name -> id dictionary
    sample_id_dic = {}
    for i, sn in enumerate(all_tr_sample_names):
        sample_id_dic[sn] = i 

    all_train_X = all_tr[0]; all_train_y = all_tr[1]
    domain_train_X = []; domain_train_y = []
    for sn in domain_train_sample_names:
        domain_train_X.append(all_train_X[sample_id_dic[sn]])
        domain_train_y.append(all_train_y[sample_id_dic[sn]])
    all_train_X = np.array(all_train_X); all_train_y = np.array(all_train_y)
    domain_train_X = np.array(domain_train_X); domain_train_y = np.array(domain_train_y)

    # parameter to search 
    if hyper_params_option == 1:
        batch_sizes=(8, )
        learning_rates=(1e-3, )
        dropouts=(0.0, )
        L1_reg = (1e-6, )
        L2_reg = (1e-7, )
    elif hyper_params_option == 2:
        # batch_sizes=(8, 16)
        # learning_rates=(1e-3, 3e-3)
        # dropouts=(0.0, 0.1)
        # L1_reg = (1e-6, 1e-5)
        # L2_reg = (1e-7, 3e-7)
        batch_sizes=(16,)
        learning_rates=(1e-3,)
        dropouts=(0.1,)
        L1_reg = (1e-6, 1e-5)
        L2_reg = (1e-7, 3e-7)
    elif hyper_params_option == 3:
        batch_sizes=(8, 16, 32)
        learning_rates=(1e-3, 3e-3, 1e-2)
        dropouts=(0.0, 0.1, 0.3)
        L1_reg = (1e-6, 1e-5, 1e-4)
        L2_reg = (1e-7, 3e-7, 1e-6)
    elif hyper_params_option == 4:
        batch_sizes=(8, 16, 32, 64)
        learning_rates=(1e-3, 3e-3, 1e-2, 3e-2)
        dropouts=(0.0, 0.1, 0.3, 0.5)
        L1_reg = (1e-6, 1e-5, 1e-4, 1e-3)
        L2_reg = (1e-7, 3e-7, 1e-6, 3e-6)

    architecture, trainer = _resolve_tl_trainer(architecture)

    base_hparams = hyperparam_search_cv(all_train_X, all_train_y,
                        train_AI_model=trainer,
                        HParams=nn_HParams,
                        batch_sizes=batch_sizes,
                        learning_rates=learning_rates,
                        L1_reg=L1_reg,
                        L2_reg=L2_reg,
                        dropouts=dropouts,
                        n_epochs =50,
                        )
    
    base_model, _ = trainer(all_train_X, all_train_y, h=base_hparams)


    finetune_hparams = hyperparam_search_cv(domain_train_X, domain_train_y,
                        train_AI_model=trainer,
                        HParams=nn_HParams,
                        model = base_model, 
                        batch_sizes=batch_sizes,
                        learning_rates=learning_rates,
                        L1_reg=L1_reg,
                        L2_reg=L2_reg,
                        dropouts=dropouts,
                        n_epochs =10,
                        )

    tuned_model, _ = trainer(domain_train_X, domain_train_y, model=base_model,h=finetune_hparams)
    device = next(tuned_model.parameters()).device
 

    # make predictions and save 
    tuned_model.eval() 
    with torch.no_grad():
        logits = tuned_model(torch.tensor(x_te[test_ind], dtype=torch.float32).to(device)).squeeze(1).cpu().numpy()
    pred_prob = _probs_from_logits_binary(logits)
    prediction_dic['pred'].append(np.array(pred_prob).reshape((-1, 2)))
    prediction_dic['sample_name'].append(test_sample_names)

# ICL_AI for discrete ancestry
# ICL_AI for discrete ancestry
def develop_Discrete_Stractification_ICL_AI(all_tr_sample_names, all_tr, ancestral_data_tr, test_data, prediction_dic, selected_id):

    # model specificiation  
    x_te = test_data[0]; 

    # get test samples 
    test_sample_names = []; test_ind = []
    for i, sn in enumerate(ancestral_data_tr):
        if sn in selected_id:
            test_sample_names.append(sn)
            test_ind.append(i)
    test_ind = np.array(test_ind, dtype=int)
    if len(test_sample_names) == 0:
        prediction_dic['pred'].append(np.zeros((0, 2)))
        prediction_dic['sample_name'].append(test_sample_names)
    else:
        domain_train_sample_names =[]
        for s in all_tr_sample_names:
            if s in selected_id and not s in test_sample_names:
                domain_train_sample_names.append(s)

        # sample_name -> id dictionary
        sample_id_dic = {}
        for i, sn in enumerate(all_tr_sample_names):
            sample_id_dic[sn] = i 

        all_train_X = all_tr[0]; all_train_y = all_tr[1]
        domain_train_X = []; domain_train_y = []
        for sn in domain_train_sample_names:
            domain_train_X.append(all_train_X[sample_id_dic[sn]])
            domain_train_y.append(all_train_y[sample_id_dic[sn]])

        domain_train_X = np.array(domain_train_X); domain_train_y = np.array(domain_train_y)


        # variables to be used 
        pred_prob = []
        model = TabPFNClassifier()

        model.fit(domain_train_X, domain_train_y)

        pred_prob =  model.predict_proba(x_te[test_ind])
        prediction_dic['pred'].append(pred_prob.reshape((-1, 2)))
        prediction_dic['sample_name'].append(test_sample_names)



def develop_Discrete_Stractification_Enhanced_ICL_AI(all_tr_sample_names, all_tr, ancestral_data_tr, test_data, prediction_dic, context_search_mode='one_pass', selected_id = None, distance_metric='euclidean'):
    # model specificiation  
    x_te = test_data[0]; 

    # get test samples 
    test_sample_names = []; test_ind = []
    for i, sn in enumerate(ancestral_data_tr):
        if sn in selected_id:
            test_sample_names.append(sn)
            test_ind.append(i)
    test_ind = np.array(test_ind, dtype=int)

    if len(test_sample_names) == 0:
        prediction_dic['pred'].append(np.zeros((0, 2)))
        prediction_dic['sample_name'].append(test_sample_names)
    else:
        domain_train_sample_names =[]
        out_of_domain_train_sample_names = []
        for s in all_tr_sample_names:
            if s in selected_id and not s in test_sample_names:
                domain_train_sample_names.append(s)
            elif not s in selected_id:
                out_of_domain_train_sample_names.append(s)

        # sample_name -> id dictionary
        sample_id_dic = {}
        for i, sn in enumerate(all_tr_sample_names):
            sample_id_dic[sn] = i 

        all_train_X = all_tr[0]; all_train_y = all_tr[1]
        domain_train_X = []; domain_train_y = [];  
        out_of_domain_train_X = []; out_of_domain_train_y = []
        for sn in domain_train_sample_names:
            domain_train_X.append(all_train_X[sample_id_dic[sn]])
            domain_train_y.append(all_train_y[sample_id_dic[sn]])

        for sn in out_of_domain_train_sample_names:
            out_of_domain_train_X.append(all_train_X[sample_id_dic[sn]])
            out_of_domain_train_y.append(all_train_y[sample_id_dic[sn]])

        domain_train_X = np.array(domain_train_X); domain_train_y = np.array(domain_train_y)
        out_of_domain_train_X = np.array(out_of_domain_train_X); out_of_domain_train_y = np.array(out_of_domain_train_y)
        if context_search_mode == 'batch':
            similar_ind = batched_search_context_cls(out_of_domain_train_X, out_of_domain_train_y, domain_train_X, domain_train_y, distance_metric=distance_metric)
        elif context_search_mode == 'one_pass':
            similar_ind = search_context_cls(out_of_domain_train_X, out_of_domain_train_y, domain_train_X, domain_train_y, distance_metric=distance_metric)
        print(f'out of domain population size: {len(out_of_domain_train_X)} | training size: {len(domain_train_X)} | extended size: {len(similar_ind)}')
        
        # obtain extended training set
        ext_X = np.vstack([out_of_domain_train_X[similar_ind], domain_train_X])
        ext_Y = np.hstack([out_of_domain_train_y[similar_ind], domain_train_y])

        model = TabPFNClassifier()

        # make prediction with tabpfn using extended context 
        model.fit(ext_X, ext_Y)

        pred_prob =  model.predict_proba(x_te[test_ind])
        prediction_dic['pred'].append(pred_prob.reshape((-1, 2)))
        prediction_dic['sample_name'].append(test_sample_names)

def develop_TL_AI(all_tr, ancestral_data_tr, test_data, prediction_dic, hyper_params_option = 1, selected_id = None, architecture='nn'):
    # Repro
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # model specificiation  
    x_te = test_data[0]; 


    # parameter to search 
    # batch_sizes=(16, 32, 64)
    # learning_rates=(1e-3, 3e-3, 1e-2)
    # dropouts=(0.0, 0.1, 0.3, 0.5)
    # L1_reg = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    # L2_reg = (0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5)

    if hyper_params_option == 1:
        batch_sizes=(8, )
        learning_rates=(1e-3, )
        dropouts=(0.0, )
        L1_reg = (1e-6, )
        L2_reg = (1e-7, )
    elif hyper_params_option == 2:
        batch_sizes=(8, 16)
        learning_rates=(1e-3, 3e-3)
        dropouts=(0.0, 0.1)
        L1_reg = (1e-6, 1e-5)
        L2_reg = (1e-7, 3e-7)
    elif hyper_params_option == 3:
        batch_sizes=(8, 16, 32)
        learning_rates=(1e-3, 3e-3, 1e-2)
        dropouts=(0.0, 0.1, 0.3)
        L1_reg = (1e-6, 1e-5, 1e-4)
        L2_reg = (1e-7, 3e-7, 1e-6)
    elif hyper_params_option == 4:
        batch_sizes=(8, 16, 32, 64)
        learning_rates=(1e-3, 3e-3, 1e-2, 3e-2)
        dropouts=(0.0, 0.1, 0.3, 0.5)
        L1_reg = (1e-6, 1e-5, 1e-4, 1e-3)
        L2_reg = (1e-7, 3e-7, 1e-6, 3e-6)


    architecture, trainer = _resolve_tl_trainer(architecture)
    
    all_tr_x = all_tr[0]; all_tr_y = all_tr[1]
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    base_hparams = hyperparam_search_cv(all_tr_x, all_tr_y,
                        train_AI_model=trainer,
                        HParams=nn_HParams,
                        batch_sizes=batch_sizes,
                        learning_rates=learning_rates,
                        L1_reg=L1_reg,
                        L2_reg=L2_reg,
                        dropouts=dropouts,
                        n_epochs =50,
                        )
    timepoint1 = time.perf_counter() # pre-training model tuning 
    base_model, _ = trainer(all_tr_x, all_tr_y, h=base_hparams)
    timepoint2 = time.perf_counter() # pre-training model development
    time_spent =0
    time_spent += (timepoint2 - start)
    count = 0
    pred_prob = []
    running_time = []
    sample_name = []
    te_sample_name = list(ancestral_data_tr.keys())
    for i, key in enumerate(te_sample_name):
        tr_data = ancestral_data_tr[key]
        # model adaptation
        if selected_id != None and not key in selected_id: # check if a testing sample is included in the selected sample ids
            continue  
        tr_x = tr_data[0]; tr_y = tr_data[1]
        timepoint3 = time.perf_counter() # pre-training model development
        finetune_hparams = hyperparam_search_cv(tr_x, tr_y,
                            train_AI_model=trainer,
                            HParams=nn_HParams,
                            model = base_model, 
                            batch_sizes=batch_sizes,
                            learning_rates=learning_rates,
                            L1_reg=L1_reg,
                            L2_reg=L2_reg,
                            dropouts=dropouts,
                            n_epochs =10,
                            )
        timepoint4 = time.perf_counter() # finetuning model turning 
        tuned_model, _ = trainer(tr_x, tr_y, model=base_model,h=finetune_hparams)
        
        timepoint5 = time.perf_counter() # finetuning model development 

        # make predictions and save 
        tuned_model.eval() 
        with torch.no_grad():
            logits = tuned_model(torch.tensor(x_te[i].reshape((1, -1)), dtype=torch.float32).to(device)).squeeze(1).cpu().numpy()
        pred_prob.append(_probs_from_logits_binary(logits))
        timepoint6 = time.perf_counter() # inference time 
        running_time.append((start, timepoint1, timepoint2, timepoint3, timepoint4, timepoint5, timepoint6))
        time_spent+=timepoint6-timepoint3
        print(f'finished {count +1} predictions |  tr size: {len(tr_x)} | tuning + pretraining + tuning + adaptation + prediction + time: {time_spent:.3f}s')
        # increase counter 
        count+=1 
        sample_name.append(key)
    prediction_dic['pred'].append(np.array(pred_prob).reshape((-1, 2)))
    prediction_dic['time'].append(np.array(running_time))
    prediction_dic['sample_name'].append(sample_name)


def develop_ICL_AI(ancestral_data_tr, test_data, prediction_dic, selected_id = None):

    # model specificiation  
    x_te = test_data[0]; 

    # variables to be used 
    count = 0
    pred_prob = []
    running_time = []
    model = TabPFNClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_name = []
    te_sample_name = list(ancestral_data_tr.keys())
    for i, key in enumerate(te_sample_name):
        tr_data = ancestral_data_tr[key]
        # model adaptation
        if selected_id != None and not key in selected_id: # check if a testing sample is included in the selected sample ids
            continue  
        tr_x = tr_data[0]; tr_y = tr_data[1]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.fit(tr_x, tr_y)
        # model = develop_ICL_AI_models(X_multimodal_train=[tr_x], 
        #                                             y_multimodal_train=[tr_y],)[0]
        
        end1 = time.perf_counter()
        # print(f"Elapsed time for {key}: {end - start:.3f} seconds")
        # make predictions and save 
        pred_prob.append(model.predict_proba(x_te[i].reshape((1, -1))))

        end2 = time.perf_counter()
        # save running time for the pretraining for an ancestral point
        running_time.append(((end1 - start), (end2 - end1)))
        # print(f'finished the {count +1}th prediction  |  tr size: {len(tr_x)} ||  development time: {end1 - start:.3f}s | inference time: {end2 - end1:.3f}s')
        # increase counter 
        count+=1 
        sample_name.append(key)
    if len(sample_name) == 0:
        prediction_dic['pred'].append(np.zeros((0, 2)))
        prediction_dic['sample_name'].append(sample_name)
        prediction_dic['time'].append(-1) # negative time to indicate no prediction is made for this run
    else:
        prediction_dic['pred'].append(np.array(pred_prob).reshape((-1, 2)))
        prediction_dic['time'].append(np.array(running_time))
        prediction_dic['sample_name'].append(sample_name)


def develop_Enhanced_ICL_AI(all_tr, ancestral_data_tr, test_data, prediction_dic, context_search_mode='one_pass', selected_id = None, distance_metric='euclidean'):
    # Repro
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # model specificiation  
    x_te = test_data[0]; 

    # variables to be used 
    count = 0
    pred_prob = []
    running_time = []
    model = TabPFNClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_name = []
    te_sample_name = list(ancestral_data_tr.keys())
    for i, key in enumerate(te_sample_name):
        tr_data = ancestral_data_tr[key]
        # model adaptation
        if selected_id != None and not key in selected_id: # check if a testing sample is included in the selected sample ids
            continue  
        tr_x = tr_data[0]; tr_y = tr_data[1]
        all_tr_x = all_tr[count][0]; all_tr_y = all_tr[count][1]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        # seach other points from whole training data to enhance context
        if context_search_mode == 'one_pass':
            similar_ind = search_context_cls(all_tr_x, all_tr_y, tr_x, tr_y, distance_metric=distance_metric)
        elif context_search_mode == 'batch':
            similar_ind = batched_search_context_cls(all_tr_x, all_tr_y, tr_x, tr_y, distance_metric=distance_metric)
        print(f'entire population size: {len(all_tr_x)} | training size: {len(tr_x)} | extended size: {len(similar_ind)}')
        
        # obtain extended training set
        ext_X = np.vstack([ all_tr_x[similar_ind], tr_x])
        ext_Y = np.hstack([all_tr_y[similar_ind], tr_y])

        # make prediction with tabpfn using extended context 
        model.fit(ext_X, ext_Y)
        
        
        end1 = time.perf_counter()
  
        # make predictions and save 
        pred_prob.append(model.predict_proba(x_te[i].reshape((1, -1))))
        end2 = time.perf_counter()
        # save running time for the pretraining for an ancestral point
        running_time.append((end1 - start, end2 - end1))
        print(f'finished the {count +1}th prediction  |  tr size: {len(ext_X)} |  development time: {end1 - start:.3f} | inference time: {end2 - end1:.3f}')
        # increase counter 
        count+=1 
        sample_name.append(key)

    if len(sample_name) == 0:
        prediction_dic['pred'].append(np.zeros((0, 2)))
        prediction_dic['sample_name'].append(sample_name)
        prediction_dic['time'].append(-1) # negative time to indicate no prediction is made for this run
    else:
        prediction_dic['pred'].append(np.array(pred_prob).reshape((-1, 2)))
        prediction_dic['time'].append(np.array(running_time))
        prediction_dic['sample_name'].append(sample_name)


# ICL-AI that uses the features with merged ancestry and omics info.
def develop_merge_ancestry_omics_ICL_AI(ancestry_dim, train_sample_id, test_sample_id, train_data, test_data, prediction_dic):

    # data preparation
    train_X = []; train_y = [] 
    test_X = [] 
    for key in train_sample_id:
        train_X.append(train_data[key][0])
        train_y.append(train_data[key][1])

    for key in test_sample_id:
        test_X.append(test_data[key][0])

    feat_dim = train_X[0].shape[-1]
    train_X = np.array(train_X).reshape((-1, feat_dim))
    test_X = np.array(test_X).reshape((-1, feat_dim))
    train_y = np.array(train_y)

    # normalize genetic features
    scaler = StandardScaler()
    train_genetic_X = scaler.fit_transform(train_X[:, -ancestry_dim:])
    test_genetic_X = scaler.transform(test_X[:, -ancestry_dim:])
    train_X[:, -ancestry_dim:] = train_genetic_X
    test_X[:, -ancestry_dim:] = test_genetic_X
    # prediction
    start = time.perf_counter()
    model = TabPFNClassifier()

    if len(test_X) >0 :
        model.fit(train_X, train_y)

        pred_prob =  model.predict_proba(test_X)
        end = time.perf_counter()
        prediction_dic['pred'].append(pred_prob.reshape((-1, 2)))
        prediction_dic['sample_name'].append(test_sample_id)
        prediction_dic['time'].append(end - start)
    else:
        prediction_dic['pred'].append(np.zeros((0, 2)))
        prediction_dic['sample_name'].append(test_sample_id)
        prediction_dic['time'].append(-1) # negative time to indicate no prediction is made for this run

#===================================================================
# Approaches for the raw sample split, where the training and testing 
# samples are provided in the form of two dictionaries (instead of a 
# dictionary for training samples and a list for testing samples as 
# in the radius_knn_hybrid split). The keys of the dictionaries are 
# sample ids and the values are lists of [predictive features, 
# ancestry features, label (only for training samples)].
#===================================================================

# ICL-AI that uses the features with merged ancestry and omics info.
# this is a version for the raw sample split 
def develop_merge_ancestry_omics_ICL_AI_raw_sample_split(train_data, test_data, prediction_dic, weight_use = False, selected_id = None, mode = 'mix'):

    # data preparation
    train_X = []; train_y = [] 
    test_X = []; 
    predictive_feat_len = None 
    ancestry_feat_len = None 

    if selected_id is not None:
        selected_id = set(selected_id)

    for key, val in train_data.items():
        # select samples from majority or minority class to construct training sets 
        if mode == 'min':
            if selected_id is not None and not key in selected_id:
                continue 
        elif selected_id is not None and mode == 'maj':
            if key in selected_id:
                continue 

        predictive_feat = val[0]; predictive_feat_len = len(predictive_feat)
        ancestry_feat = val[1]; ancestry_feat_len = len(ancestry_feat)
        label = val[2]

        train_X.append(np.hstack([predictive_feat, ancestry_feat]))
        train_y.append(label)

    for key, val in test_data.items():
        predictive_feat = val[0]
        ancestry_feat = val[1]
        label = val[2]
        test_X.append(np.hstack([predictive_feat, ancestry_feat]))

    test_sample_id = list(test_data.keys())
    feat_dim = train_X[0].shape[-1]
    train_X = np.array(train_X).reshape((-1, feat_dim))
    test_X = np.array(test_X).reshape((-1, feat_dim))
    train_y = np.array(train_y)

    
    # normalize genetic features
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
  
    if weight_use:
        train_X[:, predictive_feat_len] = train_X[:, predictive_feat_len]  / np.sqrt(predictive_feat_len)
        train_X[:, predictive_feat_len:] = train_X[:, predictive_feat_len:]  / np.sqrt(ancestry_feat_len)
        
        test_X[:, predictive_feat_len] = test_X[:, predictive_feat_len]  / np.sqrt(predictive_feat_len)
        test_X[:, predictive_feat_len:] = test_X[:, predictive_feat_len:]  / np.sqrt(ancestry_feat_len)
    # prediction
    start = time.perf_counter()
    model = TabPFNClassifier()

    if len(test_X) >0 :
        model.fit(train_X, train_y)

        pred_prob =  model.predict_proba(test_X)
        end = time.perf_counter()
        prediction_dic['pred'].append(pred_prob.reshape((-1, 2)))
        prediction_dic['sample_name'].append(test_sample_id)
        prediction_dic['time'].append(end - start)
    else:
        prediction_dic['pred'].append(np.zeros((0, 2)))
        prediction_dic['sample_name'].append(test_sample_id)
        prediction_dic['time'].append(-1) # negative time to indicate no prediction is made for this run


def develop_sum_ancestry_omics_ICL_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
    tabpfn_n_estimators=8,
    tabpfn_softmax_temperature=0.9,
    tabpfn_balance_probabilities=False,
):
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
                "Summed ancestry-omics ICL requires omics and genetics to share the same dimensionality, "
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
                "Summed ancestry-omics ICL requires omics and genetics to share the same dimensionality, "
                f"got {predictive_feat.shape[-1]} and {ancestry_feat.shape[-1]}."
            )
        test_omics.append(predictive_feat)
        test_genetics.append(ancestry_feat)

    if not train_omics:
        raise ValueError("No training samples available for summed ancestry-plus-omics ICL.")

    feat_dim = train_omics[0].shape[-1]
    train_omics = np.asarray(train_omics, dtype=np.float32).reshape((-1, feat_dim))
    train_genetics = np.asarray(train_genetics, dtype=np.float32).reshape((-1, feat_dim))
    train_y = np.asarray(train_y)

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

    train_X = (train_omics + train_genetics).astype(np.float32)
    test_X = (test_omics + test_genetics).astype(np.float32)

    start = time.perf_counter()
    model = TabPFNClassifier(
        n_estimators=tabpfn_n_estimators,
        softmax_temperature=tabpfn_softmax_temperature,
        balance_probabilities=tabpfn_balance_probabilities,
    )

    if len(test_X) > 0:
        model.fit(train_X, train_y)
        pred_prob = model.predict_proba(test_X)
        end = time.perf_counter()
        classes = model.classes_

        if len(classes) == 1:
            if classes[0] == 0:
                pred_prob = np.hstack(
                    (
                        np.ones(len(pred_prob)).reshape((-1, 1)),
                        np.zeros(len(pred_prob)).reshape((-1, 1)),
                    )
                )
            else:
                pred_prob = np.hstack(
                    (
                        np.zeros(len(pred_prob)).reshape((-1, 1)),
                        np.ones(len(pred_prob)).reshape((-1, 1)),
                    )
                )
        prediction_dic["pred"].append(pred_prob.reshape((-1, 2)))
        prediction_dic["sample_name"].append(test_sample_id)
        prediction_dic["time"].append(end - start)
    else:
        prediction_dic["pred"].append(np.zeros((0, 2)))
        prediction_dic["sample_name"].append(test_sample_id)
        prediction_dic["time"].append(-1)

def develop_DS_ICL_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
    selected_id=None,
    mode='mix',
    tabpfn_n_estimators=8,
    tabpfn_softmax_temperature=0.9,
    tabpfn_balance_probabilities=False,
):
    train_X, train_y, test_X, test_sample_id = _prepare_discrete_predictive_arrays(
        train_data,
        test_data,
        selected_id=selected_id,
        mode=mode,
    )
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
  
    # prediction
    start = time.perf_counter()
    model = TabPFNClassifier(
        n_estimators=tabpfn_n_estimators,
        softmax_temperature=tabpfn_softmax_temperature,
        balance_probabilities=tabpfn_balance_probabilities,
    )

    if len(test_X) >0 :
        model.fit(train_X, train_y)

        pred_prob =  model.predict_proba(test_X)
        end = time.perf_counter()
        classes = model.classes_

        if len(classes) == 1:
            # only one class seen in training
            if classes[0] == 0:
                pred_prob = np.hstack((np.ones(len(pred_prob)).reshape((-1, 1)), np.zeros(len(pred_prob)).reshape((-1, 1))))

            else:
                pred_prob = np.hstack((np.zeros(len(pred_prob)).reshape((-1, 1)), np.ones(len(pred_prob)).reshape((-1, 1))))
        prediction_dic['pred'].append(pred_prob.reshape((-1, 2)))
        prediction_dic['sample_name'].append(test_sample_id)
        prediction_dic['time'].append(end - start)
    else:
        prediction_dic['pred'].append(np.zeros((0, 2)))
        prediction_dic['sample_name'].append(test_sample_id)
        prediction_dic['time'].append(-1) # negative time to indicate no prediction is made for this run


def _elasticnet_feature_mask(
    train_X,
    train_y,
    min_features=1,
    coef_threshold=1e-8,
):
    train_X = np.asarray(train_X, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=int).reshape(-1)
    if train_X.ndim != 2 or train_X.shape[1] == 0:
        raise ValueError("ElasticNet feature selection requires a 2D feature matrix with at least one feature.")
    if len(np.unique(train_y)) < 2:
        return np.ones(train_X.shape[1], dtype=bool)

    scaler = StandardScaler()
    scaled_train_X = scaler.fit_transform(train_X)
    selector = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=0,
    )
    selector.fit(scaled_train_X, train_y)
    coef_abs = np.abs(np.asarray(selector.coef_).reshape(-1))
    selected = coef_abs > coef_threshold
    if selected.any():
        return selected

    min_features = max(1, min(int(min_features), train_X.shape[1]))
    top_idx = np.argsort(coef_abs)[::-1][:min_features]
    selected = np.zeros(train_X.shape[1], dtype=bool)
    selected[top_idx] = True
    return selected


def develop_DS_ElasticNetSelected_ICL_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
    selected_id=None,
    mode='mix',
    tabpfn_n_estimators=8,
    tabpfn_softmax_temperature=0.9,
    tabpfn_balance_probabilities=False,
    elasticnet_min_features=1,
):
    start = time.perf_counter()
    train_X, train_y, test_X, test_sample_id = _prepare_discrete_predictive_arrays(
        train_data,
        test_data,
        selected_id=selected_id,
        mode=mode,
    )
    selected_features = _elasticnet_feature_mask(
        train_X,
        train_y,
        min_features=elasticnet_min_features,
    )
    train_X = train_X[:, selected_features]
    test_X = test_X[:, selected_features] if test_X.size else np.zeros((0, int(selected_features.sum())), dtype=np.float32)

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    model = TabPFNClassifier(
        n_estimators=tabpfn_n_estimators,
        softmax_temperature=tabpfn_softmax_temperature,
        balance_probabilities=tabpfn_balance_probabilities,
    )

    if len(test_X) > 0:
        model.fit(train_X, train_y)
        pred_prob = model.predict_proba(test_X)
        classes = model.classes_

        if len(classes) == 1:
            if classes[0] == 0:
                pred_prob = np.hstack((np.ones(len(pred_prob)).reshape((-1, 1)), np.zeros(len(pred_prob)).reshape((-1, 1))))
            else:
                pred_prob = np.hstack((np.zeros(len(pred_prob)).reshape((-1, 1)), np.ones(len(pred_prob)).reshape((-1, 1))))

        end = time.perf_counter()
        prediction_dic['pred'].append(pred_prob.reshape((-1, 2)))
        prediction_dic['sample_name'].append(test_sample_id)
        prediction_dic['time'].append(end - start)
        prediction_dic.setdefault('selected_feature_count', []).append(int(selected_features.sum()))
        prediction_dic.setdefault('original_feature_count', []).append(int(len(selected_features)))
    else:
        prediction_dic['pred'].append(np.zeros((0, 2)))
        prediction_dic['sample_name'].append(test_sample_id)
        prediction_dic['time'].append(-1)
        prediction_dic.setdefault('selected_feature_count', []).append(int(selected_features.sum()))
        prediction_dic.setdefault('original_feature_count', []).append(int(len(selected_features)))


def develop_DS_ElasticNetSelected_Enhanced_ICL_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
    selected_id,
    mode='mix',
    context_search_mode='one_pass_by_class',
    distance_metric='euclidean',
    tabpfn_n_estimators=8,
    tabpfn_softmax_temperature=0.9,
    tabpfn_balance_probabilities=False,
    elasticnet_min_features=1,
):
    start = time.perf_counter()
    (
        train_src_X,
        train_src_y,
        train_src_context_X,
        train_tgt_X,
        train_tgt_y,
        train_tgt_context_X,
        test_X,
        test_sample_id,
        train_src_ids,
        train_tgt_ids,
    ) = _prepare_discrete_enhanced_icl_arrays(train_data, test_data, selected_id)
    normalized_search_map = _normalize_augmented_feature_blocks(
        [
            ("train_src", train_src_context_X, train_src_y),
            ("train_tgt", train_tgt_context_X, train_tgt_y),
        ]
    )
    train_src_context_X = normalized_search_map["train_src"]
    train_tgt_context_X = normalized_search_map["train_tgt"]

    if context_search_mode == 'batch_by_class':
        batch_size = max(1, len(train_tgt_X))
        similar_ind = batched_search_context_cls_by_class(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            distance_metric=distance_metric,
            batch_size=batch_size,
        )
    elif context_search_mode == 'one_pass_by_class':
        similar_ind = search_context_cls_by_class(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            distance_metric=distance_metric,
        )
    else:
        raise ValueError(
            "ElasticNet-selected enhanced ICL currently supports "
            "'one_pass_by_class' and 'batch_by_class'."
        )
    # record selected source sample ids for diagnostics
    try:
        selected_src_ids = [train_src_ids[int(i)] for i in np.asarray(similar_ind, dtype=int).tolist()]
    except Exception:
        selected_src_ids = []
    prediction_dic.setdefault("selected_source_sample_ids", []).append(selected_src_ids)

    ext_X = np.vstack((train_src_X[similar_ind], train_tgt_X))
    ext_Y = np.concatenate((train_src_y[similar_ind], train_tgt_y)).ravel()
    if mode == 'mix':
        feature_selection_X = ext_X
        feature_selection_y = ext_Y
    elif mode == 'min':
        feature_selection_X = train_tgt_X
        feature_selection_y = train_tgt_y
    else:
        raise ValueError(f"Unsupported ElasticNet-selected enhanced ICL mode: {mode!r}")

    selected_features = _elasticnet_feature_mask(
        feature_selection_X,
        feature_selection_y,
        min_features=elasticnet_min_features,
    )
    ext_X = ext_X[:, selected_features]
    test_X = test_X[:, selected_features] if test_X.size else np.zeros((0, int(selected_features.sum())), dtype=np.float32)

    scaler = StandardScaler()
    ext_X = scaler.fit_transform(ext_X)
    test_X = scaler.transform(test_X)

    model = TabPFNClassifier(
        n_estimators=tabpfn_n_estimators,
        softmax_temperature=tabpfn_softmax_temperature,
        balance_probabilities=tabpfn_balance_probabilities,
    )

    if len(test_X) > 0:
        model.fit(ext_X, ext_Y)
        pred_prob = model.predict_proba(test_X)
        classes = model.classes_

        if len(classes) == 1:
            if classes[0] == 0:
                pred_prob = np.hstack((np.ones(len(pred_prob)).reshape((-1, 1)), np.zeros(len(pred_prob)).reshape((-1, 1))))
            else:
                pred_prob = np.hstack((np.zeros(len(pred_prob)).reshape((-1, 1)), np.ones(len(pred_prob)).reshape((-1, 1))))

        end = time.perf_counter()
        prediction_dic['pred'].append(pred_prob.reshape((-1, 2)))
        prediction_dic['sample_name'].append(test_sample_id)
        prediction_dic['time'].append(end - start)
    else:
        prediction_dic['pred'].append(np.zeros((0, 2)))
        prediction_dic['sample_name'].append(test_sample_id)
        prediction_dic['time'].append(-1)

    prediction_dic.setdefault('selected_feature_count', []).append(int(selected_features.sum()))
    prediction_dic.setdefault('original_feature_count', []).append(int(len(selected_features)))
    prediction_dic.setdefault('enhanced_context_count', []).append(int(len(similar_ind)))


def develop_DS_TL_AI_for_raw_sample_split(train_data, test_data, prediction_dic, selected_id, hyper_params_option = 1, architecture='nn'):
    # Repro
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_src_X = []
    train_src_y = []

    train_tgt_X = []
    train_tgt_y =[]

    # construct data for TL-AI
    test_X =[]; test_y =[]

    for key, val in train_data.items():
        predictive_feat = val[0]
        label = val[2]

        if key in selected_id:
            train_tgt_X.append(predictive_feat)
            train_tgt_y.append(label)
        else:
            train_src_X.append(predictive_feat)
            train_src_y.append(label)
    for key, val in test_data.items():
        predictive_feat = val[0]
        label = val[2]      
        test_X.append(predictive_feat)
        test_y.append(label)
    test_sample_id = list(test_data.keys())
    feat_dim = train_src_X[0].shape[-1]

    train_src_X = np.array(train_src_X).reshape((-1, feat_dim))
    train_tgt_X = np.array(train_tgt_X).reshape((-1, feat_dim))
    train_src_y = np.array(train_src_y)
    train_tgt_y = np.array(train_tgt_y)

    test_X = np.array(test_X).reshape((-1, feat_dim))
    scaler = StandardScaler()
    combined_train_X = np.vstack((train_src_X, train_tgt_X))
    scaler.fit(combined_train_X)
    train_src_X = scaler.transform(train_src_X)
    train_tgt_X = scaler.transform(train_tgt_X)
    test_X = scaler.transform(test_X)

    # parameter to search 
    if hyper_params_option == 1:
        batch_sizes=(8, )
        learning_rates=(1e-3, )
        dropouts=(0.0, )
        L1_reg = (1e-6, )
        L2_reg = (1e-7, )
    elif hyper_params_option == 2:
        # batch_sizes=(8, 16)
        # learning_rates=(1e-3, 3e-3)
        # dropouts=(0.0, 0.1)
        # L1_reg = (1e-6, 1e-5)
        # L2_reg = (1e-7, 3e-7)
        batch_sizes=(16,)
        learning_rates=(1e-3,)
        dropouts=(0.1,)
        L1_reg = (1e-6, 1e-5)
        L2_reg = (1e-7, 3e-7)
    elif hyper_params_option == 3:
        batch_sizes=(8, 16, 32)
        learning_rates=(1e-3, 3e-3, 1e-2)
        dropouts=(0.0, 0.1, 0.3)
        L1_reg = (1e-6, 1e-5, 1e-4)
        L2_reg = (1e-7, 3e-7, 1e-6)
    elif hyper_params_option == 4:
        batch_sizes=(8, 16, 32, 64)
        learning_rates=(1e-3, 3e-3, 1e-2, 3e-2)
        dropouts=(0.0, 0.1, 0.3, 0.5)
        L1_reg = (1e-6, 1e-5, 1e-4, 1e-3)
        L2_reg = (1e-7, 3e-7, 1e-6, 3e-6)

    architecture, trainer = _resolve_tl_trainer(architecture)

    base_hparams = hyperparam_search_cv(train_src_X, train_src_y,
                        train_AI_model=trainer,
                        HParams=nn_HParams,
                        batch_sizes=batch_sizes,
                        learning_rates=learning_rates,
                        L1_reg=L1_reg,
                        L2_reg=L2_reg,
                        dropouts=dropouts,
                        n_epochs =50,
                        )
    
    base_model, _ = trainer(train_src_X, train_src_y, h=base_hparams)


    finetune_hparams = hyperparam_search_cv(train_tgt_X, train_tgt_y,
                        train_AI_model=trainer,
                        HParams=nn_HParams,
                        model = base_model, 
                        batch_sizes=batch_sizes,
                        learning_rates=learning_rates,
                        L1_reg=L1_reg,
                        L2_reg=L2_reg,
                        dropouts=dropouts,
                        n_epochs =10,
                        )

    tuned_model, _ = trainer(train_tgt_X, train_tgt_y, model=base_model,h=finetune_hparams)
    device = next(tuned_model.parameters()).device
 

    # make predictions and save 
    tuned_model.eval() 
    with torch.no_grad():
        logits = tuned_model(torch.tensor(test_X, dtype=torch.float32).to(device)).squeeze(1).cpu().numpy()
    pred_prob = _probs_from_logits_binary(logits)
    prediction_dic['pred'].append(np.array(pred_prob).reshape((-1, 2)))
    prediction_dic['sample_name'].append(test_sample_id)


def develop_DS_Enhanced_ICL_AI_raw_sample_split(
    train_data,
    test_data,
    prediction_dic,
    selected_id,
    context_search_mode='radius',
    distance_metric='euclidean',
    tabpfn_n_estimators=8,
    tabpfn_softmax_temperature=0.9,
    tabpfn_balance_probabilities=False,
    radius_scale=1.0,
    knn_k=10,
):
    (
        train_src_X,
        train_src_y,
        train_src_context_X,
        train_tgt_X,
        train_tgt_y,
        train_tgt_context_X,
        test_X,
        test_sample_id,
        train_src_ids,
        train_tgt_ids,
    ) = _prepare_discrete_enhanced_icl_arrays(train_data, test_data, selected_id)
    normalized_search_map = _normalize_augmented_feature_blocks(
        [
            ("train_src", train_src_context_X, train_src_y),
            ("train_tgt", train_tgt_context_X, train_tgt_y),
        ]
    )
    train_src_context_X = normalized_search_map["train_src"]
    train_tgt_context_X = normalized_search_map["train_tgt"]

    knn_graph_k = int(knn_k)

    if context_search_mode == 'batch':
        batch_size = max(1, len(train_tgt_X))
        similar_ind = batched_search_context_cls(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            distance_metric=distance_metric,
            batch_size=batch_size,
        )
    elif context_search_mode == 'batch_by_class':
        batch_size = max(1, len(train_tgt_X))
        similar_ind = batched_search_context_cls_by_class(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            distance_metric=distance_metric,
            batch_size=batch_size,
        )
    elif context_search_mode == 'one_pass_by_class':
        similar_ind = search_context_cls_by_class(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            distance_metric=distance_metric,
        )
    elif context_search_mode == 'one_pass':
        similar_ind = search_context_cls(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            distance_metric=distance_metric,
        )
    elif context_search_mode == 'batch_knn':
        similar_ind = knn_graph_search_context_cls(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            k=knn_graph_k,
            distance_metric=distance_metric,
        )
    elif context_search_mode == 'knn_by_class':
        selected = []
        for cls in np.unique(np.asarray(train_tgt_y).reshape(-1)):
            src_mask = np.asarray(train_src_y).reshape(-1) == cls
            tgt_mask = np.asarray(train_tgt_y).reshape(-1) == cls
            if not np.any(src_mask) or not np.any(tgt_mask):
                continue
            local = knn_graph_search_context_cls(
                train_src_context_X[src_mask],
                train_src_y[src_mask],
                train_tgt_context_X[tgt_mask],
                train_tgt_y[tgt_mask],
                k=knn_graph_k,
                distance_metric=distance_metric,
            )
            selected.extend(np.where(src_mask)[0][np.asarray(local, dtype=int)].tolist())
        similar_ind = np.array(sorted(set(selected)), dtype=int)
    elif context_search_mode == 'radius':
        radius = float(radius_scale) * np.sqrt(max(1, train_src_context_X.shape[-1]))
        similar_ind = radius_search_context_cls(
            train_src_context_X,
            train_src_y,
            train_tgt_context_X,
            train_tgt_y,
            radius=radius,
            distance_metric=distance_metric,
        )
    elif context_search_mode == 'radius_by_class':
        radius = float(radius_scale) * np.sqrt(max(1, train_src_context_X.shape[-1]))
        selected = []
        for cls in np.unique(np.asarray(train_tgt_y).reshape(-1)):
            src_mask = np.asarray(train_src_y).reshape(-1) == cls
            tgt_mask = np.asarray(train_tgt_y).reshape(-1) == cls
            if not np.any(src_mask) or not np.any(tgt_mask):
                continue
            local = radius_search_context_cls(
                train_src_context_X[src_mask],
                train_src_y[src_mask],
                train_tgt_context_X[tgt_mask],
                train_tgt_y[tgt_mask],
                radius=radius,
                distance_metric=distance_metric,
            )
            selected.extend(np.where(src_mask)[0][np.asarray(local, dtype=int)].tolist())
        similar_ind = np.array(sorted(set(selected)), dtype=int)
    else:
        raise ValueError(f"Unsupported context_search_mode: {context_search_mode}")
    # record selected source sample ids for diagnostics
    try:
        selected_src_ids = [train_src_ids[int(i)] for i in np.asarray(similar_ind, dtype=int).tolist()]
    except Exception:
        selected_src_ids = []
    prediction_dic.setdefault("selected_source_sample_ids", []).append(selected_src_ids)

    print(f'majority group size: {len(train_src_X)} | minority group size: {len(train_tgt_y)} | extended size: {len(similar_ind)}')
    if context_search_mode == 'batch_by_class':
        print('context search uses the normalized context feature space with separate batch retrieval in class 0 and class 1, then unions the retrieved source indices')
    elif context_search_mode == 'one_pass_by_class':
        print('context search uses the normalized context feature space with separate one-pass retrieval in class 0 and class 1, then unions the retrieved source indices')
    elif context_search_mode == 'radius_by_class':
        print('context search uses the normalized context feature space with separate radius retrieval in class 0 and class 1, then unions the retrieved source indices')
    elif context_search_mode == 'knn_by_class':
        print('context search uses the normalized context feature space with separate KNN graph retrieval in class 0 and class 1, then unions the retrieved source indices')
    else:
        print('context search uses the normalized context feature space without labels')
    if context_search_mode == 'batch':
        print(f'batch context search uses majority batches of size {batch_size}')
    if context_search_mode == 'batch_by_class':
        print(f'batch_by_class context search uses majority batches of size {batch_size} within each label stratum')
    if context_search_mode == 'one_pass_by_class':
        print('one_pass_by_class context search performs MST retrieval separately within each label stratum')
    if context_search_mode == 'batch_knn':
        print(f'batch_knn context search uses k={knn_graph_k}')
    if context_search_mode == 'knn_by_class':
        print(f'knn_by_class context search uses k={knn_graph_k} within each label stratum')
    if context_search_mode == 'radius':
        print(f'radius used for {train_src_context_X.shape[-1]} normalized context-search features: {radius} (scale={radius_scale})')
    if context_search_mode == 'radius_by_class':
        print(f'radius_by_class uses radius={radius} for {train_src_context_X.shape[-1]} normalized context-search features (scale={radius_scale})')
    
    # obtain extended training set

    ext_X = np.vstack((train_src_X[similar_ind], train_tgt_X))
    ext_Y = np.concatenate((train_src_y[similar_ind], train_tgt_y)).ravel()
    scaler = StandardScaler()
    ext_X = scaler.fit_transform(ext_X)
    test_X = scaler.transform(test_X)

    model = TabPFNClassifier(
        n_estimators=tabpfn_n_estimators,
        softmax_temperature=tabpfn_softmax_temperature,
        balance_probabilities=tabpfn_balance_probabilities,
    )

    # make prediction with tabpfn using extended context 
    model.fit(ext_X, ext_Y)

    pred_prob =  model.predict_proba(test_X)
    classes = model.classes_

    if len(classes) == 1:
        # only one class seen in training
        if classes[0] == 0:
            pred_prob = np.hstack((np.ones(len(pred_prob)).reshape((-1, 1)), np.zeros(len(pred_prob)).reshape((-1, 1))))

        else:
            pred_prob = np.hstack((np.zeros(len(pred_prob)).reshape((-1, 1)), np.ones(len(pred_prob)).reshape((-1, 1))))
    # print(pred_prob)
    prediction_dic['pred'].append(pred_prob.reshape((-1, 2)))
    prediction_dic['sample_name'].append(test_sample_id)

def develop_Cont_Enhanced_ICL_AI_raw_sample_split(
    dev_data: Dict[str, Dict[str, Any]],
    test_data: Dict[str, Dict[str, Any]],
    prediction_dic: Dict[str, list],
    selected_id=None,
    context_search_mode: str = "radius",
    distance_metric: str = "euclidean",
):
    """
    Validate the radius on the validation set, then use the selected radius
    on the test set for context-enhanced in-context learning.

    Expected structure of dev_data:
        {
            "train_original": dict,      # sample_id -> tuple(..., predictive_feat, ..., label) or similar
            "train_core": dict,
            "validation_query": dict,
            "validation_context": dict,  # sample_id -> {"neighbor_ids": [...]}
        }

    Expected structure of test_data:
        {
            "test_query": dict,
            "test_context": dict,        # sample_id -> {"neighbor_ids": [...]}
        }

    Assumed sample tuple layout:
        val[0] = predictive omic feature
        val[2] = label

    Notes:
    - selected_id is kept in the signature for compatibility, but is not used here.
    - prediction_dic will be updated in place with:
        prediction_dic["pred"]
        prediction_dic["sample_name"]
    """
    max_batch_context_train_size = 9000

    def _get_feature_and_label(sample_val):
        """
        Assumes:
            sample_val[0] = feature
            sample_val[2] = label
        """
        label = int(np.asarray(sample_val[2]).reshape(-1)[0])
        return sample_val[0], label

    def _pack_samples(sample_dict):
        sample_ids = list(sample_dict.keys())
        if not sample_ids:
            return sample_ids, np.empty((0, 0)), np.empty((0,), dtype=int)

        first_feat, _ = _get_feature_and_label(sample_dict[sample_ids[0]])
        feat_dim = np.asarray(first_feat).shape[-1]
        X = np.empty((len(sample_ids), feat_dim), dtype=np.asarray(first_feat).dtype)
        y = np.empty(len(sample_ids), dtype=int)

        for i, sid in enumerate(sample_ids):
            feat, label = _get_feature_and_label(sample_dict[sid])
            X[i] = feat
            y[i] = label

        return sample_ids, X, y

    def _build_context_arrays(query_ids, context_dict, source_ids, source_X, source_y):
        source_index = {sid: idx for idx, sid in enumerate(source_ids)}
        context_X = []
        context_y = []

        for sid in query_ids:
            neighbor_ids = context_dict[sid]["neighbor_ids"]
            if neighbor_ids:
                neighbor_ind = np.fromiter(
                    (source_index[nid] for nid in neighbor_ids),
                    dtype=int,
                    count=len(neighbor_ids),
                )
                context_X.append(source_X[neighbor_ind])
                context_y.append(source_y[neighbor_ind])
            else:
                context_X.append(np.empty((0, source_X.shape[1]), dtype=source_X.dtype))
                context_y.append(np.empty((0,), dtype=source_y.dtype))

        return context_X, context_y

    # =========================================================
    # Build train_original
    # =========================================================
    train_original_ids, train_original_X, train_original_y = _pack_samples(
        dev_data["train_original"]
    )

    # =========================================================
    # Build train_core
    # =========================================================
    train_core_ids, train_core_X, train_core_y = _pack_samples(dev_data["train_core"])

    # =========================================================
    # Build validation query
    # =========================================================
    val_sample_ids, val_X, val_y = _pack_samples(dev_data["validation_query"])

    # =========================================================
    # Build test query
    # =========================================================
    test_sample_ids, te_X, te_y = _pack_samples(test_data["test_query"])

    # =========================================================
    # Build validation context from train_core
    # =========================================================
    val_context_X, val_context_y = _build_context_arrays(
        val_sample_ids,
        dev_data["validation_context"],
        train_core_ids,
        train_core_X,
        train_core_y,
    )

    # =========================================================
    # Build test context from train_original
    # =========================================================
    te_context_X, te_context_y = _build_context_arrays(
        test_sample_ids,
        test_data["test_context"],
        train_original_ids,
        train_original_X,
        train_original_y,
    )
    feat_dim = train_original_X.shape[-1]

    normalized_search_map = _normalize_augmented_feature_blocks(
        [
            ("train_original", train_original_X, train_original_y),
            ("train_core", train_core_X, train_core_y),
            ("val_query", val_X, val_y),
            ("test_query", te_X, te_y),
        ]
        + [
            ("val_context_{}".format(idx), ctx_X, ctx_y)
            for idx, (ctx_X, ctx_y) in enumerate(zip(val_context_X, val_context_y))
        ]
        + [
            ("test_context_{}".format(idx), ctx_X, ctx_y)
            for idx, (ctx_X, ctx_y) in enumerate(zip(te_context_X, te_context_y))
        ]
    )

    train_original_context_X = normalized_search_map["train_original"]
    train_core_context_X = normalized_search_map["train_core"]
    val_X_context = normalized_search_map["val_query"]
    val_context_X_search = [
        normalized_search_map[f"val_context_{idx}"] for idx in range(len(val_context_X))
    ]
    te_context_X_search = [
        normalized_search_map[f"test_context_{idx}"] for idx in range(len(te_context_X))
    ]

    # =========================================================
    # Retrieval for test context expansion
    # =========================================================
    if context_search_mode == "radius":
        candidate_radii = scaled_radius_candidates(train_original_context_X.shape[-1])

  
        similar_ind_list, optimal_radius = cont_radius_search_context_cls_with_param_search(
            train_original_context_X,
            train_original_y,
            te_context_X_search,
            te_context_y,
            train_core_context_X,
            train_core_y,
            val_context_X_search,
            val_context_y,
            val_X_context,
            val_y,
            radii=candidate_radii,
            distance_metric=distance_metric,
        )
        print(f"The radius selected by validation is: {optimal_radius}")
        print(
            "Radius candidates for continuum-enhanced ICL "
            f"({train_original_context_X.shape[-1]} normalized context-search features): {candidate_radii}"
        )
    elif context_search_mode == "radius_by_class":
        candidate_radii = scaled_radius_candidates(train_original_context_X.shape[-1])
        similar_ind_list, optimal_radius = cont_radius_search_context_cls_with_param_search_by_class(
            train_original_context_X,
            train_original_y,
            te_context_X_search,
            te_context_y,
            train_core_context_X,
            train_core_y,
            val_context_X_search,
            val_context_y,
            val_X_context,
            val_y,
            radii=candidate_radii,
            distance_metric=distance_metric,
        )
        print(f"The class-stratified radius selected by validation is: {optimal_radius}")
        print(
            "Radius candidates for continuum-enhanced ICL class-stratified search "
            f"({train_original_context_X.shape[-1]} normalized context-search features): {candidate_radii}"
        )
    elif context_search_mode == "one_pass":
        similar_ind_list = None
        print("Using one_pass context retrieval for continuum-enhanced ICL.")
    elif context_search_mode == "one_pass_by_class":
        similar_ind_list = None
        print("Using one_pass_by_class context retrieval for continuum-enhanced ICL.")
    elif context_search_mode == "batch":
        similar_ind_list = None
        print("Using batch context retrieval for continuum-enhanced ICL.")
    elif context_search_mode == "batch_by_class":
        similar_ind_list = None
        print("Using batch_by_class context retrieval for continuum-enhanced ICL.")
    else:
        raise ValueError(
            f"Unsupported context_search_mode='{context_search_mode}'. "
            f"Supported modes are 'radius', 'radius_by_class', 'one_pass', 'one_pass_by_class', 'batch', and 'batch_by_class'."
        )
    if context_search_mode == "batch_by_class":
        print(
            "Context search uses the normalized context feature space with separate"
            " batch retrieval in class 0 and class 1, then unions the retrieved"
            " source indices."
        )
    elif context_search_mode == "one_pass_by_class":
        print(
            "Context search uses the normalized context feature space with separate"
            " one-pass retrieval in class 0 and class 1, then unions the retrieved"
            " source indices."
        )
    elif context_search_mode == "radius_by_class":
        print(
            "Context search uses the normalized context feature space with separate"
            " radius retrieval in class 0 and class 1, then unions the retrieved"
            " source indices."
        )
    else:
        print("Context search uses the normalized context feature space without labels.")

    # =========================================================
    # Prediction on test set
    # =========================================================
    pred_prob = []
    
    for i, (current_context_X, current_context_y) in enumerate(zip(te_context_X_search, te_context_y)):
        if context_search_mode in {"radius", "radius_by_class"}:
            similar_ind = np.asarray(similar_ind_list[i], dtype=int)
        elif context_search_mode == "one_pass":
            similar_ind = search_context_cls(
                train_original_context_X,
                train_original_y,
                current_context_X,
                current_context_y,
                distance_metric=distance_metric,
            )
            similar_ind = np.asarray(similar_ind, dtype=int)
        elif context_search_mode == "one_pass_by_class":
            similar_ind = search_context_cls_by_class(
                train_original_context_X,
                train_original_y,
                current_context_X,
                current_context_y,
                distance_metric=distance_metric,
            )
            similar_ind = np.asarray(similar_ind, dtype=int)
        else:  # batch
            batch_size = max(1, len(current_context_X))
            if context_search_mode == "batch":
                similar_ind = batched_search_context_cls(
                    train_original_context_X,
                    train_original_y,
                    current_context_X,
                    current_context_y,
                    distance_metric=distance_metric,
                    batch_size=batch_size,
                )
            else:
                similar_ind = batched_search_context_cls_by_class(
                    train_original_context_X,
                    train_original_y,
                    current_context_X,
                    current_context_y,
                    distance_metric=distance_metric,
                    batch_size=batch_size,
                )
            similar_ind = np.asarray(similar_ind, dtype=int)

            max_retrieved = max(0, max_batch_context_train_size - len(current_context_X))
            if len(similar_ind) > max_retrieved:
                similar_ind = similar_ind[:max_retrieved]

        retrieved_X = train_original_X[similar_ind]
        retrieved_y = train_original_y[similar_ind]

        current_context_X = te_context_X[i]
        current_context_y = te_context_y[i]

        # If there are no neighborhood samples, just use retrieved samples.

        ext_X = np.vstack([retrieved_X.reshape((-1, feat_dim)), current_context_X.reshape((-1, feat_dim))])
        ext_y = np.vstack([retrieved_y.reshape((-1,1)), current_context_y.reshape((-1,1))]).ravel()

        model = TabPFNClassifier()
        model.fit(ext_X, ext_y)

        # TabPFN expects 2D input for prediction
        test_x_i = te_X[i].reshape(1, -1)
        prob_i = model.predict_proba(test_x_i)  # shape: (1, n_classes)
        
        classes = model.classes_

        if len(classes) == 2:
            # normal binary case
            p = prob_i[0]   # shape (2,)
        elif len(classes) == 1:
            # only one class seen in training
            if classes[0] == 0:
                p = np.array([1.0, 0.0])
            else:
                p = np.array([0.0, 1.0])
        pred_prob.append(p)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # pdb.set_trace()
    pred_prob = np.array(pred_prob)

    if "pred" not in prediction_dic:
        prediction_dic["pred"] = []
    if "sample_name" not in prediction_dic:
        prediction_dic["sample_name"] = []

    prediction_dic["pred"].append(pred_prob)
    prediction_dic["sample_name"].append(test_sample_ids)

    return prediction_dic
