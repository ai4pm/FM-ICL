import copy
from itertools import product
from typing import Optional

import numpy as np
from torch import nn
from sklearn.model_selection import KFold

# ------------ hyperparameter search with cross-validation -----------

def five_fold_cv_val_loss(
    all_x: np.ndarray,
    all_y: np.ndarray,
    *,
    train_AI_model,
    hparams,
    model: Optional[nn.Module] = None,
    n_splits: int = 5,
    seed: int = 0,
    shuffle: bool = True,
):
    n_samples = int(len(all_x))
    if n_samples < 2:
        return float("inf")

    effective_splits = min(int(n_splits), n_samples)
    if effective_splits < 2:
        return float("inf")

    kf = KFold(n_splits=effective_splits, shuffle=shuffle, random_state=seed)

    fold_losses = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(all_x), start=1):
        X_tr, y_tr = all_x[tr_idx], all_y[tr_idx]
        X_val, y_val = all_x[val_idx], all_y[val_idx]

        if model is not None:
            model2 = copy.deepcopy(model)
        else:
            model2 = None 
        _, val_loss = train_AI_model(X_train=X_tr, 
                                     y_train=y_tr, 
                                     model = model2,
                                     X_val = X_val, 
                                     y_val = y_val,
                                     h = hparams,)

        fold_losses.append(val_loss)
    fold_losses = np.array(fold_losses)
    mean_val_loss = float(fold_losses.mean())
    return mean_val_loss

def hyperparam_search_cv(
    all_x: np.ndarray,
    all_y: np.ndarray,
    *,
    train_AI_model, 
    HParams,
    model: Optional[nn.Module] = None,
    batch_sizes:tuple = (16, 32, 64),
    learning_rates: tuple = (1e-3, 3e-3, 1e-2),
    dropouts:tuple = (0.0, 0.1, 0.3, 0.5),
    L1_reg: tuple = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
    L2_reg: tuple = (0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5),
    n_epochs: int = 50,
    n_splits: int = 5,
    seed: int = 0,
):

    best_hparams = None
    best_val_loss = None 
    count = 0 
    for bs, lr, do, l1, l2 in product(batch_sizes, learning_rates, dropouts, L1_reg, L2_reg):
        hparams = HParams(
            n_epochs=n_epochs,
            learning_rate=lr,
            dropout=do,
            batch_size=bs,
            L1_reg=l1,
            L2_reg=l2,
        )
        mean_val_loss = five_fold_cv_val_loss(
            all_x, all_y,
            train_AI_model=train_AI_model,
            model = model,
            hparams=hparams,
            n_splits=n_splits,
            seed=seed,
        )

        if best_val_loss is None or mean_val_loss < best_val_loss:
            best_hparams = hparams
            best_val_loss = mean_val_loss
        # print('Hyperparam set %d: bs=%d, lr=%.5f, do=%.2f, l1=%.7f, l2=%.7f --> mean val loss: %.5f' %
        #       (count+1, bs, lr, do, l1, l2, mean_val_loss))
        count += 1
    return best_hparams
