"""Shared prediction metric helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, log_loss, roc_auc_score


def binary_classification_metrics(y_true, y_prob) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 1:
        y_prob = np.column_stack((1.0 - y_prob, y_prob))
    y_prob = np.clip(y_prob[:, :2], 1e-15, 1.0 - 1e-15)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    y_pred = y_prob.argmax(axis=1)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob[:, 1])),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "ce": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }

