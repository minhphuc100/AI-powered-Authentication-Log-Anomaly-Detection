from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_decision_threshold(
    y_true,
    y_prob,
    bins: int = 10_000,
    min_recall_target: float = 0.8,
) -> dict[str, float]:
    """Select the validation threshold with maximum F1 using bounded memory."""
    if bins < 2:
        raise ValueError("bins must be at least two")

    labels = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if labels.size == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must be a non-empty binary array")
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("Validation data must contain at least one positive label")

    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    indexes = np.minimum((probabilities * bins).astype(np.int64), bins)
    positive_histogram = np.bincount(indexes[labels == 1], minlength=bins + 1)
    negative_histogram = np.bincount(indexes[labels == 0], minlength=bins + 1)
    true_positive = np.cumsum(positive_histogram[::-1])[::-1]
    false_positive = np.cumsum(negative_histogram[::-1])[::-1]
    false_negative = positives - true_positive

    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=(true_positive + false_positive) > 0,
    )
    recall = true_positive / positives
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    valid_recall_indexes = np.where(recall >= min_recall_target)[0]
    
    if len(valid_recall_indexes) > 0:
        # Trong các threshold đạt Recall >= 85%, lấy cái có Precision tốt nhất
        best = valid_recall_indexes[np.argmax(precision[valid_recall_indexes])]
    else:
        # Nếu không đạt nổi min_recall, lấy threshold có Recall cao nhất
        best = int(np.argmax(recall))
    return {
        "threshold": best / bins,
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
        "true_positive": int(true_positive[best]),
        "false_positive": int(false_positive[best]),
        "false_negative": int(false_negative[best]),
    }


def calculate_security_metrics(
    y_true,
    y_pred,
    y_prob=None,
    model_name: str = "model",
) -> dict[str, Any]:
    """Return metrics suitable for highly imbalanced anomaly labels."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    metrics: dict[str, Any] = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["normal", "anomaly"],
            zero_division=0,
        ),
    }

    if y_prob is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"classification_report", "confusion_matrix"}
    }
    pd.DataFrame([flat_metrics]).to_csv(output_path, index=False)


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"\n[{metrics['model']}] Classification report")
    print(metrics["classification_report"])
    print(f"PR-AUC: {metrics['pr_auc']}")
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
