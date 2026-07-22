from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_security_metrics(
    y_true,
    y_pred,
    y_prob=None,
    model_name: str = "model",
) -> dict[str, Any]:
    """Return the metrics used to compare anomaly detection models."""
    metrics: dict[str, Any] = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
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
    else:
        metrics["roc_auc"] = None

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
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")

