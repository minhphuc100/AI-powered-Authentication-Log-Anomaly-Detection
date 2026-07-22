from __future__ import annotations

from pathlib import Path
from typing import Any

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


def calculate_security_metrics(
    y_true,
    y_pred,
    y_prob=None,
    model_name: str = "model",
) -> dict[str, Any]:
    """
    Return metrics used to compare anomaly detection models.

    PR-AUC is included because the dataset is extremely
    imbalanced and the anomaly class is very rare.
    """

    metrics: dict[str, Any] = {

        "model":
            model_name,

        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred,
            ).tolist(),

        "classification_report":
            classification_report(

                y_true,

                y_pred,

                target_names=[
                    "normal",
                    "anomaly",
                ],

                zero_division=0,
            ),
    }

    # ========================================================
    # ROC-AUC
    # ========================================================

    if (
        y_prob is not None
        and len(
            set(y_true)
        ) > 1
    ):

        metrics[
            "roc_auc"
        ] = roc_auc_score(

            y_true,

            y_prob,

        )

    else:

        metrics[
            "roc_auc"
        ] = None

    # ========================================================
    # PR-AUC
    #
    # Especially useful for your dataset because:
    #
    # Train attacks:      523
    # Validation attacks: 126
    # Test attacks:       53
    #
    # The positive class is extremely rare.
    # ========================================================

    if (
        y_prob is not None
        and len(
            set(y_true)
        ) > 1
    ):

        metrics[
            "pr_auc"
        ] = average_precision_score(

            y_true,

            y_prob,

        )

    else:

        metrics[
            "pr_auc"
        ] = None

    return metrics


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics: dict[str, Any],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    flat_metrics = {

        key: value

        for key, value in metrics.items()

        if key
        not in {
            "classification_report",
            "confusion_matrix",
        }

    }

    pd.DataFrame(
        [flat_metrics]
    ).to_csv(

        output_path,

        index=False,

    )


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    metrics: dict[str, Any],
) -> None:

    print(
        f"\n[{metrics['model']}] "
        "Classification report"
    )

    print(
        metrics[
            "classification_report"
        ]
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']}"
    )

    print(
        f"PR-AUC: "
        f"{metrics['pr_auc']}"
    )

    print(
        "Confusion matrix: "
        f"{metrics['confusion_matrix']}"
    )