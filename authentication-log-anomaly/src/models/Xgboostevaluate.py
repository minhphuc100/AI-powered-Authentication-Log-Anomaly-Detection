from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CALCULATE SECURITY METRICS
# ============================================================

def calculate_security_metrics(
    y_true,
    y_pred,
    y_prob,
    model_name="model",
):

    # --------------------------------------------------------
    # Basic classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )

    except ValueError:

        roc_auc = float(
            "nan"
        )

    # --------------------------------------------------------
    # PR-AUC
    #
    # More useful for highly imbalanced
    # anomaly detection problems.
    # --------------------------------------------------------

    try:

        pr_auc = average_precision_score(
            y_true,
            y_prob,
        )

    except ValueError:

        pr_auc = float(
            "nan"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    # --------------------------------------------------------
    # False Positive Rate
    # --------------------------------------------------------

    false_positive_rate = (
        fp
        / max(
            fp + tn,
            1,
        )
    )

    # --------------------------------------------------------
    # False Negative Rate
    # --------------------------------------------------------

    false_negative_rate = (
        fn
        / max(
            fn + tp,
            1,
        )
    )

    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {

        "model": model_name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc,

        "pr_auc": pr_auc,

        "true_negative": int(
            tn
        ),

        "false_positive": int(
            fp
        ),

        "false_negative": int(
            fn
        ),

        "true_positive": int(
            tp
        ),

        "false_positive_rate":
            false_positive_rate,

        "false_negative_rate":
            false_negative_rate,
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    metrics: dict,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MODEL EVALUATION RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Model: "
        f"{metrics.get('model', 'N/A')}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1 Score: "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC: "
        f"{metrics['pr_auc']:.6f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"True Negative: "
        f"{metrics['true_negative']:,}"
    )

    print(
        f"False Positive: "
        f"{metrics['false_positive']:,}"
    )

    print(
        f"False Negative: "
        f"{metrics['false_negative']:,}"
    )

    print(
        f"True Positive: "
        f"{metrics['true_positive']:,}"
    )

    print(
        "\nError Rates:"
    )

    print(
        f"False Positive Rate: "
        f"{metrics['false_positive_rate']:.6f}"
    )

    print(
        f"False Negative Rate: "
        f"{metrics['false_negative_rate']:.6f}"
    )

    print(
        "=" * 60
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics: dict,
    output_path: Path,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Convert dictionary to
    # one-row DataFrame.

    df = pd.DataFrame(
        [metrics]
    )

    # If the file already exists,
    # append the new result.

    if output_path.exists():

        df.to_csv(
            output_path,
            mode="a",
            header=False,
            index=False,
        )

    else:

        df.to_csv(
            output_path,
            index=False,
        )

    print(
        f"\nMetrics saved to:"
        f"\n{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "XGBoost evaluation module."
    )

    print(
        "Run Xgboostbaseline.py or "
        "Xgboostimproved.py to train "
        "and evaluate a model."
    )