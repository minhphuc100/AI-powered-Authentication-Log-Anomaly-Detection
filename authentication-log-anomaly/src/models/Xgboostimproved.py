from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

try:
    from .Xgboostbaseline import (
        FEATURE_COLUMNS,
        load_features,
        prepare_xy,
    )

    from .Xgboostevaluate import (
        calculate_security_metrics,
        print_metrics,
        save_metrics,
    )

except ImportError:

    from Xgboostbaseline import (
        FEATURE_COLUMNS,
        load_features,
        prepare_xy,
    )

    from Xgboostevaluate import (
        calculate_security_metrics,
        print_metrics,
        save_metrics,
    )


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_improved.pkl"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "xgboost_improved_metrics.csv"
)


# ============================================================
# EARLY STOPPING TRAINING
# ============================================================

def _fit_with_early_stopping(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):

    print(
        "\nTraining improved XGBoost "
        "with validation set..."
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (
                X_val,
                y_val,
            )
        ],

        verbose=False,
    )

    return model


# ============================================================
# THRESHOLD TUNING
# ============================================================

def find_best_threshold(
    y_true,
    y_prob,
):

    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = np.arange(
        0.01,
        1.00,
        0.01,
    )

    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        tp = (
            (y_true == 1)
            & (y_pred == 1)
        ).sum()

        fp = (
            (y_true == 0)
            & (y_pred == 1)
        ).sum()

        fn = (
            (y_true == 1)
            & (y_pred == 0)
        ).sum()

        precision = (
            tp
            / max(tp + fp, 1)
        )

        recall = (
            tp
            / max(tp + fn, 1)
        )

        if (
            precision + recall
        ) == 0:

            f1 = 0.0

        else:

            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    print(
        f"\nBest validation threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best validation F1: "
        f"{best_f1:.4f}"
    )

    return best_threshold


# ============================================================
# TRAIN IMPROVED MODEL
# ============================================================

def train_improved_model():

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_df = load_features(
        TRAIN_PATH
    )

    validation_df = load_features(
        VALIDATION_PATH
    )

    test_df = load_features(
        TEST_PATH
    )

    # --------------------------------------------------------
    # Prepare train
    # --------------------------------------------------------

    X_train, y_train = prepare_xy(
        train_df
    )

    # --------------------------------------------------------
    # Prepare validation
    # --------------------------------------------------------

    X_val, y_val = prepare_xy(
        validation_df
    )

    # --------------------------------------------------------
    # Prepare test
    # --------------------------------------------------------

    X_test, y_test = prepare_xy(
        test_df
    )

    print(
        "\nDataset sizes:"
    )

    print(
        f"Train: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation: "
        f"{len(X_val):,}"
    )

    print(
        f"Test: "
        f"{len(X_test):,}"
    )

    print(
        "\nAttack counts:"
    )

    print(
        f"Train attacks: "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Validation attacks: "
        f"{int(y_val.sum()):,}"
    )

    print(
        f"Test attacks: "
        f"{int(y_test.sum()):,}"
    )

    # --------------------------------------------------------
    # Calculate class imbalance
    # --------------------------------------------------------

    n_normal = (
        y_train == 0
    ).sum()

    n_anomaly = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        n_normal
        / max(n_anomaly, 1)
    )

    print(
        f"\nscale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    # --------------------------------------------------------
    # Create improved model
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=1000,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        scale_pos_weight=scale_pos_weight,

        reg_lambda=1.0,

        reg_alpha=0.0,

        objective="binary:logistic",

        eval_metric="aucpr",

        random_state=42,

        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Train using validation set
    # --------------------------------------------------------

    model = _fit_with_early_stopping(

        model,

        X_train,
        y_train,

        X_val,
        y_val,
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    y_val_prob = model.predict_proba(
        X_val
    )[:, 1]

    # --------------------------------------------------------
    # Find threshold using validation set
    # --------------------------------------------------------

    threshold = find_best_threshold(

        y_val,

        y_val_prob,
    )

    # --------------------------------------------------------
    # Final test predictions
    # --------------------------------------------------------

    y_test_prob = model.predict_proba(
        X_test
    )[:, 1]

    y_test_pred = (
        y_test_prob >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Evaluate test set
    # --------------------------------------------------------

    metrics = calculate_security_metrics(

        y_test,

        y_test_pred,

        y_test_prob,

        model_name="xgboost_improved",
    )

    # Save selected threshold
    metrics[
        "threshold"
    ] = threshold

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    save_metrics(
        metrics,
        METRICS_PATH,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_metrics(
        metrics
    )

    print(
        f"\nSelected threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_PATH}"
    )

    print(
        f"\nMetrics saved to:"
        f"\n{METRICS_PATH}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_improved_model()