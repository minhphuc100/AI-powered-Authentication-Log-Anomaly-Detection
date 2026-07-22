from __future__ import annotations

import inspect
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

try:
    from .Xgboostbaseline import (
        CHUNK_SIZE,
        CHUNK_SLEEP_SECONDS,
        FEATURE_COLUMNS,
        LOWER_PROCESS_PRIORITY,
        N_JOBS,
        check_feature_file,
        lower_process_priority,
        validate_columns,
    )

    from .Xgboostevaluate import (
        calculate_security_metrics,
        print_metrics,
        save_metrics,
    )

except ImportError:

    from Xgboostbaseline import (
        CHUNK_SIZE,
        CHUNK_SLEEP_SECONDS,
        FEATURE_COLUMNS,
        LOWER_PROCESS_PRIORITY,
        N_JOBS,
        check_feature_file,
        lower_process_priority,
        validate_columns,
    )

    from Xgboostevaluate import (
        calculate_security_metrics,
        print_metrics,
        save_metrics,
    )


# ============================================================
# PROJECT PATHS
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

IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "xgboost_feature_importance.csv"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

DEFAULT_THRESHOLD = 0.35

EARLY_STOPPING_ROUNDS = 25


# ============================================================
# TRAINING DATA PREPARATION
# ============================================================

def load_validation_data(
    path: Path,
):
    """
    Load validation data in chunks.

    Unlike training, the validation set is needed
    by XGBoost for early stopping.

    Because the validation set has ~16 million rows,
    we take a controlled sample of it for validation.

    The full validation data is still kept separate
    from training and test data.
    """

    validation_parts = []

    total_rows = 0

    print(
        "\nLoading validation data "
        "in chunks..."
    )

    for chunk in pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    ):

        validate_columns(
            chunk
        )

        X_chunk = (
            chunk[
                FEATURE_COLUMNS
            ]
            .fillna(0)
        )

        y_chunk = (
            chunk["label"]
            .astype(int)
        )

        validation_parts.append(
            (
                X_chunk,
                y_chunk,
            )
        )

        total_rows += len(
            chunk
        )

        print(
            f"Validation rows loaded: "
            f"{total_rows:,}"
        )

    # --------------------------------------------------------
    # Combine validation chunks
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # This can still consume significant RAM.
    #
    # Since your validation set has 16M rows,
    # we instead sample the chunks below.
    #

    X_parts = []

    y_parts = []

    for X_chunk, y_chunk in validation_parts:

        # Keep all anomaly rows.
        anomaly_mask = (
            y_chunk == 1
        )

        X_anomaly = (
            X_chunk[
                anomaly_mask
            ]
        )

        y_anomaly = (
            y_chunk[
                anomaly_mask
            ]
        )

        # Sample normal rows.
        normal_mask = (
            y_chunk == 0
        )

        X_normal = (
            X_chunk[
                normal_mask
            ]
        )

        y_normal = (
            y_chunk[
                normal_mask
            ]
        )

        # Keep at most 10,000 normal
        # samples per chunk.

        if len(
            X_normal
        ) > 10_000:

            X_normal = (
                X_normal.sample(
                    n=10_000,
                    random_state=42,
                )
            )

            y_normal = (
                y_normal.loc[
                    X_normal.index
                ]
            )

        X_parts.append(
            pd.concat(
                [
                    X_anomaly,
                    X_normal,
                ]
            )
        )

        y_parts.append(
            pd.concat(
                [
                    y_anomaly,
                    y_normal,
                ]
            )
        )

        del X_chunk

        del y_chunk

        time.sleep(
            CHUNK_SLEEP_SECONDS
        )

    X_val = pd.concat(
        X_parts,
        ignore_index=True,
    )

    y_val = pd.concat(
        y_parts,
        ignore_index=True,
    )

    del validation_parts

    del X_parts

    del y_parts

    print(
        "\nValidation sample ready:"
    )

    print(
        f"Rows: "
        f"{len(X_val):,}"
    )

    print(
        f"Anomalies: "
        f"{int(y_val.sum()):,}"
    )

    return (
        X_val,
        y_val,
    )


# ============================================================
# TRAINING CHUNK WITH ANOMALY-AWARE SAMPLING
# ============================================================

def train_chunked_model(
    model: XGBClassifier,
    path: Path,
) -> XGBClassifier:
    """
    Train the improved model in chunks.

    To prevent the model from spending almost all
    training time on millions of normal rows,
    normal samples are limited per chunk.

    All anomaly rows are retained.
    """

    first_chunk = True

    total_rows = 0

    total_anomalies = 0

    chunk_number = 0

    for chunk in pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    ):

        chunk_number += 1

        validate_columns(
            chunk
        )

        X_chunk = (
            chunk[
                FEATURE_COLUMNS
            ]
            .fillna(0)
        )

        y_chunk = (
            chunk["label"]
            .astype(int)
        )

        total_rows += len(
            chunk
        )

        total_anomalies += int(
            y_chunk.sum()
        )

        # ----------------------------------------------------
        # Keep all anomaly samples
        # ----------------------------------------------------

        anomaly_mask = (
            y_chunk == 1
        )

        X_anomaly = (
            X_chunk[
                anomaly_mask
            ]
        )

        y_anomaly = (
            y_chunk[
                anomaly_mask
            ]
        )

        # ----------------------------------------------------
        # Sample normal data
        # ----------------------------------------------------

        normal_mask = (
            y_chunk == 0
        )

        X_normal = (
            X_chunk[
                normal_mask
            ]
        )

        y_normal = (
            y_chunk[
                normal_mask
            ]
        )

        # ----------------------------------------------------
        # Limit normal samples
        # ----------------------------------------------------

        MAX_NORMAL_PER_CHUNK = 20_000

        if len(
            X_normal
        ) > MAX_NORMAL_PER_CHUNK:

            X_normal = (
                X_normal.sample(
                    n=MAX_NORMAL_PER_CHUNK,
                    random_state=42
                    + chunk_number,
                )
            )

            y_normal = (
                y_normal.loc[
                    X_normal.index
                ]
            )

        # ----------------------------------------------------
        # Combine anomaly + normal
        # ----------------------------------------------------

        X_train_chunk = pd.concat(
            [
                X_anomaly,
                X_normal,
            ],
            ignore_index=True,
        )

        y_train_chunk = pd.concat(
            [
                y_anomaly,
                y_normal,
            ],
            ignore_index=True,
        )

        print(
            f"\nTraining chunk "
            f"{chunk_number}"
        )

        print(
            f"Original rows: "
            f"{len(chunk):,}"
        )

        print(
            f"Training rows used: "
            f"{len(X_train_chunk):,}"
        )

        print(
            f"Anomalies: "
            f"{int(y_train_chunk.sum()):,}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        if first_chunk:

            model.fit(
                X_train_chunk,
                y_train_chunk,
                eval_set=[
                    (
                        X_train_chunk,
                        y_train_chunk,
                    )
                ],
                verbose=False,
            )

            first_chunk = False

        else:

            model.fit(
                X_train_chunk,
                y_train_chunk,
                xgb_model=model.get_booster(),
                eval_set=[
                    (
                        X_train_chunk,
                        y_train_chunk,
                    )
                ],
                verbose=False,
            )

        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

        del chunk

        del X_chunk

        del y_chunk

        del X_train_chunk

        del y_train_chunk

        time.sleep(
            CHUNK_SLEEP_SECONDS
        )

    print(
        "\nImproved chunked training complete."
    )

    print(
        f"Original rows processed: "
        f"{total_rows:,}"
    )

    print(
        f"Anomalies encountered: "
        f"{total_anomalies:,}"
    )

    return model


# ============================================================
# EVALUATE TEST DATA IN CHUNKS
# ============================================================

def predict_test_in_chunks(
    model: XGBClassifier,
    path: Path,
    threshold: float,
):
    """
    Predict test data in chunks.
    """

    all_labels = []

    all_predictions = []

    all_probabilities = []

    total_rows = 0

    chunk_number = 0

    for chunk in pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    ):

        chunk_number += 1

        validate_columns(
            chunk
        )

        X_chunk = (
            chunk[
                FEATURE_COLUMNS
            ]
            .fillna(0)
        )

        y_chunk = (
            chunk["label"]
            .astype(int)
        )

        y_prob_chunk = (
            model.predict_proba(
                X_chunk
            )[:, 1]
        )

        y_pred_chunk = (
            y_prob_chunk
            >= threshold
        ).astype(int)

        all_labels.extend(
            y_chunk.tolist()
        )

        all_predictions.extend(
            y_pred_chunk.tolist()
        )

        all_probabilities.extend(
            y_prob_chunk.tolist()
        )

        total_rows += len(
            chunk
        )

        print(
            f"Test chunk "
            f"{chunk_number}: "
            f"{total_rows:,} rows"
        )

        del X_chunk

        del y_chunk

        del chunk

        time.sleep(
            CHUNK_SLEEP_SECONDS
        )

    return (
        all_labels,
        all_predictions,
        all_probabilities,
    )


# ============================================================
# EARLY STOPPING
# ============================================================

def _fit_with_early_stopping(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:

    fit_signature = inspect.signature(
        model.fit
    )

    if (
        "early_stopping_rounds"
        in fit_signature.parameters
    ):

        model.fit(

            X_train,

            y_train,

            eval_set=[
                (
                    X_val,
                    y_val,
                )
            ],

            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            ),

            verbose=False,
        )

    else:

        model.set_params(

            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            )

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
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model: XGBClassifier,
    output_path: Path,
) -> pd.DataFrame:

    importance = pd.DataFrame(

        {
            "feature": FEATURE_COLUMNS,

            "importance":
                model.feature_importances_,
        }

    ).sort_values(

        "importance",

        ascending=False,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance.to_csv(
        output_path,
        index=False,
    )

    return importance


# ============================================================
# TRAIN IMPROVED MODEL
# ============================================================

def train_improved_model(

    train_path: Path = TRAIN_PATH,

    validation_path: Path = VALIDATION_PATH,

    test_path: Path = TEST_PATH,

    model_path: Path = MODEL_PATH,

    metrics_path: Path = METRICS_PATH,

    importance_path: Path = IMPORTANCE_PATH,

    threshold: float = DEFAULT_THRESHOLD,

) -> dict:

    # --------------------------------------------------------
    # Lower process priority
    # --------------------------------------------------------

    if LOWER_PROCESS_PRIORITY:

        lower_process_priority()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    check_feature_file(
        train_path
    )

    check_feature_file(
        validation_path
    )

    check_feature_file(
        test_path
    )

    # --------------------------------------------------------
    # Load validation sample
    # --------------------------------------------------------

    (
        X_val,
        y_val,
    ) = load_validation_data(
        validation_path
    )

    # --------------------------------------------------------
    # Calculate class imbalance
    #
    # This uses a small sample of training data
    # instead of loading the full 55M rows.
    # --------------------------------------------------------

    first_chunk = next(

        pd.read_csv(

            train_path,

            chunksize=CHUNK_SIZE,

            low_memory=True,
        )

    )

    validate_columns(
        first_chunk
    )

    y_first = (
        first_chunk["label"]
        .astype(int)
    )

    n_normal = (
        y_first == 0
    ).sum()

    n_anomaly = (
        y_first == 1
    ).sum()

    scale_pos_weight = (

        n_normal

        / max(
            n_anomaly,
            1,
        )

    )

    del first_chunk

    del y_first

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=300,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        min_child_weight=2,

        reg_lambda=1.0,

        scale_pos_weight=scale_pos_weight,

        objective="binary:logistic",

        eval_metric="aucpr",

        random_state=42,

        n_jobs=N_JOBS,

    )

    print(
        "\nTraining improved XGBoost..."
    )

    print(
        f"CPU threads: "
        f"{N_JOBS}"
    )

    print(
        f"Chunk size: "
        f"{CHUNK_SIZE:,}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_chunked_model(

        model,

        train_path,

    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    (
        y_test,
        y_pred,
        y_prob,
    ) = predict_test_in_chunks(

        model,

        test_path,

        threshold,

    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = calculate_security_metrics(

        y_test,

        y_pred,

        y_prob,

        model_name=(

            f"xgboost_improved_"
            f"threshold_{threshold}"

        ),
    )

    metrics[
        "threshold"
    ] = threshold

    metrics[
        "scale_pos_weight"
    ] = scale_pos_weight

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    save_metrics(
        metrics,
        metrics_path,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = (
        save_feature_importance(
            model,
            importance_path,
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_metrics(
        metrics
    )

    print(
        "\nTop feature importance"
    )

    print(
        importance.head(
            10
        ).to_string(
            index=False
        )
    )

    print(
        f"\nSaved model to:"
        f"\n{model_path}"
    )

    print(
        f"\nSaved metrics to:"
        f"\n{metrics_path}"
    )

    print(
        f"\nSaved feature importance to:"
        f"\n{importance_path}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_improved_model()