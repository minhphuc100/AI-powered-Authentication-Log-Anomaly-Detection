from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

try:
    from .Xgboostevaluate import (
        calculate_security_metrics,
        print_metrics,
        save_metrics,
    )
except ImportError:
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

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_baseline.pkl"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "xgboost_baseline_metrics.csv"
)


# ============================================================
# RESOURCE CONTROL
# ============================================================

# Number of rows loaded into RAM at once.
#
# Your dataset contains:
#   Train:      ~55.9 million rows
#   Validation: ~16.1 million rows
#   Test:       ~44.4 million rows
#
# Start with 100,000.
#
# If your PC is stable, you can try 250,000 or 500,000.
CHUNK_SIZE = 100_000


# Number of CPU threads XGBoost is allowed to use.
#
# Your machine has 16 GB RAM.
# Start with 2 to reduce CPU/RAM pressure.
#
# You can increase this to 4 if training is stable.
N_JOBS = 2


# Small pause between training chunks.
#
# This does not make training faster.
# It gives the system a little breathing room.
CHUNK_SLEEP_SECONDS = 0.2


# Lower process priority on Windows.
LOWER_PROCESS_PRIORITY = True


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "is_success",
    "prior_user_attempts_5m",
    "prior_user_logins_today",
]


# ============================================================
# LOWER CPU PROCESS PRIORITY
# ============================================================

def lower_process_priority() -> None:
    """
    Lower the priority of the current process.

    Windows:
        BELOW_NORMAL_PRIORITY_CLASS

    Linux/macOS:
        Uses nice value when possible.

    This helps prevent the training process from
    taking all available CPU resources.
    """

    try:

        if os.name == "nt":

            import ctypes

            kernel32 = ctypes.windll.kernel32

            handle = kernel32.GetCurrentProcess()

            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

            result = kernel32.SetPriorityClass(
                handle,
                BELOW_NORMAL_PRIORITY_CLASS,
            )

            if result:

                print(
                    "Process priority set to "
                    "BELOW_NORMAL."
                )

            else:

                print(
                    "Warning: Could not lower "
                    "Windows process priority."
                )

        else:

            try:

                os.nice(5)

                print(
                    "Process priority lowered "
                    "using nice()."
                )

            except PermissionError:

                print(
                    "Warning: Could not change "
                    "process priority."
                )

    except Exception as exc:

        print(
            "Warning: Failed to change "
            f"process priority: {exc}"
        )


# ============================================================
# CHECK DATA FILE
# ============================================================

def check_feature_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Please check your feature-builder "
            "output path."
        )


# ============================================================
# VALIDATE COLUMNS
# ============================================================

def validate_columns(
    df: pd.DataFrame,
) -> None:

    required_columns = (
        FEATURE_COLUMNS
        + ["label"]
    )

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            f"{missing}"
        )


# ============================================================
# TRAIN ONE CHUNK
# ============================================================

def train_chunked_model(
    model: XGBClassifier,
    path: Path,
) -> XGBClassifier:
    """
    Train XGBoost incrementally using CSV chunks.

    Each chunk is loaded into RAM, used for training,
    then released before the next chunk is loaded.
    """

    print(
        f"\nStarting chunked training:"
        f"\n{path}"
    )

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

        # ----------------------------------------------------
        # Validate columns on first chunk
        # ----------------------------------------------------

        if first_chunk:

            validate_columns(
                chunk
            )

        # ----------------------------------------------------
        # Prepare X and y
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        rows = len(
            chunk
        )

        anomalies = int(
            y_chunk.sum()
        )

        total_rows += rows

        total_anomalies += anomalies

        print(
            f"\nChunk {chunk_number}"
        )

        print(
            f"Rows: "
            f"{rows:,}"
        )

        print(
            f"Anomalies: "
            f"{anomalies:,}"
        )

        print(
            f"Total processed: "
            f"{total_rows:,}"
        )

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        if first_chunk:

            model.fit(
                X_chunk,
                y_chunk,
                verbose=False,
            )

            first_chunk = False

        else:

            model.fit(
                X_chunk,
                y_chunk,
                xgb_model=model.get_booster(),
                verbose=False,
            )

        # ----------------------------------------------------
        # Release chunk memory
        # ----------------------------------------------------

        del X_chunk

        del y_chunk

        del chunk

        # ----------------------------------------------------
        # Give system a small break
        # ----------------------------------------------------

        time.sleep(
            CHUNK_SLEEP_SECONDS
        )

    print(
        "\nChunked training complete."
    )

    print(
        f"Total rows processed: "
        f"{total_rows:,}"
    )

    print(
        f"Total anomalies: "
        f"{total_anomalies:,}"
    )

    return model


# ============================================================
# EVALUATE TEST SET IN CHUNKS
# ============================================================

def predict_test_in_chunks(
    model: XGBClassifier,
    path: Path,
):
    """
    Predict the test set chunk by chunk.

    This prevents loading the entire test dataset
    into memory.
    """

    all_predictions = []

    all_probabilities = []

    all_labels = []

    total_rows = 0

    chunk_number = 0

    print(
        f"\nEvaluating test data:"
        f"\n{path}"
    )

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

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_prob_chunk = (
            model.predict_proba(
                X_chunk
            )[:, 1]
        )

        y_pred_chunk = (
            y_prob_chunk >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

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
            f"{total_rows:,} rows processed"
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
# TRAIN BASELINE MODEL
# ============================================================

def train_baseline_model(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:

    # --------------------------------------------------------
    # Lower process priority
    # --------------------------------------------------------

    if LOWER_PROCESS_PRIORITY:

        lower_process_priority()

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    check_feature_file(
        train_path
    )

    check_feature_file(
        test_path
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=100,

        max_depth=3,

        learning_rate=0.1,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=42,

        n_jobs=N_JOBS,

    )

    # --------------------------------------------------------
    # Train in chunks
    # --------------------------------------------------------

    model = train_chunked_model(
        model,
        train_path,
    )

    # --------------------------------------------------------
    # Test in chunks
    # --------------------------------------------------------

    (
        y_test,
        y_pred,
        y_prob,
    ) = predict_test_in_chunks(
        model,
        test_path,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = calculate_security_metrics(

        y_test,

        y_pred,

        y_prob,

        model_name=(
            "xgboost_baseline"
        ),
    )

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
    # Print results
    # --------------------------------------------------------

    print_metrics(
        metrics
    )

    print(
        f"\nSaved model to:"
        f"\n{model_path}"
    )

    print(
        f"\nSaved metrics to:"
        f"\n{metrics_path}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_baseline_model()