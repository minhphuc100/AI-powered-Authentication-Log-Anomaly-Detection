from __future__ import annotations

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
# PATHS
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
# DATA LOADING
# ============================================================

def load_features(path: Path) -> pd.DataFrame:
    """
    Load a processed feature file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

    print(f"Loading: {path}")

    df = pd.read_csv(path)

    print(
        f"Loaded {len(df):,} rows "
        f"and {len(df.columns)} columns"
    )

    return df


# ============================================================
# PREPARE X AND Y
# ============================================================

def prepare_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:

    required_columns = (
        FEATURE_COLUMNS
        + ["label"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        "label"
    ].astype(int)

    # Replace missing numerical values
    # with zero.
    X = X.fillna(0)

    return X, y


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
    # Load pre-split datasets
    # --------------------------------------------------------

    train_df = load_features(
        train_path
    )

    test_df = load_features(
        test_path
    )

    # --------------------------------------------------------
    # Prepare training data
    # --------------------------------------------------------

    X_train, y_train = prepare_xy(
        train_df
    )

    # --------------------------------------------------------
    # Prepare test data
    # --------------------------------------------------------

    X_test, y_test = prepare_xy(
        test_df
    )

    print(
        f"\nTraining rows: {len(X_train):,}"
    )

    print(
        f"Test rows: {len(X_test):,}"
    )

    print(
        f"Training attacks: "
        f"{int(y_train.sum()):,}"
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
        f"\nNormal samples: "
        f"{n_normal:,}"
    )

    print(
        f"Anomaly samples: "
        f"{n_anomaly:,}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    # --------------------------------------------------------
    # Create baseline XGBoost model
    # --------------------------------------------------------

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,

        subsample=0.8,
        colsample_bytree=0.8,

        scale_pos_weight=scale_pos_weight,

        objective="binary:logistic",
        eval_metric="logloss",

        random_state=42,

        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nTraining baseline XGBoost..."
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = calculate_security_metrics(
        y_test,
        y_pred,
        y_prob,
        model_name="xgboost_baseline",
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
    # Print metrics
    # --------------------------------------------------------

    print_metrics(
        metrics
    )

    print(
        f"\nModel saved to:"
        f"\n{model_path}"
    )

    print(
        f"\nMetrics saved to:"
        f"\n{metrics_path}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_baseline_model()