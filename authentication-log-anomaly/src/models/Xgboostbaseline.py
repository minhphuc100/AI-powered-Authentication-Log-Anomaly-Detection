"""
XGBoost Baseline Model

Trains the first supervised XGBoost classifier for authentication-log anomaly
detection. The loader accepts both the canonical feature_builder.py output and
the older train/valid/test split schema used by some notebooks/scripts.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

try:
    from .Xgboostevaluate import calculate_security_metrics, print_metrics, save_metrics
except ImportError:
    from Xgboostevaluate import calculate_security_metrics, print_metrics, save_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VALID_PATH = PROCESSED_DIR / "valid.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_baseline.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "xgboost_baseline_metrics.csv"

# Canonical feature names shared with feature_builder.py and the deployment app.
FEATURE_COLUMNS = [
    "auth_attempts_5m_per_user",
    "unique_src_computers_1h_per_user",
    "unique_users_1h_per_src_computer",
    "unique_dst_computers_1h_per_user",
    "unique_dst_users_1h_per_src_computer",
    "hour_of_day",
    "is_network_logon",
]

# Older split files use different names. These aliases let the model train on
# either CSV layout without throwing "usecols do not match columns" errors.
FEATURE_ALIASES = {
    "auth_attempts_5m_per_user": "failed_logins_5m_user",
    "unique_src_computers_1h_per_user": "unique_src_ip_1h",
    "unique_users_1h_per_src_computer": "unique_users_1h_per_ip",
}


def resolve_feature_path(path: Path) -> Path:
    """Find a split file in the current or legacy processed-data locations."""
    if path.exists():
        return path

    legacy_path = SPLIT_DIR / path.name
    if legacy_path.exists():
        return legacy_path

    processed_path = PROCESSED_DIR / path.name
    if processed_path.exists():
        return processed_path

    raise FileNotFoundError(
        f"Feature split not found: {path}. Run parser.py and feature_builder.py first."
    )


def load_features(path: Path) -> pd.DataFrame:
    """Load features and normalize any known legacy column names."""
    path = resolve_feature_path(path)
    header = pd.read_csv(path, nrows=0)
    available_columns = set(header.columns)

    usecols = ["label"]
    missing_columns: list[str] = []
    for column in FEATURE_COLUMNS:
        alias = FEATURE_ALIASES.get(column)
        if column in available_columns:
            usecols.append(column)
        elif alias in available_columns:
            usecols.append(alias)
        else:
            missing_columns.append(column)

    if "label" not in available_columns:
        raise ValueError(
            f"Label column not found in {path}. Expected a binary 'label' column."
        )

    df = pd.read_csv(path, usecols=usecols)
    for canonical, alias in FEATURE_ALIASES.items():
        if canonical not in df.columns and alias in df.columns:
            df = df.rename(columns={alias: canonical})

    for column in missing_columns:
        print(f"[xgboost] Warning: {path.name} is missing '{column}', filling with 0.")
        df[column] = 0

    return df[FEATURE_COLUMNS + ["label"]]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a normalized feature table into model inputs and labels."""
    missing = [col for col in FEATURE_COLUMNS + ["label"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["label"].astype(int)
    return X, y


def train_baseline_model(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    """Train on the chronological train split and evaluate once on test."""
    # The split files may be in data/processed or data/processed/splits.
    X_train, y_train = prepare_xy(load_features(train_path))
    X_test, y_test = prepare_xy(load_features(test_path))

    # Conservative baseline: shallow trees and a moderate learning rate.
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Use both class predictions and probabilities for model-quality metrics.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = calculate_security_metrics(
        y_test,
        y_pred,
        y_prob,
        model_name="xgboost_baseline",
    )

    # Save the trained model and metrics so they can be inspected later.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    save_metrics(metrics, metrics_path)
    print_metrics(metrics)
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")

    return metrics


if __name__ == "__main__":
    train_baseline_model()

