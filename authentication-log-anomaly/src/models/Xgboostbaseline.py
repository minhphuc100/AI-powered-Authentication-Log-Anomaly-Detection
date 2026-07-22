from __future__ import annotations

from cProfile import label
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    from .Xgboostevaluate import calculate_security_metrics, print_metrics, save_metrics
except ImportError:
    from Xgboostevaluate import calculate_security_metrics, print_metrics, save_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2] # Adjust as needed based on your project structure
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_baseline.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "xgboost_baseline_metrics.csv"

FEATURE_COLUMNS = [
    "auth_attempts_5m_per_user",
    "unique_src_computers_1h_per_user",
    "unique_users_1h_per_src_computer",
    "unique_dst_computers_1h_per_user",
    "unique_dst_users_1h_per_src_computer",
    "hour_of_day",
    "is_network_logon"
]


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}. Run parser.py and feature_builder.py first."
        )
    return pd.read_csv(path)


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [col for col in FEATURE_COLUMNS + ["label"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["label"].astype(int)
    return X, y


def train_baseline_model(
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    """Train a simple supervised XGBoost baseline for normal/anomaly labels."""
    df = load_features(features_path)
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = calculate_security_metrics(
        y_test,
        y_pred,
        y_prob,
        model_name="xgboost_baseline",
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    save_metrics(metrics, metrics_path)
    print_metrics(metrics)
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")

    return metrics


if __name__ == "__main__":
    train_baseline_model()
