from __future__ import annotations

import gc
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

try:
    from .Xgboostevaluate import calculate_security_metrics, print_metrics, save_metrics
except ImportError:
    
    from Xgboostevaluate import calculate_security_metrics, print_metrics, save_metrics
FEATURE_COLUMNS = [
    "auth_attempts_5m_per_src_user",
    "successful_auths_5m_per_src_user",
    "unique_src_computers_1h_per_src_user",
    "unique_dst_computers_5m_per_src_user",
    "unique_dst_computers_1h_per_src_user",
    "unique_dst_users_1h_per_src_computer",
    "unique_src_users_1h_per_dst_computer",
    "prior_auth_count_1h_src_user_dst_computer",
    "is_first_seen_src_user_src_computer",
    "is_first_seen_src_user_dst_computer",
    "is_first_seen_src_computer_dst_computer",
    "seconds_since_last_auth_by_src_user",
    "seconds_since_last_src_user_dst_computer",
    "current_event_is_success",
    "hour_of_day",
    "is_network_logon",
]
FEATURE_SCHEMA_VERSION = "auth_anomaly_v2"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
TRAIN_PATH = SPLIT_DIR / "train.csv"
SMOTE_TRAIN_PATH = SPLIT_DIR / "train_smote.csv"
VALID_PATH = SPLIT_DIR / "valid.csv"
TEST_PATH = SPLIT_DIR / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_baseline.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "xgboost_baseline_metrics.csv"

FEATURE_DTYPES = {column: "float32" for column in FEATURE_COLUMNS}
READ_DTYPES = {**FEATURE_DTYPES, "label": "int8"}


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature split not found: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [column for column in FEATURE_COLUMNS + ["label"] if column not in header]
    if missing:
        raise ValueError(
            f"{path} does not match {FEATURE_SCHEMA_VERSION}; "
            f"missing columns: {missing}"
        )
    return pd.read_csv(
        path,
        usecols=FEATURE_COLUMNS + ["label"],
        dtype=READ_DTYPES,
    )


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in FEATURE_COLUMNS + ["label"] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    unexpected = [
        column
        for column in df.columns
        if column not in FEATURE_COLUMNS + ["label"]
    ]
    if unexpected:
        raise ValueError(f"Unexpected model columns: {unexpected}")

    # load_features() already reads only FEATURE_COLUMNS + label. Popping the
    # target avoids creating another multi-gigabyte copy of the feature matrix.
    y = df.pop("label").fillna(0).astype("int8", copy=False)
    if df.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Feature columns are not in the canonical model order.")
    X = df
    if X.isna().any().any():
        invalid = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Feature columns contain NaN: {invalid}")
    X = X.astype("float32", copy=False)
    return X, y


def train_baseline_model(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    """Train an unweighted chronological baseline and evaluate at threshold 0.5."""
    train_df = load_features(train_path)
    X_train, y_train = prepare_xy(train_df)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        max_bin=256,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    del train_df, X_train, y_train
    gc.collect()

    test_df = load_features(test_path)
    X_test, y_test = prepare_xy(test_df)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype("int8")
    metrics = calculate_security_metrics(
        y_test,
        y_pred,
        y_prob,
        model_name="xgboost_baseline_unweighted",
    )
    metrics["threshold"] = 0.5
    metrics["train_source"] = train_path.name

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": 0.5,
        "train_source": train_path.name,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    save_metrics(metrics, metrics_path)
    print_metrics(metrics)
    print(f"Saved model bundle to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    return metrics


if __name__ == "__main__":
    train_baseline_model()
