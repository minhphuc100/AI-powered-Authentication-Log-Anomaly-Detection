"""
Improved XGBoost Model

Trains a stronger XGBoost classifier with validation-based early stopping,
class-imbalance handling, a lower anomaly threshold, and feature-importance
reporting.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

try:
    from .Xgboostbaseline import (
        FEATURE_COLUMNS,
        TEST_PATH,
        TRAIN_PATH,
        VALID_PATH,
        load_features,
        prepare_xy,
        resolve_feature_path,
    )
    from .Xgboostevaluate import calculate_security_metrics, print_metrics, save_metrics
except ImportError:
    from Xgboostbaseline import (
        FEATURE_COLUMNS,
        TEST_PATH,
        TRAIN_PATH,
        VALID_PATH,
        load_features,
        prepare_xy,
        resolve_feature_path,
    )
    from Xgboostevaluate import calculate_security_metrics, print_metrics, save_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_improved.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "xgboost_improved_metrics.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "results" / "metrics" / "xgboost_feature_importance.csv"


def _fit_with_early_stopping(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:
    """Fit with early stopping across older and newer XGBoost versions."""
    fit_signature = inspect.signature(model.fit)
    if "early_stopping_rounds" in fit_signature.parameters:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=25,
            verbose=False,
        )
    else:
        model.set_params(early_stopping_rounds=25)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    return model


def save_feature_importance(model: XGBClassifier, output_path: Path) -> pd.DataFrame:
    """Save the XGBoost feature-importance ranking to CSV."""
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    return importance


def train_improved_model(
    train_path: Path = TRAIN_PATH,
    valid_path: Path = VALID_PATH,
    test_path: Path = TEST_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    importance_path: Path = IMPORTANCE_PATH,
    threshold: float = 0.35,
) -> dict:
    """Train an improved XGBoost model for imbalanced auth-log anomalies."""
    # Resolve paths first so errors show the actual file location being used.
    train_path = resolve_feature_path(train_path)
    valid_path = resolve_feature_path(valid_path)
    test_path = resolve_feature_path(test_path)

    X_train, y_train = prepare_xy(load_features(train_path))
    X_val, y_val = prepare_xy(load_features(valid_path))
    X_test, y_test = prepare_xy(load_features(test_path))

    # Weight anomaly examples higher when the dataset is mostly normal logins.
    n_normal = (y_train == 0).sum()
    n_anomaly = (y_train == 1).sum()
    scale_pos_weight = n_normal / max(n_anomaly, 1)

    # Stronger configuration than the baseline: more trees, regularization,
    # subsampling, and class weighting for imbalanced labels.
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
        eval_metric="auc",
        random_state=42,
    )

    _fit_with_early_stopping(model, X_train, y_train, X_val, y_val)

    # Lower threshold favors recall, which is useful for suspicious-login alerts.
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = calculate_security_metrics(
        y_test,
        y_pred,
        y_prob,
        model_name=f"xgboost_improved_threshold_{threshold}",
    )
    metrics["threshold"] = threshold
    metrics["scale_pos_weight"] = scale_pos_weight

    # Persist the model, scalar metrics, and feature-importance report.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    save_metrics(metrics, metrics_path)
    importance = save_feature_importance(model, importance_path)

    print_metrics(metrics)
    print("\nTop feature importance")
    print(importance.head(10).to_string(index=False))
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved feature importance to: {importance_path}")

    return metrics


if __name__ == "__main__":
    train_improved_model()

