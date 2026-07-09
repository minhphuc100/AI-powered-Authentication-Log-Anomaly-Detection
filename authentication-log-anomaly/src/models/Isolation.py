from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

try:
    from .Xgboostbaseline import FEATURE_COLUMNS, FEATURES_PATH, load_features, prepare_xy
    from .Xgboostevaluate import calculate_security_metrics, print_metrics, save_metrics
except ImportError:
    from Xgboostbaseline import FEATURE_COLUMNS, FEATURES_PATH, load_features, prepare_xy
    from Xgboostevaluate import calculate_security_metrics, print_metrics, save_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "isolation_forest.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "isolation_forest_metrics.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "results" / "metrics" / "isolation_forest_feature_importance.csv"


def save_feature_importance(model: IsolationForest, output_path: Path) -> pd.DataFrame:
    tree_importances = pd.DataFrame(
        [tree.feature_importances_ for tree in model.estimators_],
        columns=FEATURE_COLUMNS,
    )
    importance = (
        tree_importances.mean()
        .rename_axis("feature")
        .reset_index(name="importance")
        .sort_values("importance", ascending=False)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    return importance


def train_isolation_model(
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    importance_path: Path = IMPORTANCE_PATH,
) -> dict:
    """Train an Isolation Forest model for auth-log anomaly detection."""
    df = load_features(features_path)
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    contamination = max(float(y_train.mean()), 0.01)
    contamination = min(contamination, 0.5)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    raw_pred = model.predict(X_test)
    y_pred = (raw_pred == -1).astype(int)
    y_score = -model.decision_function(X_test)

    metrics = calculate_security_metrics(
        y_test,
        y_pred,
        y_score,
        model_name="isolation_forest",
    )
    metrics["contamination"] = contamination

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
    train_isolation_model()
