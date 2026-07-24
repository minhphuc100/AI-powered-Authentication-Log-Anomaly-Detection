from __future__ import annotations

import argparse
import gc
import inspect
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

try:
    from .Xgboostbaseline import (
        FEATURE_COLUMNS,
        SMOTE_TRAIN_PATH,
        TEST_PATH,
        TRAIN_PATH,
        VALID_PATH,
        load_features,
        prepare_xy,
    )
    from .Xgboostevaluate import (
        calculate_security_metrics,
        print_metrics,
        save_metrics,
        select_decision_threshold,
    )
except ImportError:
    from Xgboostbaseline import (
        FEATURE_COLUMNS,
        SMOTE_TRAIN_PATH,
        TEST_PATH,
        TRAIN_PATH,
        VALID_PATH,
        load_features,
        prepare_xy,
    )
    from Xgboostevaluate import (
        calculate_security_metrics,
        print_metrics,
        save_metrics,
        select_decision_threshold,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def _fit_with_early_stopping(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> XGBClassifier:
    fit_signature = inspect.signature(model.fit)
    if "early_stopping_rounds" in fit_signature.parameters:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=40,
            verbose=False,
        )
    else:
        model.set_params(early_stopping_rounds=40)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


def _resolve_paths(
    train_source: str,
    model_path: Path | None,
    metrics_path: Path | None,
    importance_path: Path | None,
) -> tuple[Path, Path, Path, Path]:
    if train_source not in {"original", "smote"}:
        raise ValueError("train_source must be 'original' or 'smote'")
    train_path = TRAIN_PATH if train_source == "original" else SMOTE_TRAIN_PATH
    suffix = "weighted" if train_source == "original" else "smote"
    return (
        train_path,
        model_path or MODEL_DIR / f"xgboost_{suffix}.pkl",
        metrics_path or METRICS_DIR / f"xgboost_{suffix}_metrics.csv",
        importance_path or METRICS_DIR / f"xgboost_{suffix}_feature_importance.csv",
    )


def save_feature_importance(model: XGBClassifier, output_path: Path) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    return importance


def train_improved_model(
    train_source: str = "original",
    valid_path: Path = VALID_PATH,
    test_path: Path = TEST_PATH,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    importance_path: Path | None = None,
    threshold_bins: int = 10_000,
    min_recall_target: float = 0.8,
) -> dict:
    """Train weighted XGBoost, select threshold on valid, evaluate test once."""
    train_path, model_path, metrics_path, importance_path = _resolve_paths(
        train_source,
        model_path,
        metrics_path,
        importance_path,
    )
    train_df = load_features(train_path)
    valid_df = load_features(valid_path)
    X_train, y_train = prepare_xy(train_df)
    X_valid, y_valid = prepare_xy(valid_df)

    normal_count = int((y_train == 0).sum())
    anomaly_count = int((y_train == 1).sum())
    if anomaly_count == 0:
        raise ValueError(f"No anomaly labels found in {train_path}")
    raw_scale_pos_weight = normal_count / anomaly_count
    scale_pos_weight = (
        float(np.sqrt(raw_scale_pos_weight))
        if train_source == "original"
        else 1.0
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=1.0,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        max_bin=256,
        n_jobs=-1,
        random_state=42,
    )
    _fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid)

    valid_probability = model.predict_proba(X_valid)[:, 1]
    threshold_summary = select_decision_threshold(
        y_valid,
        valid_probability,
        bins=threshold_bins,
        min_recall_target=min_recall_target,
    )
    threshold = threshold_summary["threshold"]
    del train_df, valid_df, X_train, y_train, X_valid, y_valid, valid_probability
    gc.collect()

    test_df = load_features(test_path)
    X_test, y_test = prepare_xy(test_df)
    test_probability = model.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= threshold).astype("int8")
    model_name = f"xgboost_{'weighted' if train_source == 'original' else 'smote'}"
    metrics = calculate_security_metrics(
        y_test,
        test_prediction,
        test_probability,
        model_name=model_name,
    )
    metrics.update(
        {
            "threshold": threshold,
            "validation_precision": threshold_summary["precision"],
            "validation_recall": threshold_summary["recall"],
            "validation_f1": threshold_summary["f1"],
            "validation_false_positive": threshold_summary["false_positive"],
            "min_recall_target": min_recall_target,
            "recall_target_met": threshold_summary["recall_target_met"],
            "threshold_strategy": threshold_summary["threshold_strategy"],
            "raw_scale_pos_weight": raw_scale_pos_weight,
            "scale_pos_weight": scale_pos_weight,
            "train_source": train_path.name,
            "normal_train_rows": normal_count,
            "anomaly_train_rows": anomaly_count,
            "best_iteration": getattr(model, "best_iteration", None),
        }
    )

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "train_source": train_path.name,
        "raw_scale_pos_weight": raw_scale_pos_weight,
        "scale_pos_weight": scale_pos_weight,
        "min_recall_target": min_recall_target,
        "validation_recall": threshold_summary["recall"],
        "validation_precision": threshold_summary["precision"],
        "threshold_strategy": threshold_summary["threshold_strategy"],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    save_metrics(metrics, metrics_path)
    importance = save_feature_importance(model, importance_path)

    print(f"Validation-selected threshold: {threshold:.4f}")
    print_metrics(metrics)
    print("\nTop feature importance")
    print(importance.head(10).to_string(index=False))
    print(f"Saved model bundle to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved feature importance to: {importance_path}")
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train chronological anomaly XGBoost.")
    parser.add_argument(
        "--train-source",
        choices=("original", "smote"),
        default="original",
    )
    parser.add_argument("--valid", type=Path, default=VALID_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--threshold-bins", type=int, default=10_000)
    parser.add_argument("--min-recall-target", type=float, default=0.8)
    return parser.parse_args()


def run() -> None:
    args = _parse_args()
    train_improved_model(
        train_source=args.train_source,
        valid_path=args.valid,
        test_path=args.test,
        threshold_bins=args.threshold_bins,
        min_recall_target=args.min_recall_target,
    )


if __name__ == "__main__":
    run()
