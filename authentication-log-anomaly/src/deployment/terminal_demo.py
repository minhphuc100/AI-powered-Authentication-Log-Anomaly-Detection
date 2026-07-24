"""Terminal demo for the AIC211 Windows authentication anomaly model.

The model was trained on LANL authentication logs. Windows Event Log support is
therefore an integration demo, not a production-quality Windows detector.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
import streamlit  # noqa: F401
import xgboost as xgb

# Importing app_simple registers Streamlit cache decorators. Silence the harmless
# "No runtime found" message because this program intentionally has no web UI.
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

from src.deployment.app_simple import (  # noqa: E402
    FEATURE_COLUMNS,
    LoadedModel,
    OnlineFeatureEngine,
    _validate_feature_schema,
    predict_probabilities,
    read_windows_events,
)


FEATURE_NAMES_VI = {
    "auth_attempts_5m_per_src_user": "số lần xác thực của user trong 5 phút",
    "successful_auths_5m_per_src_user": "số lần đăng nhập thành công trong 5 phút",
    "unique_src_computers_1h_per_src_user": "số máy nguồn khác nhau của user trong 1 giờ",
    "unique_dst_computers_5m_per_src_user": "số máy đích khác nhau của user trong 5 phút",
    "unique_dst_computers_1h_per_src_user": "số máy đích khác nhau của user trong 1 giờ",
    "unique_dst_users_1h_per_src_computer": "số user đích từ máy nguồn trong 1 giờ",
    "unique_src_users_1h_per_dst_computer": "số user nguồn tới máy đích trong 1 giờ",
    "prior_auth_count_1h_src_user_dst_computer": "số lần user đã truy cập máy đích trong 1 giờ",
    "is_first_seen_src_user_src_computer": "lần đầu thấy cặp user–máy nguồn",
    "is_first_seen_src_user_dst_computer": "lần đầu thấy cặp user–máy đích",
    "is_first_seen_src_computer_dst_computer": "lần đầu thấy cặp máy nguồn–máy đích",
    "seconds_since_last_auth_by_src_user": "thời gian từ lần xác thực trước của user",
    "seconds_since_last_src_user_dst_computer": "thời gian từ lần truy cập trước tới máy đích",
    "current_event_is_success": "sự kiện hiện tại đăng nhập thành công",
    "hour_of_day": "giờ trong ngày",
    "is_network_logon": "đăng nhập qua mạng",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cli_model(model_path: Path) -> LoadedModel:
    loaded = joblib.load(model_path)
    if not isinstance(loaded, dict) or "model" not in loaded:
        raise ValueError(
            "CLI yêu cầu model bundle mới gồm model, feature_columns và threshold."
        )

    feature_columns = list(loaded.get("feature_columns", []))
    _validate_feature_schema(feature_columns, "Model")
    threshold = float(loaded.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold trong model không hợp lệ: {threshold}")

    model = loaded["model"]
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model phải hỗ trợ predict_proba().")

    return LoadedModel(
        model=model,
        feature_columns=feature_columns,
        threshold=threshold,
        train_source=str(loaded.get("train_source", "unknown")),
    )


def model_contributions(
    bundle: LoadedModel,
    feature_frame: pd.DataFrame,
) -> dict[str, float]:
    """Return per-event XGBoost contributions in log-odds space."""
    try:
        booster = bundle.model.get_booster()
        matrix = xgb.DMatrix(
            feature_frame[bundle.feature_columns],
            feature_names=bundle.feature_columns,
        )
        values = booster.predict(matrix, pred_contribs=True)[0]
    except (AttributeError, TypeError, ValueError, xgb.core.XGBoostError):
        return {}
    return {
        feature: float(value)
        for feature, value in zip(bundle.feature_columns, values[:-1])
    }


def top_reasons(
    features: dict[str, int],
    contributions: dict[str, float],
    limit: int = 3,
) -> list[str]:
    if contributions:
        ranked = sorted(
            contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:limit]
        return [
            (
                f"{'tăng' if value >= 0 else 'giảm'} điểm: "
                f"{FEATURE_NAMES_VI.get(name, name)}={features[name]} "
                f"(đóng góp {value:+.3f})"
            )
            for name, value in ranked
        ]

    fallback: list[str] = []
    for name in (
        "is_first_seen_src_user_dst_computer",
        "is_first_seen_src_computer_dst_computer",
        "is_first_seen_src_user_src_computer",
        "is_network_logon",
    ):
        if features.get(name) == 1:
            fallback.append(FEATURE_NAMES_VI[name])
    if not fallback:
        fallback.append("model kết hợp đồng thời toàn bộ 16 feature")
    return fallback[:limit]


def score_event(
    engine: OnlineFeatureEngine,
    bundle: LoadedModel,
    event: dict[str, object],
    threshold: float,
) -> dict[str, object]:
    features = engine.extract(event)
    frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    score = float(predict_probabilities(bundle, frame)[0])
    contributions = model_contributions(bundle, frame)
    return {
        **event,
        **features,
        "score": score,
        "prediction": int(score >= threshold),
        "reasons": top_reasons(features, contributions),
    }


def format_analysis(result: dict[str, object], threshold: float) -> str:
    status = "CẢNH BÁO ANOMALY" if result["prediction"] else "Bình thường"
    lines = [
        "",
        "=" * 78,
        f"[{status}] {result['time_str']} | Event ID {result['event_id']}",
        f"Score: {float(result['score']):.6f} | Threshold: {threshold:.6f}",
        (
            f"User: {result['src_username']} -> {result['username']} | "
            f"Kết quả: {result['result']}"
        ),
        (
            f"Máy: {result['src_host']} -> {result['destination_host']} | "
            f"Logon type: {result['logon_type']}"
        ),
        "Phân tích các yếu tố có ảnh hưởng lớn nhất:",
    ]
    lines.extend(f"  - {reason}" for reason in result["reasons"])
    return "\n".join(lines)


def append_jsonl(path: Path, results: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for result in results:
            serializable = {
                key: value
                for key, value in result.items()
                if key != "reasons" or isinstance(value, list)
            }
            stream.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def read_events_or_explain(max_events: int) -> list[dict[str, object]]:
    try:
        return read_windows_events(max_events)
    except Exception as exc:
        if "Access is denied" in str(exc) or "access is denied" in str(exc).lower():
            raise PermissionError(
                "Windows từ chối đọc Security Event Log. "
                "Hãy mở PowerShell/Terminal bằng 'Run as administrator' rồi chạy lại."
            ) from exc
        raise RuntimeError(f"Không đọc được Windows Security Event Log: {exc}") from exc


def process_unseen(
    events: list[dict[str, object]],
    seen: set[str],
    engine: OnlineFeatureEngine,
    bundle: LoadedModel,
    threshold: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for event in events:
        record_id = str(event["record_id"])
        if record_id in seen:
            continue
        if engine.last_time is not None and int(event["timestamp"]) < engine.last_time:
            continue
        result = score_event(engine, bundle, event, threshold)
        seen.add(record_id)
        results.append(result)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Nạp XGBoost và phân tích sự kiện đăng nhập Windows ngay trong terminal."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root() / "models" / "xgboost_smote.pkl",
        help="Đường dẫn model bundle.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2_000,
        help="Số sự kiện gần nhất dùng để tạo trạng thái lịch sử (mặc định: 2000).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Ghi đè threshold lưu trong model.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Phân tích log hiện có một lần rồi thoát.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="Với --once, hiển thị N sự kiện mới nhất (mặc định: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Số Event gần nhất đọc lại ở mỗi vòng theo dõi.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=3.0,
        help="Khoảng nghỉ giữa hai lần đọc Event Log.",
    )
    parser.add_argument(
        "--alerts-only",
        action="store_true",
        help="Trong chế độ liên tục, chỉ in sự kiện bị cảnh báo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Tùy chọn lưu kết quả dưới dạng JSON Lines.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 1 or args.show < 1 or args.batch_size < 1:
        raise ValueError("--warmup, --show và --batch-size phải lớn hơn 0.")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds phải lớn hơn 0.")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold phải nằm trong đoạn [0, 1].")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    print("AIC211 TERMINAL AUTHENTICATION ANOMALY DEMO")
    print("-" * 78)
    print(f"Đang tải model: {model_path}")
    bundle = load_cli_model(model_path)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(bundle.threshold)
    )
    print(
        f"Đã tải model | features={len(bundle.feature_columns)} | "
        f"train={bundle.train_source} | threshold={threshold:.6f}"
    )
    print(
        "LƯU Ý: model học từ LANL; kết quả trên Windows laptop chỉ là demo "
        "tích hợp và có thể có domain shift/báo động giả."
    )
    print(f"Đang đọc {args.warmup:,} Security Event gần nhất để warm-up...")

    events = read_events_or_explain(args.warmup)
    engine = OnlineFeatureEngine()
    seen: set[str] = set()
    warmup_results = process_unseen(events, seen, engine, bundle, threshold)
    alert_count = sum(int(item["prediction"]) for item in warmup_results)
    print(
        f"Warm-up hoàn tất: {len(warmup_results):,} Event | "
        f"cảnh báo theo threshold hiện tại: {alert_count:,}"
    )

    if args.once:
        selected = warmup_results[-args.show :]
        for result in selected:
            print(format_analysis(result, threshold))
        if args.output:
            append_jsonl(args.output, selected)
            print(f"\nĐã lưu {len(selected)} kết quả vào: {args.output.resolve()}")
        return 0

    print(
        "\nĐang theo dõi sự kiện mới. Hãy thử đăng nhập/đăng xuất; "
        "nhấn Ctrl+C để dừng."
    )
    try:
        while True:
            events = read_events_or_explain(args.batch_size)
            results = process_unseen(events, seen, engine, bundle, threshold)
            displayed = [
                item
                for item in results
                if not args.alerts_only or int(item["prediction"]) == 1
            ]
            for result in displayed:
                print(format_analysis(result, threshold))
            if args.output and results:
                append_jsonl(args.output, results)
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nĐã dừng theo dõi.")
        return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"\nLỖI: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
