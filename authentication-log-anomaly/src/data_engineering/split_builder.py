from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

try:
    from .feature_builder import (
        FEATURE_SCHEMA_VERSION,
        FEATURES_PATH,
        OUTPUT_COLUMNS,
    )
except ImportError:
    from feature_builder import (
        FEATURE_SCHEMA_VERSION,
        FEATURES_PATH,
        OUTPUT_COLUMNS,
    )


SPLIT_DIR = Path("data/processed/splits")
REPORT_PATH = SPLIT_DIR / "split_report.json"
CHUNK_SIZE = 500_000
SECONDS_PER_DAY = 86_400
SPLIT_NAMES = ("train", "valid", "test")


def _validate_header(input_path: Path) -> None:
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    if header != OUTPUT_COLUMNS:
        missing = [column for column in OUTPUT_COLUMNS if column not in header]
        unexpected = [column for column in header if column not in OUTPUT_COLUMNS]
        raise ValueError(
            f"Feature file does not match {FEATURE_SCHEMA_VERSION}; "
            f"missing={missing}; unexpected={unexpected}; "
            "or column order differs."
        )


def _flush_and_sync(path: Path) -> None:
    with path.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def build_chronological_splits(
    input_path: Path = FEATURES_PATH,
    output_dir: Path = SPLIT_DIR,
    report_path: Path = REPORT_PATH,
    chunksize: int = CHUNK_SIZE,
) -> dict[str, object]:
    """Split three consecutive LANL days without shuffling or label leakage."""
    if not input_path.exists():
        raise FileNotFoundError(f"Feature file not found: {input_path}")
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")
    _validate_header(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: output_dir / f"{name}.csv"
        for name in SPLIT_NAMES
    }
    temporary_paths = {
        name: output_dir / f".{name}.csv.tmp"
        for name in SPLIT_NAMES
    }
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    statistics = {
        name: {
            "rows": 0,
            "benign": 0,
            "anomaly": 0,
            "min_timestamp": None,
            "max_timestamp": None,
        }
        for name in SPLIT_NAMES
    }
    wrote_header = {name: False for name in SPLIT_NAMES}
    first_day: int | None = None
    previous_timestamp: int | None = None
    total_rows = 0
    total_labels = 0
    started = time.perf_counter()

    try:
        reader = pd.read_csv(input_path, chunksize=chunksize)
        try:
            for chunk_number, chunk in enumerate(reader, start=1):
                if chunk.columns.tolist() != OUTPUT_COLUMNS:
                    raise ValueError(
                        f"Feature schema changed at chunk {chunk_number}."
                    )
                timestamps = pd.to_numeric(
                    chunk["timestamp"],
                    errors="raise",
                ).astype("int64")
                labels = pd.to_numeric(
                    chunk["label"],
                    errors="raise",
                ).astype("int8")
                if not labels.isin([0, 1]).all():
                    raise ValueError(
                        f"Non-binary label found in chunk {chunk_number}."
                    )
                if not timestamps.is_monotonic_increasing:
                    raise ValueError(
                        f"Feature timestamps are not sorted in chunk {chunk_number}."
                    )
                if (
                    previous_timestamp is not None
                    and int(timestamps.iloc[0]) < previous_timestamp
                ):
                    raise ValueError("Feature timestamps are not globally sorted.")
                previous_timestamp = int(timestamps.iloc[-1])

                day_numbers = timestamps // SECONDS_PER_DAY
                if first_day is None:
                    first_day = int(day_numbers.iloc[0])
                split_indexes = day_numbers - first_day
                if ((split_indexes < 0) | (split_indexes > 2)).any():
                    bad_days = sorted(split_indexes.unique().tolist())
                    raise ValueError(
                        "Expected exactly three consecutive chronological days; "
                        f"found relative day indexes {bad_days}."
                    )

                for split_index, name in enumerate(SPLIT_NAMES):
                    split_chunk = chunk.loc[split_indexes.eq(split_index)]
                    if split_chunk.empty:
                        continue
                    split_chunk.to_csv(
                        temporary_paths[name],
                        mode="w" if not wrote_header[name] else "a",
                        header=not wrote_header[name],
                        index=False,
                    )
                    wrote_header[name] = True
                    split_labels = labels.loc[split_chunk.index]
                    split_rows = len(split_chunk)
                    split_anomaly = int(split_labels.sum())
                    split_stats = statistics[name]
                    split_stats["rows"] = int(split_stats["rows"]) + split_rows
                    split_stats["anomaly"] = (
                        int(split_stats["anomaly"]) + split_anomaly
                    )
                    split_stats["benign"] = (
                        int(split_stats["benign"])
                        + split_rows
                        - split_anomaly
                    )
                    split_min = int(split_chunk["timestamp"].iloc[0])
                    split_max = int(split_chunk["timestamp"].iloc[-1])
                    if split_stats["min_timestamp"] is None:
                        split_stats["min_timestamp"] = split_min
                    split_stats["max_timestamp"] = split_max

                total_rows += len(chunk)
                total_labels += int(labels.sum())
                print(
                    f"[split_builder] chunk {chunk_number:,}: "
                    f"total_rows={total_rows:,}, labels={total_labels:,}",
                    flush=True,
                )
        finally:
            reader.close()

        if first_day is None:
            raise ValueError("Feature file contains no data rows.")
        empty = [name for name in SPLIT_NAMES if not wrote_header[name]]
        if empty:
            raise ValueError(f"Chronological splits are empty: {empty}")
        if sum(int(item["rows"]) for item in statistics.values()) != total_rows:
            raise AssertionError("Split row counts do not preserve all input rows.")
        if sum(int(item["anomaly"]) for item in statistics.values()) != total_labels:
            raise AssertionError("Split labels do not preserve all input labels.")

        for path in temporary_paths.values():
            _flush_and_sync(path)
        for name in SPLIT_NAMES:
            temporary_paths[name].replace(output_paths[name])

        elapsed = time.perf_counter() - started
        report: dict[str, object] = {
            "valid": True,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_file": str(input_path.resolve()),
            "total_rows": total_rows,
            "total_anomaly": total_labels,
            "first_day_index": first_day,
            "splits": {
                name: {
                    "path": str(output_paths[name].resolve()),
                    **statistics[name],
                }
                for name in SPLIT_NAMES
            },
            "duration_seconds": round(elapsed, 2),
        }
        temporary_report = report_path.with_name(f".{report_path.name}.tmp")
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _flush_and_sync(temporary_report)
        temporary_report.replace(report_path)
        print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
        return report
    except Exception:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create three chronological auth_anomaly_v2 CSV splits.",
    )
    parser.add_argument("--input", type=Path, default=FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--chunksize", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def run() -> None:
    args = _parse_args()
    build_chronological_splits(
        input_path=args.input,
        output_dir=args.output_dir,
        report_path=args.report,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    run()
