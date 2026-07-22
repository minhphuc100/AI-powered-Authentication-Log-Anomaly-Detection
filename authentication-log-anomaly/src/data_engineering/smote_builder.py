from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_DIR = Path("data/processed/splits")
TRAIN_PATH = SPLIT_DIR / "train.csv"
OUTPUT_PATH = SPLIT_DIR / "train_smote.csv"

CHUNK_SIZE = 250_000
DEFAULT_MULTIPLIER = 20
DEFAULT_K_NEIGHBORS = 5
DEFAULT_RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "failed_logins_5m_user",
    "unique_src_ip_1h",
    "failed_logins_5m_ip",
    "failure_rate_1h",
    "unique_users_1h_per_ip",
    "hour_of_day",
    "day_of_week",
    "is_business_hours",
    "is_network_logon",
]

INTEGER_COLUMNS = [
    "failed_logins_5m_user",
    "unique_src_ip_1h",
    "failed_logins_5m_ip",
    "unique_users_1h_per_ip",
    "hour_of_day",
    "day_of_week",
    "is_business_hours",
    "is_network_logon",
]

OUTPUT_COLUMNS = FEATURE_COLUMNS + ["is_synthetic", "label"]


def _collect_class_counts_and_attacks(
    train_path: Path,
    chunksize: int,
) -> tuple[int, int, pd.DataFrame]:
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")

    benign = 0
    attack = 0
    attack_frames: list[pd.DataFrame] = []
    reader = pd.read_csv(
        train_path,
        usecols=FEATURE_COLUMNS + ["label"],
        chunksize=chunksize,
    )
    try:
        for chunk in reader:
            labels = pd.to_numeric(chunk["label"], errors="coerce").fillna(0).astype("int8")
            is_attack = labels.eq(1)
            benign += int((~is_attack).sum())
            attack += int(is_attack.sum())
            if is_attack.any():
                attack_frames.append(chunk.loc[is_attack, FEATURE_COLUMNS].copy())
    finally:
        reader.close()

    if attack < 2:
        raise ValueError(f"SMOTE requires at least two attack rows; found {attack}.")
    attacks = pd.concat(attack_frames, ignore_index=True)
    attacks = attacks.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return benign, attack, attacks


def _generate_synthetic_attacks(
    attacks: pd.DataFrame,
    target_attack_count: int,
    k_neighbors: int,
    random_state: int,
) -> pd.DataFrame:
    original_count = len(attacks)
    synthetic_count = target_attack_count - original_count
    if synthetic_count <= 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be greater than zero")

    values = attacks[FEATURE_COLUMNS].astype("float64").to_numpy(copy=True)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales == 0] = 1.0
    scaled = (values - means) / scales
    neighbor_count = min(k_neighbors, original_count - 1)
    differences = scaled[:, np.newaxis, :] - scaled[np.newaxis, :, :]
    squared_distances = np.einsum("ijk,ijk->ij", differences, differences)
    np.fill_diagonal(squared_distances, np.inf)
    neighbor_indexes = np.argpartition(
        squared_distances,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]

    rng = np.random.default_rng(random_state)
    base_indexes = rng.integers(0, original_count, size=synthetic_count)
    neighbor_choices = rng.integers(0, neighbor_count, size=synthetic_count)
    selected_neighbors = neighbor_indexes[base_indexes, neighbor_choices]
    interpolation = rng.random((synthetic_count, 1))
    synthetic_values = values[base_indexes] + interpolation * (
        values[selected_neighbors] - values[base_indexes]
    )

    synthetic = pd.DataFrame(synthetic_values, columns=FEATURE_COLUMNS)
    synthetic[INTEGER_COLUMNS] = synthetic[INTEGER_COLUMNS].round().astype("int64")
    synthetic["failure_rate_1h"] = synthetic["failure_rate_1h"].clip(0.0, 1.0)
    for column in (
        "failed_logins_5m_user",
        "unique_src_ip_1h",
        "failed_logins_5m_ip",
        "unique_users_1h_per_ip",
        "hour_of_day",
        "day_of_week",
    ):
        synthetic[column] = synthetic[column].clip(lower=0)
    synthetic["hour_of_day"] = synthetic["hour_of_day"].clip(upper=23)
    synthetic["day_of_week"] = synthetic["day_of_week"].clip(upper=6)
    synthetic["is_business_hours"] = synthetic["is_business_hours"].clip(0, 1)
    synthetic["is_network_logon"] = synthetic["is_network_logon"].clip(0, 1)
    return synthetic


def _write_original_train(
    train_path: Path,
    temporary_path: Path,
    chunksize: int,
) -> int:
    wrote_header = False
    rows = 0
    reader = pd.read_csv(
        train_path,
        usecols=FEATURE_COLUMNS + ["label"],
        chunksize=chunksize,
    )
    try:
        for chunk in reader:
            chunk["is_synthetic"] = 0
            chunk = chunk[OUTPUT_COLUMNS]
            chunk.to_csv(
                temporary_path,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
            rows += len(chunk)
    finally:
        reader.close()
    return rows


def build_smote_train(
    train_path: Path = TRAIN_PATH,
    output_path: Path = OUTPUT_PATH,
    chunksize: int = CHUNK_SIZE,
    attack_multiplier: int = DEFAULT_MULTIPLIER,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    """Preserve the original train set and append SMOTE-style attack rows.

    Only attack rows are retained in memory. The full train file is copied in
    chunks, so memory usage does not grow with the number of benign rows.
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_path}")
    if attack_multiplier < 1:
        raise ValueError("attack_multiplier must be at least one")

    started = time.perf_counter()
    benign, attack, attacks = _collect_class_counts_and_attacks(train_path, chunksize)
    target_attack = attack * attack_multiplier
    synthetic = _generate_synthetic_attacks(
        attacks,
        target_attack,
        k_neighbors,
        random_state,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        original_rows = _write_original_train(train_path, temporary_path, chunksize)
        if not synthetic.empty:
            synthetic["is_synthetic"] = 1
            synthetic["label"] = 1
            synthetic[OUTPUT_COLUMNS].to_csv(
                temporary_path,
                mode="a",
                header=False,
                index=False,
            )
        with temporary_path.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - started
    report = {
        "train_input": str(train_path.resolve()),
        "train_output": str(output_path.resolve()),
        "benign_original": benign,
        "attack_original": attack,
        "attack_fit_sample": len(attacks),
        "synthetic_attack": len(synthetic),
        "attack_after": attack + len(synthetic),
        "all_original_train_rows_preserved": original_rows == benign + attack,
        "validation_test_modified": False,
        "attack_multiplier": attack_multiplier,
        "k_neighbors": min(k_neighbors, attack - 1),
        "random_state": random_state,
        "duration_seconds": round(elapsed, 2),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a bounded-memory SMOTE-augmented train CSV.",
    )
    parser.add_argument("--input", type=Path, default=TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--chunksize", type=int, default=CHUNK_SIZE)
    parser.add_argument("--attack-multiplier", type=int, default=DEFAULT_MULTIPLIER)
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def run() -> None:
    args = _parse_args()
    build_smote_train(
        train_path=args.input,
        output_path=args.output,
        chunksize=args.chunksize,
        attack_multiplier=args.attack_multiplier,
        k_neighbors=args.k_neighbors,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    run()
