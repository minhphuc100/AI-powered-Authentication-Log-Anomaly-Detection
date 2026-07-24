from collections import Counter, defaultdict, deque
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Deque, Hashable

import pandas as pd


PROCESSED_PATH = Path("data/processed/parsed_auth_logs.csv")
FEATURES_PATH = Path("data/processed/features.csv")

CHUNK_SIZE = 500_000
FIVE_MINUTES = 300
ONE_HOUR = 3_600
SECONDS_PER_DAY = 86_400
MISSING_TIME_DELTA = -1

FEATURE_SCHEMA_VERSION = "auth_anomaly_v2"
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
OUTPUT_COLUMNS = ["timestamp", *FEATURE_COLUMNS, "label"]
REQUIRED_PARSED_COLUMNS = [
    "timestamp",
    "src_username",
    "username",
    "src_host",
    "destination_host",
    "logon_type",
    "event_id",
    "label",
]


class WindowCounter:
    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self.events: Deque[tuple[int, Hashable]] = deque()
        self.counts: Counter[Hashable] = Counter()

    def expire(self, current_time: int) -> None:
        cutoff = current_time - self.window_seconds
        while self.events and self.events[0][0] < cutoff:
            _, key = self.events.popleft()
            self.counts[key] -= 1
            if self.counts[key] <= 0:
                del self.counts[key]

    def add(self, current_time: int, key: Hashable) -> None:
        if pd.isna(key):
            return
        self.events.append((current_time, key))
        self.counts[key] += 1

    def get(self, key: Hashable) -> int:
        if pd.isna(key):
            return 0
        return int(self.counts.get(key, 0))


class WindowDistinctCounter:
    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self.events: Deque[tuple[int, Hashable, Hashable]] = deque()
        self.counts_by_group: dict[Hashable, Counter[Hashable]] = defaultdict(Counter)

    def expire(self, current_time: int) -> None:
        cutoff = current_time - self.window_seconds
        while self.events and self.events[0][0] < cutoff:
            _, group_key, value_key = self.events.popleft()
            group_counts = self.counts_by_group[group_key]
            group_counts[value_key] -= 1
            if group_counts[value_key] <= 0:
                del group_counts[value_key]
            if not group_counts:
                del self.counts_by_group[group_key]

    def add(self, current_time: int, group_key: Hashable, value_key: Hashable) -> None:
        if pd.isna(group_key) or pd.isna(value_key):
            return
        self.events.append((current_time, group_key, value_key))
        self.counts_by_group[group_key][value_key] += 1

    def nunique(self, group_key: Hashable) -> int:
        if pd.isna(group_key):
            return 0
        return len(self.counts_by_group.get(group_key, ()))


def _as_bool_int(value: object) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def _hour_of_day(seconds: int) -> int:
    return (seconds % SECONDS_PER_DAY) // 3_600


def _is_network_logon(logon_type: object) -> int:
    return int(str(logon_type) in {"Network", "3", "8"})


def _is_success(event_id: object) -> int:
    if pd.isna(event_id):
        return 0
    try:
        return int(float(event_id) == 4624)
    except (TypeError, ValueError):
        return 0


def _valid_key(*values: object) -> bool:
    return all(
        not pd.isna(value) and str(value).strip() not in {"", "?", "-"}
        for value in values
    )


def _first_seen_flag(seen: set[Hashable], key: Hashable, valid: bool) -> int:
    return int(valid and key not in seen)


def _seconds_since(last_seen: dict[Hashable, int], key: Hashable, current_time: int, valid: bool) -> int:
    if not valid:
        return MISSING_TIME_DELTA
    previous = last_seen.get(key)
    if previous is None:
        return MISSING_TIME_DELTA
    return current_time - previous


def build_features_stream(
    input_path: Path = PROCESSED_PATH,
    output_path: Path = FEATURES_PATH,
    chunksize: int = CHUNK_SIZE,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Parsed auth log not found: {input_path}. Run parser.py first.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    missing = [column for column in REQUIRED_PARSED_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"Parsed auth log is missing required columns: {missing}")

    attempts_5m_by_src_user = WindowCounter(FIVE_MINUTES)
    successes_5m_by_src_user = WindowCounter(FIVE_MINUTES)
    src_computers_1h_by_src_user = WindowDistinctCounter(ONE_HOUR)
    dst_computers_5m_by_src_user = WindowDistinctCounter(FIVE_MINUTES)
    dst_computers_1h_by_src_user = WindowDistinctCounter(ONE_HOUR)
    dst_users_1h_by_src_computer = WindowDistinctCounter(ONE_HOUR)
    src_users_1h_by_dst_computer = WindowDistinctCounter(ONE_HOUR)
    auth_count_1h_by_src_user_dst_computer = WindowCounter(ONE_HOUR)

    seen_src_user_src_computer: set[tuple[object, object]] = set()
    seen_src_user_dst_computer: set[tuple[object, object]] = set()
    seen_src_computer_dst_computer: set[tuple[object, object]] = set()
    last_auth_by_src_user: dict[object, int] = {}
    last_auth_by_src_user_dst_computer: dict[tuple[object, object], int] = {}

    wrote_header = False
    total_rows = 0
    total_labels = 0
    last_time: int | None = None

    reader = pd.read_csv(
        input_path,
        usecols=REQUIRED_PARSED_COLUMNS,
        chunksize=chunksize,
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        rows: list[dict[str, object]] = []

        for row in chunk.itertuples(index=False):
            item = row._asdict()
            current_time = int(item["timestamp"])

            if last_time is not None and current_time < last_time:
                raise ValueError(
                    "Parsed data must be sorted by time for streaming features. "
                    "Sort the auth log before feature building."
                )
            last_time = current_time

            src_username = item.get("src_username")
            dst_username = item.get("username")
            src_computer = item.get("src_host")
            dst_computer = item.get("destination_host")
            logon_type = item.get("logon_type")
            event_id = item.get("event_id")

            for window in (
                attempts_5m_by_src_user,
                successes_5m_by_src_user,
                src_computers_1h_by_src_user,
                dst_computers_5m_by_src_user,
                dst_computers_1h_by_src_user,
                dst_users_1h_by_src_computer,
                src_users_1h_by_dst_computer,
                auth_count_1h_by_src_user_dst_computer,
            ):
                window.expire(current_time)

            label = _as_bool_int(item.get("label"))
            current_event_is_success = _is_success(event_id)

            src_user_src_computer = (src_username, src_computer)
            src_user_dst_computer = (src_username, dst_computer)
            src_computer_dst_computer = (src_computer, dst_computer)
            valid_src_user = _valid_key(src_username)
            valid_src_user_src_computer = _valid_key(src_username, src_computer)
            valid_src_user_dst_computer = _valid_key(src_username, dst_computer)
            valid_src_computer_dst_computer = _valid_key(src_computer, dst_computer)

            feature_row = {
                "timestamp": current_time,
                "auth_attempts_5m_per_src_user": attempts_5m_by_src_user.get(src_username),
                "successful_auths_5m_per_src_user": successes_5m_by_src_user.get(src_username),
                "unique_src_computers_1h_per_src_user": src_computers_1h_by_src_user.nunique(src_username),
                "unique_dst_computers_5m_per_src_user": dst_computers_5m_by_src_user.nunique(src_username),
                "unique_dst_computers_1h_per_src_user": dst_computers_1h_by_src_user.nunique(src_username),
                "unique_dst_users_1h_per_src_computer": dst_users_1h_by_src_computer.nunique(src_computer),
                "unique_src_users_1h_per_dst_computer": src_users_1h_by_dst_computer.nunique(dst_computer),
                "prior_auth_count_1h_src_user_dst_computer": auth_count_1h_by_src_user_dst_computer.get(
                    src_user_dst_computer
                ),
                "is_first_seen_src_user_src_computer": _first_seen_flag(
                    seen_src_user_src_computer,
                    src_user_src_computer,
                    valid_src_user_src_computer,
                ),
                "is_first_seen_src_user_dst_computer": _first_seen_flag(
                    seen_src_user_dst_computer,
                    src_user_dst_computer,
                    valid_src_user_dst_computer,
                ),
                "is_first_seen_src_computer_dst_computer": _first_seen_flag(
                    seen_src_computer_dst_computer,
                    src_computer_dst_computer,
                    valid_src_computer_dst_computer,
                ),
                "seconds_since_last_auth_by_src_user": _seconds_since(
                    last_auth_by_src_user,
                    src_username,
                    current_time,
                    valid_src_user,
                ),
                "seconds_since_last_src_user_dst_computer": _seconds_since(
                    last_auth_by_src_user_dst_computer,
                    src_user_dst_computer,
                    current_time,
                    valid_src_user_dst_computer,
                ),
                "current_event_is_success": current_event_is_success,
                "hour_of_day": _hour_of_day(current_time),
                "is_network_logon": _is_network_logon(logon_type),
                "label": label,
            }
            rows.append(feature_row)

            attempts_5m_by_src_user.add(current_time, src_username)
            if current_event_is_success:
                successes_5m_by_src_user.add(current_time, src_username)
            src_computers_1h_by_src_user.add(current_time, src_username, src_computer)
            dst_computers_5m_by_src_user.add(current_time, src_username, dst_computer)
            dst_computers_1h_by_src_user.add(current_time, src_username, dst_computer)
            dst_users_1h_by_src_computer.add(current_time, src_computer, dst_username)
            src_users_1h_by_dst_computer.add(current_time, dst_computer, src_username)
            if valid_src_user_dst_computer:
                auth_count_1h_by_src_user_dst_computer.add(
                    current_time,
                    src_user_dst_computer,
                )

            if valid_src_user_src_computer:
                seen_src_user_src_computer.add(src_user_src_computer)
            if valid_src_user_dst_computer:
                seen_src_user_dst_computer.add(src_user_dst_computer)
                last_auth_by_src_user_dst_computer[src_user_dst_computer] = current_time
            if valid_src_computer_dst_computer:
                seen_src_computer_dst_computer.add(src_computer_dst_computer)
            if valid_src_user:
                last_auth_by_src_user[src_username] = current_time

        features = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        features.to_csv(output_path, mode="w" if not wrote_header else "a", header=not wrote_header, index=False)
        wrote_header = True

        total_rows += len(features)
        total_labels += int(features["label"].sum()) if not features.empty else 0
        print(
            f"[feature_builder] chunk {chunk_index:,}: rows={len(features):,}, "
            f"labels={int(features['label'].sum()) if not features.empty else 0:,}, "
            f"total={total_rows:,}"
        )

    print(f"[feature_builder] Saved {total_rows:,} rows -> {output_path}")
    print(f"[feature_builder] Labels in feature file: {total_labels:,}")


def load(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[feature_builder] Loaded {len(df):,} rows")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    with TemporaryDirectory(prefix="auth_anomaly_features_") as temp_dir:
        temp_root = Path(temp_dir)
        temp_input = temp_root / "parsed.csv"
        temp_output = temp_root / "features.csv"
        df.to_csv(temp_input, index=False)
        build_features_stream(temp_input, temp_output, chunksize=len(df) or 1)
        return pd.read_csv(temp_output)


def run() -> None:
    build_features_stream()


if __name__ == "__main__":
    run()
