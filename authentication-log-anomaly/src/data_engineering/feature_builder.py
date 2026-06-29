from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Hashable

import pandas as pd


PROCESSED_PATH = Path("data/processed/parsed_auth_logs.csv")
FEATURES_PATH = Path("data/processed/features.csv")

CHUNK_SIZE = 500_000
FIVE_MINUTES = 300
ONE_HOUR = 3_600
SECONDS_PER_DAY = 86_400


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


def _time_features(seconds: int) -> dict[str, int]:
    seconds_in_day = seconds % SECONDS_PER_DAY
    hour_of_day = seconds_in_day // 3_600
    day_of_week = (seconds // SECONDS_PER_DAY) % 7
    return {
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_business_hours": int(8 <= hour_of_day <= 18 and day_of_week <= 4),
    }


def _is_failure(result: object, event_id: object) -> bool:
    return result == "Fail" or event_id == 4625


def _is_network_logon(logon_type: object) -> int:
    return int(str(logon_type) in {"Network", "3", "8"})


def build_features_stream(
    input_path: Path = PROCESSED_PATH,
    output_path: Path = FEATURES_PATH,
    chunksize: int = CHUNK_SIZE,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Parsed auth log not found: {input_path}. Run parser.py first.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    failed_5m_user = WindowCounter(FIVE_MINUTES)
    failed_5m_ip = WindowCounter(FIVE_MINUTES)
    failed_1h_user = WindowCounter(ONE_HOUR)
    total_1h_user = WindowCounter(ONE_HOUR)
    src_ips_1h_by_user = WindowDistinctCounter(ONE_HOUR)
    users_1h_by_ip = WindowDistinctCounter(ONE_HOUR)

    wrote_header = False
    total_rows = 0
    total_labels = 0
    last_time = None

    reader = pd.read_csv(input_path, chunksize=chunksize)
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

            username = item.get("username")
            src_ip = item.get("src_host")
            event_id = item.get("event_id")
            logon_type = item.get("logon_type")
            result = item.get("result")

            for window in (
                failed_5m_user,
                failed_5m_ip,
                failed_1h_user,
                total_1h_user,
                src_ips_1h_by_user,
                users_1h_by_ip,
            ):
                window.expire(current_time)

            total_user_1h = total_1h_user.get(username)
            failed_user_1h = failed_1h_user.get(username)
            failure_rate_1h = round(failed_user_1h / total_user_1h, 4) if total_user_1h else 0.0

            feature_row = {
                "failed_logins_5m_user": failed_5m_user.get(username),
                "unique_src_ip_1h": src_ips_1h_by_user.nunique(username),
                "failed_logins_5m_ip": failed_5m_ip.get(src_ip),
                "failure_rate_1h": failure_rate_1h,
                "unique_users_1h_per_ip": users_1h_by_ip.nunique(src_ip),
                **_time_features(current_time),
                "is_network_logon": _is_network_logon(logon_type),
                "label": _as_bool_int(item.get("label")),
            }
            rows.append(feature_row)

            total_1h_user.add(current_time, username)
            src_ips_1h_by_user.add(current_time, username, src_ip)
            users_1h_by_ip.add(current_time, src_ip, username)

            if _is_failure(result, event_id):
                failed_5m_user.add(current_time, username)
                failed_5m_ip.add(current_time, src_ip)
                failed_1h_user.add(current_time, username)

        features = pd.DataFrame(rows)
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
    temp_input = FEATURES_PATH.parent / "_tmp_parsed_for_features.csv"
    temp_output = FEATURES_PATH.parent / "_tmp_features.csv"
    df.to_csv(temp_input, index=False)
    build_features_stream(temp_input, temp_output, chunksize=len(df) or 1)
    features = pd.read_csv(temp_output)
    temp_input.unlink(missing_ok=True)
    temp_output.unlink(missing_ok=True)
    return features


def run() -> None:
    build_features_stream()


if __name__ == "__main__":
    run()
