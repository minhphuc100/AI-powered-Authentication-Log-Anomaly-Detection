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
    return {
        "hour_of_day": seconds_in_day // 3_600,
        "day_of_week": (seconds // SECONDS_PER_DAY) % 7,
        "is_business_hours": int(8 <= seconds_in_day // 3_600 <= 18),
        "is_weekend": int(((seconds // SECONDS_PER_DAY) % 7) >= 5),
    }


def build_features_stream(
    input_path: Path = PROCESSED_PATH,
    output_path: Path = FEATURES_PATH,
    chunksize: int = CHUNK_SIZE,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Parsed auth log not found: {input_path}. Run parser.py first.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    failed_5m_user = WindowCounter(FIVE_MINUTES)
    failed_5m_src = WindowCounter(FIVE_MINUTES)
    failed_1h_user = WindowCounter(ONE_HOUR)
    total_1h_user = WindowCounter(ONE_HOUR)
    dst_1h_by_src = WindowDistinctCounter(ONE_HOUR)
    src_1h_by_dst = WindowDistinctCounter(ONE_HOUR)
    dst_1h_by_user = WindowDistinctCounter(ONE_HOUR)
    users_1h_by_src = WindowDistinctCounter(ONE_HOUR)

    seen_user_src: set[tuple[str, str]] = set()
    seen_src_dst: set[tuple[str, str]] = set()
    seen_user_dst: set[tuple[str, str]] = set()

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
            src_username = item.get("src_username")
            src_host = item.get("src_host")
            destination_host = item.get("destination_host")
            result = item.get("result")

            for window in (
                failed_5m_user,
                failed_5m_src,
                failed_1h_user,
                total_1h_user,
                dst_1h_by_src,
                src_1h_by_dst,
                dst_1h_by_user,
                users_1h_by_src,
            ):
                window.expire(current_time)

            user_src_key = (src_username, src_host)
            src_dst_key = (src_host, destination_host)
            user_dst_key = (src_username, destination_host)

            total_user_1h = total_1h_user.get(src_username)
            failed_user_1h = failed_1h_user.get(src_username)
            failure_rate_1h_user = round(failed_user_1h / total_user_1h, 4) if total_user_1h else 0.0

            feature_row = {
                "timestamp": current_time,
                "username": username,
                "src_username": src_username,
                "src_host": src_host,
                "destination_host": destination_host,
                "event_type": item.get("event_type"),
                "event_id": item.get("event_id"),
                "auth_type": item.get("auth_type"),
                "logon_type": item.get("logon_type"),
                "auth_orientation": item.get("auth_orientation"),
                "result": result,
                "failed_logins_5m_user": failed_5m_user.get(src_username),
                "failed_logins_5m_src_host": failed_5m_src.get(src_host),
                "failed_logins_1h_user": failed_user_1h,
                "failure_rate_1h_user": failure_rate_1h_user,
                "unique_destination_hosts_1h_per_src_host": dst_1h_by_src.nunique(src_host),
                "unique_src_hosts_1h": src_1h_by_dst.nunique(destination_host),
                "unique_destination_hosts_1h_per_user": dst_1h_by_user.nunique(src_username),
                "unique_users_1h_per_src_host": users_1h_by_src.nunique(src_host),
                "is_new_user_src": int(user_src_key not in seen_user_src),
                "is_new_src_dst": int(src_dst_key not in seen_src_dst),
                "is_new_user_dst": int(user_dst_key not in seen_user_dst),
                "is_success": _as_bool_int(item.get("is_success")),
                "is_fail": _as_bool_int(item.get("is_fail")),
                "is_logon": _as_bool_int(item.get("is_logon")),
                "is_logoff": _as_bool_int(item.get("is_logoff")),
                "is_tgs": _as_bool_int(item.get("is_tgs")),
                "is_machine_account": _as_bool_int(item.get("is_machine_account")),
                **_time_features(current_time),
                "label": _as_bool_int(item.get("label")),
            }
            rows.append(feature_row)

            total_1h_user.add(current_time, src_username)
            dst_1h_by_src.add(current_time, src_host, destination_host)
            src_1h_by_dst.add(current_time, destination_host, src_host)
            dst_1h_by_user.add(current_time, src_username, destination_host)
            users_1h_by_src.add(current_time, src_host, src_username)

            if result == "Fail":
                failed_5m_user.add(current_time, src_username)
                failed_5m_src.add(current_time, src_host)
                failed_1h_user.add(current_time, src_username)

            if not pd.isna(src_username) and not pd.isna(src_host):
                seen_user_src.add(user_src_key)
            if not pd.isna(src_host) and not pd.isna(destination_host):
                seen_src_dst.add(src_dst_key)
            if not pd.isna(src_username) and not pd.isna(destination_host):
                seen_user_dst.add(user_dst_key)

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
