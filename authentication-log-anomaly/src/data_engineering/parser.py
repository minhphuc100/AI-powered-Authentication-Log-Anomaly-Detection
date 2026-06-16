# src/data_engineering/parser.py

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH       = Path("data/raw/auth_logs_raw.csv")
PROCESSED_PATH = Path("data/processed/parsed_auth_logs.csv")

EVENT_MAP = {
    4624: "logon_success",
    4625: "logon_failure",
    4648: "explicit_logon",
    4634: "logoff"
}

LOGON_TYPE_MAP = {
    "2":  "interactive",
    "3":  "network",
    "4":  "batch",
    "5":  "service",
    "7":  "unlock",
    "8":  "network_cleartext",
    "10": "remote_interactive",
    "11": "cached_interactive"
}

SYSTEM_ACCOUNTS = {"SYSTEM", "LOCAL SERVICE", "NETWORK SERVICE", "-", ""}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # FIX 1: Chỉ định rõ format timestamp → không còn warning
    df["TimeCreated"] = pd.to_datetime(
        df["TimeCreated"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )

    # FIX 2: Drop Status/SubStatus ngay từ đầu vì ~97% null, không dùng được
    df = df.drop(columns=["Status", "SubStatus"], errors="ignore")

    # FIX 3: Fix IP "-" thành NaN ngay lúc load (trước khi rename cột)
    df["IpAddress"] = df["IpAddress"].replace({"-": np.nan, "::1": np.nan})

    print(f"[parser] Loaded {len(df):,} rows from {path}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Chuẩn hóa tên cột
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df = df.rename(columns={
        "timecreated": "timestamp",
        "eventid":     "event_id",
        "username":    "username",
        "ipaddress":   "src_ip",
        "logontype":   "logon_type",
        "workstation": "workstation",
    })

    # Drop timestamp không hợp lệ
    df = df.dropna(subset=["timestamp"])

    # FIX 4: Deduplicate — inject_bruteforce chạy 2 lần gây ra 909 dòng trùng
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"[parser] Removed {before - after:,} duplicate rows")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Map event_id → tên rõ nghĩa
    df["event_type"] = df["event_id"].map(EVENT_MAP).fillna("unknown")

    # Map logon_type → tên rõ nghĩa
    df["logon_type_name"] = df["logon_type"].astype(str).map(LOGON_TYPE_MAP).fillna("unknown")

    # Làm sạch username — system accounts không cần detect anomaly
    df["username"] = df["username"].str.strip()
    df.loc[df["username"].isin(SYSTEM_ACCOUNTS), "username"] = np.nan

    # Làm sạch workstation
    df["workstation"] = df["workstation"].replace({"-": np.nan})

    # KHÔNG gắn label ở đây — label phụ thuộc vào time-window features
    # nên để feature_builder.py xử lý sau

    print(f"[parser] After cleaning: {len(df):,} rows")
    print(f"[parser] Event distribution:")
    print(df["event_type"].value_counts().to_string())
    print(f"[parser] IP coverage: {df['src_ip'].notna().sum():,} / {len(df):,} rows có IP")
    return df


def save(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[parser] Saved {len(df):,} rows → {path}")


def run() -> pd.DataFrame:
    df = load_raw()
    df = clean(df)
    save(df)
    return df


if __name__ == "__main__":
    run()
