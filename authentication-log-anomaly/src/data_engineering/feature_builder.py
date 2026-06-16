# src/data_engineering/feature_builder.py

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed/parsed_auth_logs.csv")
FEATURES_PATH  = Path("data/processed/features.csv")

# ============================================================
# Ngưỡng gắn nhãn - chỉnh ở đây nếu muốn thay đổi độ nhạy
# ============================================================
BRUTE_FORCE_THRESHOLD    = 10   # >= 10 lần thất bại từ 1 IP trong 5 phút
SPRAY_THRESHOLD          = 5    # >= 5 user khác nhau từ 1 IP trong 1 giờ
FAILURE_RATE_THRESHOLD   = 0.8  # >= 80% tỉ lệ thất bại trong 1 giờ
POST_BRUTE_SUCCESS       = 5    # login thành công sau >= 5 lần thất bại


def load(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[feature_builder] Loaded {len(df):,} rows")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính time-window features cho từng event.
    Đây là core của anomaly detection pipeline.
    """
    df = df.set_index("timestamp").sort_index()
    results = []

    total = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"[feature_builder] Processing {i:,}/{total:,}...")

        user = row["username"]
        ip   = row["src_ip"]

        # Time windows
        w5m = df.loc[idx - pd.Timedelta("5min") : idx]
        w1h = df.loc[idx - pd.Timedelta("1h")   : idx]

        # --------------------------------------------------
        # FEATURE 1: Số lần thất bại trong 5 phút (per user)
        # Phát hiện: brute force nhắm vào 1 tài khoản
        # --------------------------------------------------
        failed_5m_user = 0
        if pd.notna(user):
            failed_5m_user = len(w5m[
                (w5m["username"] == user) & (w5m["event_id"] == 4625)
            ])

        # --------------------------------------------------
        # FEATURE 2: Số IP khác nhau tấn công 1 user trong 1 giờ
        # Phát hiện: distributed brute force (nhiều máy cùng tấn công)
        # --------------------------------------------------
        unique_ip_1h = 0
        if pd.notna(user):
            unique_ip_1h = w1h[w1h["username"] == user]["src_ip"].nunique()

        # --------------------------------------------------
        # FEATURE 3: Số lần thất bại từ 1 IP trong 5 phút
        # Phát hiện: brute force từ 1 nguồn tấn công
        # --------------------------------------------------
        failed_5m_ip = 0
        if pd.notna(ip):
            failed_5m_ip = len(w5m[
                (w5m["src_ip"] == ip) & (w5m["event_id"] == 4625)
            ])

        # --------------------------------------------------
        # FEATURE 4: Tỉ lệ thất bại / tổng login trong 1 giờ
        # Phát hiện: tỉ lệ cao bất thường dù số lần ít
        # --------------------------------------------------
        failure_rate_1h = 0.0
        if pd.notna(user):
            total_1h = len(w1h[w1h["username"] == user])
            fail_1h  = len(w1h[
                (w1h["username"] == user) & (w1h["event_id"] == 4625)
            ])
            failure_rate_1h = round(fail_1h / total_1h, 4) if total_1h > 0 else 0.0

        # --------------------------------------------------
        # FEATURE 5: Số user khác nhau bị tấn công từ 1 IP trong 1 giờ
        # Phát hiện: password spray (1 IP thử nhiều tài khoản)
        # --------------------------------------------------
        unique_users_1h_per_ip = 0
        if pd.notna(ip):
            unique_users_1h_per_ip = w1h[
                w1h["src_ip"] == ip
            ]["username"].nunique()

        # --------------------------------------------------
        # FEATURE 6-9: Thời gian (behavior baseline)
        # --------------------------------------------------
        hour_of_day       = idx.hour
        day_of_week       = idx.dayofweek
        is_business_hours = int((8 <= idx.hour <= 18) and (idx.dayofweek <= 4))
        is_network_logon  = int(str(row.get("logon_type", "")) in ["3", "8"])

        results.append({
            # --- Metadata ---
            "timestamp":               idx,
            "username":                user,
            "src_ip":                  ip,
            "event_id":                row["event_id"],
            "event_type":              row["event_type"],
            # --- Features ---
            "failed_logins_5m_user":   failed_5m_user,
            "unique_src_ip_1h":        unique_ip_1h,
            "failed_logins_5m_ip":     failed_5m_ip,
            "failure_rate_1h":         failure_rate_1h,
            "unique_users_1h_per_ip":  unique_users_1h_per_ip,
            "hour_of_day":             hour_of_day,
            "day_of_week":             day_of_week,
            "is_business_hours":       is_business_hours,
            "is_network_logon":        is_network_logon,
        })

    return pd.DataFrame(results)


def assign_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gắn nhãn dựa trên PATTERN (time-window features),
    không phải từng event đơn lẻ.

    Label:
        0 = normal
        1 = anomaly (brute force / password spray / suspicious)

    Lý do không dùng event_id == 4625 làm nhãn trực tiếp:
        - 1 lần gõ sai mật khẩu cũng bị label = 1 → quá nhạy
        - Không capture được pattern tấn công thật sự
    """
    df = df.copy()
    df["label"] = 0  # mặc định: normal

    # --- Rule 1: Brute force từ 1 IP ---
    # Dấu hiệu: >= 10 lần thất bại từ cùng 1 IP trong 5 phút
    mask_bf = df["failed_logins_5m_ip"] >= BRUTE_FORCE_THRESHOLD
    df.loc[mask_bf, "label"] = 1

    # --- Rule 2: Password Spray ---
    # Dấu hiệu: 1 IP thử >= 5 user khác nhau trong 1 giờ
    mask_spray = df["unique_users_1h_per_ip"] >= SPRAY_THRESHOLD
    df.loc[mask_spray, "label"] = 1

    # --- Rule 3: Tỉ lệ thất bại cao ---
    # Dấu hiệu: >= 80% login là thất bại trong 1 giờ
    mask_rate = (
        (df["failure_rate_1h"] >= FAILURE_RATE_THRESHOLD) &
        (df["event_id"] == 4625)
    )
    df.loc[mask_rate, "label"] = 1

    # --- Rule 4: Login thành công SAU brute force ---
    # Đây là dấu hiệu nguy hiểm nhất: tấn công thành công
    mask_post = (
        (df["event_id"] == 4624) &
        (df["failed_logins_5m_user"] >= POST_BRUTE_SUCCESS)
    )
    df.loc[mask_post, "label"] = 1

    # In thống kê nhãn
    counts = df["label"].value_counts()
    total  = len(df)
    print(f"\n[feature_builder] Label distribution:")
    print(f"  Normal  (0): {counts.get(0, 0):,}  ({counts.get(0,0)/total*100:.1f}%)")
    print(f"  Anomaly (1): {counts.get(1, 0):,}  ({counts.get(1,0)/total*100:.1f}%)")

    return df


def save(df: pd.DataFrame, path: Path = FEATURES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n[feature_builder] Saved {len(df):,} rows → {path}")


def run():
    df       = load()
    features = build_features(df)
    features = assign_label(features)
    save(features)
    return features


if __name__ == "__main__":
    run()
