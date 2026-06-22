# data/raw/generate_synthetic_logs.py
#
# Sinh dataset Windows Authentication Log HOÀN TOÀN MỚI (không dựa vào log thật)
# Mô phỏng 1 tổ chức nhỏ trong 30 ngày với:
#   - Nhiều user (nhân viên) hoạt động bình thường
#   - Nhiều kịch bản tấn công khác nhau, rải đều theo thời gian
#   - Tách rõ thời gian để tránh data leakage khi chia train/test
#
# Chạy: python data/raw/generate_synthetic_logs.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# ============================================================
# CẤU HÌNH
# ============================================================
N_DAYS       = 30
START_DATE   = datetime(2026, 5, 18, 0, 0, 0)
OUTPUT_PATH  = "data/raw/auth_logs_raw.csv"

EMPLOYEES = [
    "nguyen.an", "tran.binh", "le.chi", "pham.duc", "hoang.giang",
    "vu.hanh", "do.khanh", "bui.linh", "dang.minh", "ngo.nam",
    "phan.oanh", "ly.phong", "ta.quyen", "cao.son", "vo.thanh"
]

OFFICE_IPS = [
    "10.10.1.11", "10.10.1.12", "10.10.1.13", "10.10.1.14", "10.10.1.15",
    "10.10.1.16", "10.10.1.17", "10.10.1.18", "10.10.1.19", "10.10.1.20",
    "10.10.1.21", "10.10.1.22", "10.10.1.23", "10.10.1.24", "10.10.1.25",
]
EMPLOYEE_HOME_IP = dict(zip(EMPLOYEES, OFFICE_IPS))

VPN_IPS = ["172.16.5.10", "172.16.5.11", "172.16.5.12"]

ATTACKER_IPS = [
    "203.0.113.45", "198.51.100.23", "45.33.32.156",
    "185.220.101.7", "91.219.236.18", "194.26.29.156"
]

WORKSTATIONS = [f"WS-{i:03d}" for i in range(1, 16)]

rows = []


def add_row(t, event_id, username, ip, logon_type, workstation):
    rows.append({
        "TimeCreated": t.strftime("%m/%d/%Y %I:%M:%S %p"),
        "EventID":     event_id,
        "UserName":    username,
        "IpAddress":   ip,
        "LogonType":   logon_type,
        "WorkStation": workstation,
        "Status":      "0xC000006D" if event_id == 4625 else "",
        "SubStatus":   "0xC000006A" if event_id == 4625 else "",
    })


# ============================================================
# PHẦN 1 — NORMAL BEHAVIOR (chiếm đa số dữ liệu)
# ============================================================
for day in range(N_DAYS):
    current_day = START_DATE + timedelta(days=day)
    is_weekend = current_day.weekday() >= 5

    active_employees = EMPLOYEES if not is_weekend else random.sample(EMPLOYEES, 3)

    for emp in active_employees:
        home_ip = EMPLOYEE_HOME_IP[emp]
        ws = random.choice(WORKSTATIONS)

        login_hour   = random.randint(7, 9)
        login_minute = random.randint(0, 59)
        login_time   = current_day.replace(hour=login_hour, minute=login_minute,
                                            second=random.randint(0, 59))

        use_vpn = random.random() < 0.05
        ip_today = random.choice(VPN_IPS) if use_vpn else home_ip
        logon_type = "10" if use_vpn else random.choice(["2", "3"])

        # 8% khả năng gõ sai mật khẩu 1 lần trước khi vào đúng
        if random.random() < 0.08:
            add_row(login_time - timedelta(seconds=random.randint(5, 30)),
                    4625, emp, ip_today, logon_type, ws)

        add_row(login_time, 4624, emp, ip_today, logon_type, ws)

        if random.random() < 0.15:
            t_explicit = login_time + timedelta(hours=random.randint(1, 4))
            add_row(t_explicit, 4648, emp, ip_today, logon_type, ws)

        logoff_time = current_day.replace(
            hour=random.randint(16, 18), minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        add_row(logoff_time, 4634, emp, ip_today, logon_type, ws)

    # SYSTEM noise mỗi ngày
    for _ in range(random.randint(15, 25)):
        t = current_day + timedelta(
            hours=random.randint(0, 23), minutes=random.randint(0, 59)
        )
        add_row(t, 4624, "SYSTEM", np.nan, "5", np.nan)


# ============================================================
# PHẦN 2 — TẤN CÔNG: BRUTE FORCE (nhanh, dồn dập)
# ============================================================
def inject_fast_bruteforce(day_offset, target_user, attacker_ip, n_attempts=60,
                            success_at_end=True):
    base = START_DATE + timedelta(days=day_offset, hours=random.randint(0, 23))
    for i in range(n_attempts):
        t = base + timedelta(seconds=i * random.randint(3, 8))
        add_row(t, 4625, target_user, attacker_ip, "3", "ATTACKER-HOST")
    if success_at_end:
        t_success = base + timedelta(seconds=n_attempts * 6 + 10)
        add_row(t_success, 4624, target_user, attacker_ip, "3", "ATTACKER-HOST")


inject_fast_bruteforce(6,  "nguyen.an",   ATTACKER_IPS[0], n_attempts=70, success_at_end=True)
inject_fast_bruteforce(13, "tran.binh",   ATTACKER_IPS[1], n_attempts=45, success_at_end=False)
inject_fast_bruteforce(21, "le.chi",      ATTACKER_IPS[2], n_attempts=90, success_at_end=True)


# ============================================================
# PHẦN 3 — TẤN CÔNG: BRUTE FORCE CHẬM (low-and-slow)
# ============================================================
def inject_slow_bruteforce(day_offset, target_user, attacker_ip, n_attempts=20,
                            spread_minutes=90):
    base = START_DATE + timedelta(days=day_offset, hours=random.randint(0, 23))
    interval = spread_minutes * 60 / n_attempts
    for i in range(n_attempts):
        t = base + timedelta(seconds=i * interval)
        add_row(t, 4625, target_user, attacker_ip, "3", "ATTACKER-HOST")


inject_slow_bruteforce(9,  "pham.duc",    ATTACKER_IPS[3], n_attempts=25, spread_minutes=120)
inject_slow_bruteforce(17, "hoang.giang", ATTACKER_IPS[4], n_attempts=18, spread_minutes=80)


# ============================================================
# PHẦN 4 — TẤN CÔNG: PASSWORD SPRAY (1 IP, nhiều user)
# ============================================================
def inject_password_spray(day_offset, attacker_ip, n_users=12, interval_sec=15):
    base = START_DATE + timedelta(days=day_offset, hours=random.randint(0, 23))
    targets = random.sample(EMPLOYEES, min(n_users, len(EMPLOYEES)))
    for i, user in enumerate(targets):
        t = base + timedelta(seconds=i * interval_sec)
        add_row(t, 4625, user, attacker_ip, "3", "SPRAY-HOST")


inject_password_spray(4,  ATTACKER_IPS[5], n_users=10, interval_sec=12)
inject_password_spray(19, ATTACKER_IPS[0], n_users=14, interval_sec=20)
inject_password_spray(26, ATTACKER_IPS[2], n_users=8,  interval_sec=8)


# ============================================================
# PHẦN 5 — TẤN CÔNG: CREDENTIAL STUFFING TỪ NHIỀU IP (botnet-like)
# ============================================================
def inject_distributed_attack(day_offset, target_user, n_ips=5, attempts_per_ip=6):
    base = START_DATE + timedelta(days=day_offset, hours=random.randint(0, 23))
    botnet_ips = [f"103.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
                  for _ in range(n_ips)]
    for ip in botnet_ips:
        for i in range(attempts_per_ip):
            t = base + timedelta(seconds=random.randint(0, 300))
            add_row(t, 4625, target_user, ip, "3", "BOTNET")


inject_distributed_attack(23, "vu.hanh", n_ips=6, attempts_per_ip=5)


# ============================================================
# PHẦN 6 — ĐĂNG NHẬP NGOÀI GIỜ BẤT THƯỜNG (insider threat mô phỏng)
# ============================================================
def inject_odd_hour_login(day_offset, user, ip):
    t = START_DATE + timedelta(days=day_offset,
                                hours=random.randint(2, 4),
                                minutes=random.randint(0, 59))
    add_row(t, 4624, user, ip, "10", "UNKNOWN-HOST")


inject_odd_hour_login(11, "dang.minh", "172.16.99.50")
inject_odd_hour_login(24, "phan.oanh", "172.16.99.51")


# ============================================================
# GHÉP, SẮP XẾP, LƯU
# ============================================================
df = pd.DataFrame(rows)
df["_sort_ts"] = pd.to_datetime(df["TimeCreated"], format="%m/%d/%Y %I:%M:%S %p")
df = df.sort_values("_sort_ts").drop(columns=["_sort_ts"]).reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Done! Total rows: {len(df):,}")
print(f"\nEventID distribution:")
print(df["EventID"].value_counts().to_string())
print(f"\nUnique users: {df['UserName'].nunique()}")
print(f"Unique IPs: {df['IpAddress'].nunique()}")
print(f"\nSaved to: {OUTPUT_PATH}")
print(f"\nGợi ý chia train/test: dùng 22 ngày đầu cho train, 8 ngày cuối cho test")
print(f"  Train: {START_DATE.date()} -> {(START_DATE + timedelta(days=21)).date()}")
print(f"  Test:  {(START_DATE + timedelta(days=22)).date()} -> {(START_DATE + timedelta(days=N_DAYS-1)).date()}")