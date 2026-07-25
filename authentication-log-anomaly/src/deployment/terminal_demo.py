from datetime import datetime, timezone
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import joblib
import pandas as pd
import win32event
import win32evtlog

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.data_engineering.feature_builder import (
        FIVE_MINUTES,
        ONE_HOUR,
        WindowCounter,
        WindowDistinctCounter,
        _first_seen_flag,
        _hour_of_day,
        _is_network_logon,
        _is_success,
        _seconds_since,
        _valid_key,
    )
except ImportError as e:
    print(f"[ERROR] Không import được feature_builder: {e}")
    sys.exit(1)

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_weighted.pkl"
NS = {"evt": "http://schemas.microsoft.com/win/2004/08/events/event"}


def parse_iso_timestamp(sys_time_str: str) -> int:
    """Parse SystemTime ISO từ TimeCreated sang Unix Timestamp."""
    if not sys_time_str or sys_time_str == "-":
        return int(time.time())
    try:
        clean_str = sys_time_str.split(".")[0] + "Z"
        dt = datetime.strptime(clean_str, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        return int(dt.timestamp())
    except Exception:
        return int(time.time())


def parse_xml_event(xml_str: str):
    """Parse XML và lấy thời gian từ TimeCreated."""
    try:
        root = ET.fromstring(xml_str)
        system = root.find("evt:System", NS)

        event_id_elem = (
            system.find("evt:EventID", NS) if system is not None else None
        )
        event_id = (
            int(event_id_elem.text) if event_id_elem is not None else None
        )

        if event_id not in (4624, 4625):
            return None

        record_id_elem = (
            system.find("evt:EventRecordID", NS) if system is not None else None
        )
        record_id = (
            int(record_id_elem.text) if record_id_elem is not None else None
        )

        time_created_node = (
            system.find("evt:TimeCreated", NS) if system is not None else None
        )
        sys_time_str = (
            time_created_node.attrib.get("SystemTime", "-")
            if time_created_node is not None
            else "-"
        )
        timestamp = parse_iso_timestamp(sys_time_str)

        computer_elem = (
            system.find("evt:Computer", NS) if system is not None else None
        )
        dst_computer = computer_elem.text if computer_elem is not None else "-"

        event_data = {}
        data_nodes = root.findall(".//evt:EventData/evt:Data", NS)
        for node in data_nodes:
            name = node.attrib.get("Name")
            if name:
                event_data[name] = node.text or "-"

        src_username = event_data.get("SubjectUserName", "-")
        dst_username = event_data.get("TargetUserName", "-")
        logon_type = event_data.get("LogonType", "-")
        src_computer = event_data.get("WorkstationName", "-")

        if src_computer in ("-", "", None):
            src_computer = event_data.get("IpAddress", "-")

        return {
            "record_id": record_id,
            "timestamp": timestamp,
            "event_id": event_id,
            "src_username": src_username,
            "username": dst_username,
            "src_host": src_computer,
            "destination_host": dst_computer,
            "logon_type": logon_type,
        }
    except Exception:
        return None


def live_stream_detection():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Không tìm thấy model tại: {MODEL_PATH}")
        return

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    threshold = bundle["threshold"]
    print(f"[SUCCESS] Model Loaded! Threshold = {threshold:.5f}\n")

    # Windows State
    attempts_5m = WindowCounter(FIVE_MINUTES)
    successes_5m = WindowCounter(FIVE_MINUTES)
    src_computers_1h = WindowDistinctCounter(ONE_HOUR)
    dst_computers_5m = WindowDistinctCounter(FIVE_MINUTES)
    dst_computers_1h = WindowDistinctCounter(ONE_HOUR)
    dst_users_1h = WindowDistinctCounter(ONE_HOUR)
    src_users_1h = WindowDistinctCounter(ONE_HOUR)
    auth_count_1h = WindowCounter(ONE_HOUR)

    seen_src_user_src_computer = set()
    seen_src_user_dst_computer = set()
    seen_src_computer_dst_computer = set()
    last_auth_by_src_user = {}
    last_auth_by_src_user_dst_computer = {}

    processed_record_ids = set()

    h_event = win32event.CreateEvent(None, 0, 0, None)
    xpath_query = "*[System[(EventID=4624 or EventID=4625)]]"

    try:
        subscription = win32evtlog.EvtSubscribe(
            "Security",
            win32evtlog.EvtSubscribeToFutureEvents,
            SignalEvent=h_event,
            Query=xpath_query,
        )
    except Exception as e:
        print(f"[ERROR] EvtSubscribe thất bại: {e}")
        return

    print("[STATUS] LISTENING FOR LIVE REAL-TIME EVENTS...\n")

    while True:
        try:
            win32event.WaitForSingleObject(h_event, 1000)
            events = win32evtlog.EvtNext(subscription, 10, Timeout=100)

            for handle in events:
                xml_str = win32evtlog.EvtRender(
                    handle, win32evtlog.EvtRenderEventXml
                )
                item = parse_xml_event(xml_str)

                if not item:
                    continue

                record_id = item["record_id"]
                if record_id in processed_record_ids:
                    continue
                if record_id is not None:
                    processed_record_ids.add(record_id)
                    if len(processed_record_ids) > 10000:
                        processed_record_ids.pop()

                event_id = item["event_id"]
                current_time = item["timestamp"]
                src_username = item["src_username"]
                dst_username = item["username"]
                src_computer = item["src_host"]
                dst_computer = item["destination_host"]
                logon_type = item["logon_type"]

                time_formatted = datetime.fromtimestamp(current_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # Expire sliding windows
                for w in (
                    attempts_5m,
                    successes_5m,
                    src_computers_1h,
                    dst_computers_5m,
                    dst_computers_1h,
                    dst_users_1h,
                    src_users_1h,
                    auth_count_1h,
                ):
                    w.expire(current_time)

                current_event_is_success = _is_success(event_id)
                src_user_src_computer = (src_username, src_computer)
                src_user_dst_computer = (src_username, dst_computer)
                src_computer_dst_computer = (src_computer, dst_computer)

                valid_src_user = _valid_key(src_username)
                valid_src_user_src_computer = _valid_key(
                    src_username, src_computer
                )
                valid_src_user_dst_computer = _valid_key(
                    src_username, dst_computer
                )
                valid_src_computer_dst_computer = _valid_key(
                    src_computer, dst_computer
                )

                # Extract features
                feature_dict = {
                    "auth_attempts_5m_per_src_user": attempts_5m.get(
                        src_username
                    ),
                    "successful_auths_5m_per_src_user": successes_5m.get(
                        src_username
                    ),
                    "unique_src_computers_1h_per_src_user": src_computers_1h.nunique(
                        src_username
                    ),
                    "unique_dst_computers_5m_per_src_user": dst_computers_5m.nunique(
                        src_username
                    ),
                    "unique_dst_computers_1h_per_src_user": dst_computers_1h.nunique(
                        src_username
                    ),
                    "unique_dst_users_1h_per_src_computer": dst_users_1h.nunique(
                        src_computer
                    ),
                    "unique_src_users_1h_per_dst_computer": src_users_1h.nunique(
                        dst_computer
                    ),
                    "prior_auth_count_1h_src_user_dst_computer": auth_count_1h.get(
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
                }

                # Update State
                attempts_5m.add(current_time, src_username)
                if current_event_is_success:
                    successes_5m.add(current_time, src_username)
                src_computers_1h.add(current_time, src_username, src_computer)
                dst_computers_5m.add(current_time, src_username, dst_computer)
                dst_computers_1h.add(current_time, src_username, dst_computer)
                dst_users_1h.add(current_time, src_computer, dst_username)
                src_users_1h.add(current_time, dst_computer, src_username)
                if valid_src_user_dst_computer:
                    auth_count_1h.add(current_time, src_user_dst_computer)

                if valid_src_user_src_computer:
                    seen_src_user_src_computer.add(src_user_src_computer)
                if valid_src_user_dst_computer:
                    seen_src_user_dst_computer.add(src_user_dst_computer)
                    last_auth_by_src_user_dst_computer[
                        src_user_dst_computer
                    ] = current_time
                if valid_src_computer_dst_computer:
                    seen_src_computer_dst_computer.add(
                        src_computer_dst_computer
                    )
                if valid_src_user:
                    last_auth_by_src_user[src_username] = current_time

                # Inference
                X_single = pd.DataFrame([feature_dict])[feature_cols].astype(
                    "float32"
                )
                score = float(model.predict_proba(X_single)[0, 1])
                label_str = "ANOMALY" if score >= threshold else "NORMAL"

                src_str = f"{src_username}@{src_computer}"
                dst_str = f"{dst_username}@{dst_computer}"

                print(
                    f"[{label_str:<7}] rec_id={record_id} | time={time_formatted} | id={event_id} | score={score:.5f} (threshold={threshold:.5f}) | {src_str} -> {dst_str}"
                )

        except Exception:
            time.sleep(0.1)


if __name__ == "__main__":
    live_stream_detection()