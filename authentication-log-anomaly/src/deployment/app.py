from __future__ import annotations

import csv
import json
import math
import platform
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
from typing import Callable, Deque, Hashable

import joblib
import pandas as pd
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

try:
    import pywintypes  # type: ignore
    import win32evtlog  # type: ignore
except ImportError:
    pywintypes = None  # type: ignore[assignment]
    win32evtlog = None

try:
    from ..models.Xgboostbaseline import FEATURE_COLUMNS
except ImportError:
    try:
        from models.Xgboostbaseline import FEATURE_COLUMNS
    except ImportError:
        FEATURE_COLUMNS = [
            "auth_attempts_5m_per_user",
            "unique_src_computers_1h_per_user",
            "unique_users_1h_per_src_computer",
            "unique_dst_computers_1h_per_user",
            "unique_dst_users_1h_per_src_computer",
            "hour_of_day",
            "is_network_logon",
        ]

LEGACY_FEATURE_COLUMNS = [
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


APP_NAME = "Authentication Log Anomaly Desktop App"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
USER_DATA_ROOT = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "AuthAnomalyDesktopApp"
MODELS_DIR = USER_DATA_ROOT / "models"
IMPORTED_MODELS_DIR = MODELS_DIR / "imported"
EXPORT_DIR = USER_DATA_ROOT / "exports"

WINDOWS_EVENT_QUERY = "*[System[(EventID=4624 or EventID=4625 or EventID=4648)]]"
EVENT_NS = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
# `log` predicates are evaluated by macOS, so this only returns genuine entries
# from the Unified Log. The broad message clauses cover macOS versions whose
# authentication components do not expose a stable subsystem name.
MACOS_AUTH_PREDICATE = (
    'subsystem == "com.apple.authd" OR process == "sshd" OR '
    'process == "loginwindow" OR eventMessage CONTAINS[c] "authentication" OR '
    'eventMessage CONTAINS[c] "login"'
)
DEFAULT_THRESHOLD = 0.70
TABLE_MAX_ROWS = 300

FIVE_MINUTES = 300
ONE_HOUR = 3_600
SECONDS_PER_DAY = 86_400


def ensure_runtime_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def relative_model_label(path: Path) -> str:
    for base in (USER_DATA_ROOT, PROJECT_ROOT, BUNDLED_ROOT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return path.name


def is_missing(value: object) -> bool:
    return value is None or value == "" or str(value).lower() in {"none", "nan", "-"}


def fmt_ts(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def parse_windows_time(system_time: str | None) -> int:
    if not system_time:
        return int(time.time())
    normalized = system_time.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return int(time.time())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def parse_log_timestamp(timestamp: object) -> int:
    """Parse the ISO-8601 timestamps emitted by Windows and macOS logs."""
    if not timestamp:
        return int(time.time())
    normalized = str(timestamp).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return int(time.time())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


@dataclass
class NormalizedEvent:
    timestamp: int
    username: str
    src_username: str
    src_host: str
    destination_host: str
    auth_orientation: str
    result: str
    event_id: int
    logon_type: str
    event_type: str

    @property
    def display_time(self) -> str:
        return fmt_ts(self.timestamp)


@dataclass
class DetectionResult:
    timestamp: int
    username: str
    src_host: str
    destination_host: str
    result: str
    model_name: str
    anomaly_score: float
    raw_score: float
    is_alert: bool
    reason: str
    features: dict[str, float]

    @property
    def display_time(self) -> str:
        return fmt_ts(self.timestamp)


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
        if is_missing(key):
            return
        self.events.append((current_time, key))
        self.counts[key] += 1

    def get(self, key: Hashable) -> int:
        if is_missing(key):
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
        if is_missing(group_key) or is_missing(value_key):
            return
        self.events.append((current_time, group_key, value_key))
        self.counts_by_group[group_key][value_key] += 1

    def nunique(self, group_key: Hashable) -> int:
        if is_missing(group_key):
            return 0
        return len(self.counts_by_group.get(group_key, ()))


class RealtimeFeatureEngine:
    def __init__(self) -> None:
        self.attempts_5m_by_user = WindowCounter(FIVE_MINUTES)
        self.failed_attempts_5m_by_user = WindowCounter(FIVE_MINUTES)
        self.failed_attempts_5m_by_src_host = WindowCounter(FIVE_MINUTES)
        self.attempts_1h_by_user = WindowCounter(ONE_HOUR)
        self.failed_attempts_1h_by_user = WindowCounter(ONE_HOUR)
        self.src_computers_1h_by_user = WindowDistinctCounter(ONE_HOUR)
        self.users_1h_by_src_computer = WindowDistinctCounter(ONE_HOUR)
        self.dst_computers_1h_by_user = WindowDistinctCounter(ONE_HOUR)
        self.dst_users_1h_by_src_computer = WindowDistinctCounter(ONE_HOUR)
        self.last_time: int | None = None

    def transform(self, event: NormalizedEvent) -> dict[str, float]:
        current_time = int(event.timestamp)
        if self.last_time is not None and current_time < self.last_time:
            current_time = self.last_time
        self.last_time = current_time

        for window in (
            self.attempts_5m_by_user,
            self.failed_attempts_5m_by_user,
            self.failed_attempts_5m_by_src_host,
            self.attempts_1h_by_user,
            self.failed_attempts_1h_by_user,
            self.src_computers_1h_by_user,
            self.users_1h_by_src_computer,
            self.dst_computers_1h_by_user,
            self.dst_users_1h_by_src_computer,
        ):
            window.expire(current_time)

        hour_of_day = float((current_time % SECONDS_PER_DAY) // 3_600)
        day_of_week = float(datetime.fromtimestamp(current_time).weekday())
        attempts_1h = self.attempts_1h_by_user.get(event.src_username)
        failed_1h = self.failed_attempts_1h_by_user.get(event.src_username)
        failure_rate_1h = float(failed_1h / attempts_1h) if attempts_1h else 0.0

        features = {
            "auth_attempts_5m_per_user": float(self.attempts_5m_by_user.get(event.src_username)),
            "unique_src_computers_1h_per_user": float(self.src_computers_1h_by_user.nunique(event.src_username)),
            "unique_users_1h_per_src_computer": float(self.users_1h_by_src_computer.nunique(event.src_host)),
            "unique_dst_computers_1h_per_user": float(self.dst_computers_1h_by_user.nunique(event.src_username)),
            "unique_dst_users_1h_per_src_computer": float(self.dst_users_1h_by_src_computer.nunique(event.src_host)),
            "hour_of_day": hour_of_day,
            "is_network_logon": float(int(str(event.logon_type) in {"Network", "3", "8", "10"})),
            "current_result_is_fail": float(int(event.result == "Fail")),
            # Compatibility with older XGBoost models trained on split CSVs.
            "failed_logins_5m_user": float(self.failed_attempts_5m_by_user.get(event.src_username)),
            "unique_src_ip_1h": float(self.src_computers_1h_by_user.nunique(event.src_username)),
            "failed_logins_5m_ip": float(self.failed_attempts_5m_by_src_host.get(event.src_host)),
            "failure_rate_1h": failure_rate_1h,
            "unique_users_1h_per_ip": float(self.users_1h_by_src_computer.nunique(event.src_host)),
            "day_of_week": day_of_week,
            "is_business_hours": float(int(9 <= hour_of_day < 17)),
        }

        self.attempts_5m_by_user.add(current_time, event.src_username)
        self.attempts_1h_by_user.add(current_time, event.src_username)
        if event.result == "Fail":
            self.failed_attempts_5m_by_user.add(current_time, event.src_username)
            self.failed_attempts_5m_by_src_host.add(current_time, event.src_host)
            self.failed_attempts_1h_by_user.add(current_time, event.src_username)
        self.src_computers_1h_by_user.add(current_time, event.src_username, event.src_host)
        self.users_1h_by_src_computer.add(current_time, event.src_host, event.username)
        self.dst_computers_1h_by_user.add(current_time, event.src_username, event.destination_host)
        self.dst_users_1h_by_src_computer.add(current_time, event.src_host, event.username)
        return features


class InferenceEngine:
    def __init__(self) -> None:
        self.model = None
        self.model_path: Path | None = None
        self.model_name = "Heuristic fallback"
        self.feature_columns = list(FEATURE_COLUMNS)
        self.threshold = DEFAULT_THRESHOLD

    def set_threshold(self, value: float) -> None:
        self.threshold = max(0.05, min(0.99, float(value)))

    def scan_models(self) -> dict[str, Path]:
        ensure_runtime_dirs()
        result: dict[str, Path] = {}
        search_roots: list[Path] = []
        for root in (MODELS_DIR, PROJECT_ROOT / "models", BUNDLED_ROOT / "models"):
            if root.exists() and root not in search_roots:
                search_roots.append(root)

        for root in search_roots:
            for path in sorted(root.rglob("*.pkl")):
                result[relative_model_label(path)] = path
        return result

    def load_model(self, model_path: Path) -> None:
        self.model = joblib.load(model_path)
        self.model_path = model_path
        self.model_name = model_path.stem
        self.feature_columns = self._detect_model_features(self.model)

    def unload_model(self) -> None:
        self.model = None
        self.model_path = None
        self.model_name = "Heuristic fallback"
        self.feature_columns = list(FEATURE_COLUMNS)

    def import_model(self, source_path: Path) -> Path:
        ensure_runtime_dirs()
        target_path = IMPORTED_MODELS_DIR / source_path.name
        if target_path.exists():
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = IMPORTED_MODELS_DIR / f"{source_path.stem}_{suffix}{source_path.suffix}"
        shutil.copy2(source_path, target_path)
        return target_path

    def delete_model(self, model_path: Path) -> None:
        if model_path.exists():
            model_path.unlink()
        if self.model_path and self.model_path.resolve() == model_path.resolve():
            self.unload_model()

    def score(self, event: NormalizedEvent, features: dict[str, float]) -> DetectionResult:
        prepared = {column: float(features.get(column, 0.0)) for column in self.feature_columns}
        if self.model is None:
            anomaly_score = self._heuristic_score(event, features)
            raw_score = anomaly_score
            is_alert = anomaly_score >= self.threshold
        else:
            anomaly_score, raw_score, is_alert = self._model_score(prepared)

        return DetectionResult(
            timestamp=event.timestamp,
            username=event.username,
            src_host=event.src_host,
            destination_host=event.destination_host,
            result=event.result,
            model_name=self.model_name,
            anomaly_score=anomaly_score,
            raw_score=raw_score,
            is_alert=is_alert,
            reason=self._build_reason(event, features, anomaly_score, is_alert),
            features=prepared,
        )

    def _model_score(self, prepared: dict[str, float]) -> tuple[float, float, bool]:
        if self.model is None:
            return 0.0, 0.0, False

        frame = pd.DataFrame([prepared], columns=self.feature_columns)
        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(frame)[0][1])
            return proba, proba, proba >= self.threshold

        if hasattr(self.model, "decision_function"):
            raw = float(-self.model.decision_function(frame)[0])
            scaled = 1.0 - math.exp(-max(raw, 0.0))
            prediction = self.model.predict(frame)[0] if hasattr(self.model, "predict") else 0
            is_alert = bool(prediction == -1 or scaled >= self.threshold)
            return max(0.0, min(1.0, scaled)), raw, is_alert

        if hasattr(self.model, "predict"):
            prediction = float(self.model.predict(frame)[0])
            return prediction, prediction, prediction >= self.threshold

        raise TypeError("Model không hỗ trợ `predict_proba`, `decision_function` hoặc `predict`.")

    def _detect_model_features(self, model) -> list[str]:
        """Use the loaded model's expected feature order when available."""
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            return [str(column) for column in feature_names]

        if hasattr(model, "get_booster"):
            booster_features = model.get_booster().feature_names
            if booster_features:
                return [str(column) for column in booster_features]

        feature_count = getattr(model, "n_features_in_", None)
        if feature_count == len(LEGACY_FEATURE_COLUMNS):
            return list(LEGACY_FEATURE_COLUMNS)
        return list(FEATURE_COLUMNS)

    def _heuristic_score(self, event: NormalizedEvent, features: dict[str, float]) -> float:
        attempts = min(features["auth_attempts_5m_per_user"] / 8.0, 1.0)
        unique_src = min(features["unique_src_computers_1h_per_user"] / 4.0, 1.0)
        unique_dst = min(features["unique_dst_computers_1h_per_user"] / 4.0, 1.0)
        noisy_src = min(features["unique_users_1h_per_src_computer"] / 6.0, 1.0)
        fail_bonus = 0.75 if event.result == "Fail" else 0.0
        network_bonus = 0.08 if features.get("is_network_logon", 0.0) else 0.0
        score = (attempts * 0.38) + (unique_src * 0.20) + (unique_dst * 0.10) + (noisy_src * 0.12)
        score += fail_bonus + network_bonus
        return max(0.0, min(1.0, score))

    def _build_reason(self, event: NormalizedEvent, features: dict[str, float], score: float, is_alert: bool) -> str:
        if is_alert:
            return f"Phát hiện tấn công | score={score:.2f}"
        return f"Bình thường | score={score:.2f}"


class BaseCollector(threading.Thread):
    def __init__(
        self,
        event_callback: Callable[[NormalizedEvent], None],
        status_callback: Callable[[str], None],
        error_callback: Callable[[str], None],
    ) -> None:
        super().__init__(daemon=True)
        self.event_callback = event_callback
        self.status_callback = status_callback
        self.error_callback = error_callback
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def emit_event(self, event: NormalizedEvent) -> None:
        self.event_callback(event)

    def emit_status(self, message: str) -> None:
        self.status_callback(message)

    def emit_error(self, message: str) -> None:
        self.error_callback(message)


class WindowsEventCollector(BaseCollector):
    """Collect authentication events from Windows Security Event Log.

    Uses the modern EvtLog API via pywin32.  Key calls:
      - EvtQuery(ChannelPath, Flags, Query)       – one-shot historical query
      - EvtSubscribe(Channel, Flags, …)            – push or pull subscription
      - EvtNext(ResultSet, Count)                   – iterate query results
      - EvtRender(Event, EvtRenderEventXml)         – render event as XML
    """

    def run(self) -> None:
        if win32evtlog is None:
            self.emit_error(
                "Chưa cài `pywin32`, không thể đọc Windows Security Log.\n"
                "Hãy chạy: pip install pywin32"
            )
            return

        try:
            self.emit_status("Đang kết nối Windows Security Log…")
            self._read_recent_events()

            # --- Push-mode subscription (callback) ---
            # Signature: EvtSubscribe(ChannelPath, Flags,
            #     SignalEvent=None, Callback=fn, Context=None,
            #     Query=str, Session=None, Bookmark=None)
            def _on_event(action, context, event_handle):
                """Callback invoked by Windows for each new event."""
                if action == win32evtlog.EvtSubscribeActionDeliver:
                    try:
                        xml_str = win32evtlog.EvtRender(
                            event_handle, win32evtlog.EvtRenderEventXml
                        )
                        parsed = self._parse_event_xml(xml_str)
                        if parsed:
                            self.emit_event(parsed)
                    except Exception as exc:
                        self.emit_status(f"Bỏ qua event lỗi render: {exc}")
                elif action == win32evtlog.EvtSubscribeActionError:
                    self.emit_status(f"Lỗi subscription: error code {event_handle}")

            _subscription_handle = win32evtlog.EvtSubscribe(
                "Security",
                win32evtlog.EvtSubscribeToFutureEvents,
                Callback=_on_event,
                Query=WINDOWS_EVENT_QUERY,
            )
            self.emit_status("Đã bắt đầu lắng nghe log thật từ Windows.")

            # Keep the thread alive until stop is requested.
            # The callback fires on a separate internal thread managed by Windows.
            while not self.stop_event.is_set():
                self.stop_event.wait(timeout=2.0)

            self.emit_status("Đã dừng lắng nghe Windows Security Log.")
            # _subscription_handle goes out of scope → auto-closed by pywin32

        except Exception as exc:
            self._emit_detailed_error(exc)

    def _read_recent_events(self) -> None:
        """Load the most recent 40 Security events for immediate display."""
        try:
            handle = win32evtlog.EvtQuery(
                "Security",
                win32evtlog.EvtQueryReverseDirection,
                WINDOWS_EVENT_QUERY,
            )
        except Exception as exc:
            if self._is_access_denied(exc):
                raise PermissionError(
                    "Access Denied khi đọc Security log. "
                    "Vui lòng chạy lại với quyền Administrator."
                ) from exc
            raise

        try:
            recent_batch = win32evtlog.EvtNext(handle, 40)
        except StopIteration:
            recent_batch = []

        count = 0
        for item in reversed(recent_batch or []):
            try:
                xml_str = win32evtlog.EvtRender(item, win32evtlog.EvtRenderEventXml)
                parsed = self._parse_event_xml(xml_str)
                if parsed:
                    self.emit_event(parsed)
                    count += 1
            except Exception:
                continue  # skip unreadable events
        if count:
            self.emit_status(f"Đã tải {count} sự kiện gần đây từ Security log.")

    @staticmethod
    def _is_access_denied(exc: Exception) -> bool:
        """Check if an exception is a Windows Access Denied error."""
        if pywintypes is not None and isinstance(exc, pywintypes.error):
            return exc.winerror == 5  # ERROR_ACCESS_DENIED
        err_msg = str(exc).lower()
        return "access" in err_msg and "denied" in err_msg

    def _emit_detailed_error(self, exc: Exception) -> None:
        """Emit a user-friendly error message for common failures."""
        if self._is_access_denied(exc):
            self.emit_error(
                "Không có quyền đọc Windows Security Log.\n\n"
                "Hãy chạy lại ứng dụng với quyền Administrator:\n"
                "  • Click chuột phải → 'Run as administrator'\n"
                "  • Hoặc chạy terminal (PowerShell/CMD) với quyền Admin\n\n"
                f"Chi tiết lỗi: {exc}"
            )
        elif isinstance(exc, PermissionError):
            self.emit_error(str(exc))
        else:
            self.emit_error(f"Không thể đọc Windows Event Log: {exc}")

    def _parse_event_xml(self, xml_string: str) -> NormalizedEvent | None:
        root = ET.fromstring(xml_string)
        event_id_text = root.findtext(".//e:System/e:EventID", namespaces=EVENT_NS) or "0"
        event_id = int(event_id_text)
        created = root.find(".//e:System/e:TimeCreated", EVENT_NS)
        timestamp = parse_windows_time(created.attrib.get("SystemTime") if created is not None else None)

        event_data = {
            node.attrib.get("Name", ""): (node.text or "")
            for node in root.findall(".//e:EventData/e:Data", EVENT_NS)
        }
        username = (
            event_data.get("TargetUserName")
            or event_data.get("SubjectUserName")
            or event_data.get("TargetOutboundUserName")
            or "unknown"
        )
        src_username = event_data.get("SubjectUserName") or username
        src_host = (
            event_data.get("WorkstationName")
            or event_data.get("IpAddress")
            or event_data.get("ProcessName")
            or "unknown"
        )
        destination_host = event_data.get("TargetServerName") or event_data.get("TargetDomainName") or "local-machine"
        logon_type = event_data.get("LogonType") or "Unknown"
        result = "Fail" if event_id == 4625 else "Success"
        event_type = "Failure" if event_id == 4625 else "Success"

        return NormalizedEvent(
            timestamp=timestamp,
            username=username,
            src_username=src_username,
            src_host=src_host,
            destination_host=destination_host,
            auth_orientation="LogOn",
            result=result,
            event_id=event_id,
            logon_type=logon_type,
            event_type=event_type,
        )


class MacOSUnifiedLogCollector(BaseCollector):
    """Collect real authentication-related entries from the macOS Unified Log.

    macOS does not have a Security Event Log equivalent with a stable schema.
    This collector therefore keeps the original Unified Log message and derives
    common fields conservatively. It never fabricates events: each row comes
    from `/usr/bin/log show` or `/usr/bin/log stream`.
    """

    HISTORY_LIMIT = 200

    def __init__(
        self,
        event_callback: Callable[[NormalizedEvent], None],
        status_callback: Callable[[str], None],
        error_callback: Callable[[str], None],
    ) -> None:
        super().__init__(event_callback, status_callback, error_callback)
        self.process: subprocess.Popen[str] | None = None
        self.hostname = platform.node() or "localhost"

    def stop(self) -> None:
        super().stop()
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def run(self) -> None:
        if platform.system() != "Darwin":
            self.emit_error("macOS Unified Log chỉ khả dụng trên macOS.")
            return

        try:
            self.emit_status("Đang đọc authentication log thật từ macOS Unified Log…")
            loaded = self._read_recent_events()
            self.emit_status(f"Đã tải {loaded} sự kiện xác thực gần đây từ macOS.")
            self._stream_new_events()
        except PermissionError as exc:
            self.emit_error(
                "Không có quyền đọc một phần macOS Unified Log. "
                "Hãy mở Terminal/app với quyền phù hợp rồi thử lại.\n\n"
                f"Chi tiết: {exc}"
            )
        except FileNotFoundError:
            self.emit_error("Không tìm thấy `/usr/bin/log`; macOS Unified Log không khả dụng.")
        except Exception as exc:
            self.emit_error(f"Không thể đọc macOS Unified Log: {exc}")

    def _read_recent_events(self) -> int:
        command = [
            "/usr/bin/log",
            "show",
            "--style",
            "json",
            "--last",
            "1h",
            "--predicate",
            MACOS_AUTH_PREDICATE,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            self._raise_for_log_error(completed.stderr)

        events = [event for event in self._decode_json_lines(completed.stdout) if event]
        events.sort(key=lambda event: event.timestamp)
        for event in events[-self.HISTORY_LIMIT :]:
            if self.stop_event.is_set():
                break
            self.emit_event(event)
        return min(len(events), self.HISTORY_LIMIT)

    def _stream_new_events(self) -> None:
        command = [
            "/usr/bin/log",
            "stream",
            "--style",
            "json",
            "--predicate",
            MACOS_AUTH_PREDICATE,
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.emit_status("Đã bắt đầu lắng nghe log thật từ macOS.")
        assert self.process.stdout is not None
        while not self.stop_event.is_set():
            line = self.process.stdout.readline()
            if not line:
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read() if self.process.stderr else ""
                    self._raise_for_log_error(stderr)
                continue
            for event in self._decode_json_lines(line):
                if event:
                    self.emit_event(event)
        self.emit_status("Đã dừng lắng nghe macOS Unified Log.")

    def _raise_for_log_error(self, stderr: str) -> None:
        message = stderr.strip() or "Lệnh macOS `log` thất bại."
        if "not permitted" in message.lower() or "operation not permitted" in message.lower():
            raise PermissionError(message)
        raise RuntimeError(message)

    def _decode_json_lines(self, output: str) -> list[NormalizedEvent]:
        results: list[NormalizedEvent] = []
        for line in output.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in {"[", "]"}:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if isinstance(item, dict):
                    results.append(self._normalize_log_record(item))
        return results

    def _normalize_log_record(self, record: dict[str, object]) -> NormalizedEvent:
        message = str(record.get("eventMessage") or record.get("message") or "")
        process = str(record.get("process") or record.get("processImagePath") or "macOS")
        lowered = message.lower()
        failed = any(token in lowered for token in ("fail", "denied", "invalid password", "not authorized"))
        username_match = re.search(
            r"(?:user(?:name)?|account|principal)\s*[=:]\s*['\"]?([^\s,'\"]+)",
            message,
            flags=re.IGNORECASE,
        )
        username = username_match.group(1) if username_match else "unknown"
        return NormalizedEvent(
            timestamp=parse_log_timestamp(record.get("timestamp") or record.get("time")),
            username=username,
            src_username=username,
            src_host=self.hostname,
            destination_host=self.hostname,
            auth_orientation="LogOn",
            result="Fail" if failed else "Success",
            event_id=0,
            logon_type="macOS Unified Log",
            event_type=f"macOS: {process}",
        )


class StatCard(QFrame):
    def __init__(self, title: str, initial_value: str) -> None:
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 12px; color: #4b5563;")
        self.value_label = QLabel(initial_value)
        self.value_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class AuthAnomalyDesktopApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_runtime_dirs()
        self.setWindowTitle("Authentication Log Anomaly Desktop App")
        self.resize(1440, 900)

        self.runtime_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.feature_engine = RealtimeFeatureEngine()
        self.detector = InferenceEngine()
        self.collector: BaseCollector | None = None
        self.model_map: dict[str, Path] = {}
        self.recent_results: Deque[DetectionResult] = deque(maxlen=500)
        self.alert_results: Deque[DetectionResult] = deque(maxlen=500)

        self._build_ui()
        self.refresh_models(initial=True)

        self.queue_timer = QTimer(self)
        self.queue_timer.timeout.connect(self.drain_runtime_queue)
        self.queue_timer.start(300)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        export_action = QAction("Export alerts", self)
        export_action.triggered.connect(self.export_alerts)
        toolbar.addAction(export_action)

        controls = QGroupBox("Điều khiển")
        controls_layout = QGridLayout(controls)

        self.source_combo = QComboBox()
        system = platform.system()
        self.source_combo.addItems(["Windows Security Log", "macOS Unified Log"])
        self.source_combo.setCurrentText(
            "Windows Security Log" if system == "Windows" else "macOS Unified Log"
        )

        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.apply_selected_model)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.10, 0.95)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(DEFAULT_THRESHOLD)
        self.threshold_spin.valueChanged.connect(self.on_threshold_changed)

        controls_layout.addWidget(QLabel("Nguồn log"), 0, 0)
        controls_layout.addWidget(self.source_combo, 0, 1)
        controls_layout.addWidget(QLabel("Model"), 0, 2)
        controls_layout.addWidget(self.model_combo, 0, 3)
        controls_layout.addWidget(QLabel("Ngưỡng cảnh báo"), 0, 4)
        controls_layout.addWidget(self.threshold_spin, 0, 5)
        controls_layout.setColumnStretch(3, 1)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Bắt đầu")
        self.stop_button = QPushButton("Dừng")
        self.import_button = QPushButton("Import model")
        self.delete_button = QPushButton("Xóa model")
        self.refresh_button = QPushButton("Làm mới model")
        self.export_button = QPushButton("Export alerts")

        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.import_button.clicked.connect(self.import_model)
        self.delete_button.clicked.connect(self.delete_selected_model)
        self.refresh_button.clicked.connect(self.refresh_models)
        self.export_button.clicked.connect(self.export_alerts)

        for button in (
            self.start_button,
            self.stop_button,
            self.import_button,
            self.delete_button,
            self.refresh_button,
            self.export_button,
        ):
            button_row.addWidget(button)
        button_row.addStretch(1)
        controls_layout.addLayout(button_row, 1, 0, 1, 6)

        root.addWidget(controls)

        stats_layout = QHBoxLayout()
        self.events_card = StatCard("Sự kiện đã xử lý", "0")
        self.alerts_card = StatCard("Cảnh báo", "0")
        self.model_card = StatCard("Model đang dùng", "Heuristic fallback")
        self.last_alert_card = StatCard("Cảnh báo gần nhất", "Chưa có")
        for card in (self.events_card, self.alerts_card, self.model_card, self.last_alert_card):
            stats_layout.addWidget(card)
        root.addLayout(stats_layout)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlainText(
            "Ứng dụng sẵn sàng.\n"
            "- App chỉ đọc log xác thực thật; không có chế độ mô phỏng.\n"
            "- Windows: chọn `Windows Security Log`; macOS: chọn `macOS Unified Log`.\n"
            "- Model `.pkl` có sẵn được tự chọn; mọi event sẽ được feature engine và model xử lý.\n"
        )
        overview_layout.addWidget(self.summary_text)
        self.tabs.addTab(overview_tab, "Tổng quan")

        self.events_table = self._build_table(
            ["Thời gian", "EventID", "User", "Nguồn", "Đích", "Kết quả", "Score", "Model"]
        )
        self.tabs.addTab(self.events_table, "Sự kiện")

        self.alerts_table = self._build_table(
            ["Thời gian", "User", "Nguồn", "Score", "Lý do", "Model"]
        )
        self.tabs.addTab(self.alerts_table, "Cảnh báo")

        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        self.logs_text = QPlainTextEdit()
        self.logs_text.setReadOnly(True)
        logs_layout.addWidget(self.logs_text)
        self.tabs.addTab(logs_tab, "Nhật ký")

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Sẵn sàng.")

    def _build_table(self, headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.verticalHeader().setVisible(False)
        return table

    def show_info(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def log_message(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.logs_text.appendPlainText(f"[{stamp}] {message}")
        self.statusBar().showMessage(message)

    def enqueue_event(self, event: NormalizedEvent) -> None:
        self.runtime_queue.put(("event", event))

    def enqueue_status(self, message: str) -> None:
        self.runtime_queue.put(("status", message))

    def enqueue_error(self, message: str) -> None:
        self.runtime_queue.put(("error", message))

    def refresh_models(self, initial: bool = False) -> None:
        current = self.model_combo.currentText() or "Heuristic fallback"
        self.model_map = self.detector.scan_models()
        model_names = ["Heuristic fallback", *self.model_map.keys()]
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(model_names)
        if initial and len(model_names) > 1:
            # Prefer a trained model when one ships with the application, so
            # collected events are scored by model inference from first launch.
            self.model_combo.setCurrentText(model_names[1])
        elif current in model_names:
            self.model_combo.setCurrentText(current)
        else:
            self.model_combo.setCurrentText("Heuristic fallback")
        self.model_combo.blockSignals(False)
        self.apply_selected_model(log_change=not initial)

    def apply_selected_model(self, _value: str | None = None, log_change: bool = True) -> None:
        selected = self.model_combo.currentText() or "Heuristic fallback"
        try:
            if selected == "Heuristic fallback":
                self.detector.unload_model()
            else:
                self.detector.load_model(self.model_map[selected])
            self.detector.set_threshold(float(self.threshold_spin.value()))
            self.model_card.set_value(self.detector.model_name)
            if log_change:
                self.log_message(f"Đã chọn model: {selected}")
        except Exception as exc:
            self.detector.unload_model()
            self.model_combo.setCurrentText("Heuristic fallback")
            self.model_card.set_value(self.detector.model_name)
            self.show_error("Lỗi model", f"Không thể nạp model đã chọn.\n\n{exc}")
            self.log_message(f"Nạp model thất bại: {exc}")

    def on_threshold_changed(self, value: float) -> None:
        self.detector.set_threshold(value)
        self.statusBar().showMessage(f"Ngưỡng cảnh báo: {value:.2f}")

    def import_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file model",
            str(USER_DATA_ROOT),
            "Pickle model (*.pkl);;All files (*.*)",
        )
        if not file_path:
            return
        try:
            target = self.detector.import_model(Path(file_path))
            self.refresh_models()
            self.model_combo.setCurrentText(relative_model_label(target))
            self.apply_selected_model()
            self.log_message(f"Đã import model vào {target}")
        except Exception as exc:
            self.show_error("Import model thất bại", str(exc))
            self.log_message(f"Import model lỗi: {exc}")

    def delete_selected_model(self) -> None:
        selected = self.model_combo.currentText()
        if not selected or selected == "Heuristic fallback":
            self.show_info("Xóa model", "Hãy chọn một model `.pkl` để xóa.")
            return
        model_path = self.model_map.get(selected)
        if model_path is None:
            self.show_error("Xóa model", "Không tìm thấy model đã chọn.")
            return
        reply = QMessageBox.question(
            self,
            "Xóa model",
            f"Bạn có chắc muốn xóa `{selected}` không?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            self.detector.delete_model(model_path)
            self.refresh_models()
            self.log_message(f"Đã xóa model: {selected}")
        except Exception as exc:
            self.show_error("Xóa model thất bại", str(exc))
            self.log_message(f"Xóa model lỗi: {exc}")

    def start_monitoring(self) -> None:
        if self.collector and self.collector.is_alive():
            self.log_message("Monitoring đang chạy.")
            return
        source = self.source_combo.currentText()
        system = platform.system()
        if source == "Windows Security Log" and system != "Windows":
            self.show_error(
                "Nguồn log không khả dụng",
                "Windows Security Log chỉ có thể được đọc khi ứng dụng chạy trên Windows.",
            )
            return
        if source == "macOS Unified Log" and system != "Darwin":
            self.show_error(
                "Nguồn log không khả dụng",
                "macOS Unified Log chỉ có thể được đọc khi ứng dụng chạy trên macOS.",
            )
            return
        self.detector.set_threshold(float(self.threshold_spin.value()))
        self.feature_engine = RealtimeFeatureEngine()
        self.collector = self._create_collector()
        self.collector.start()
        self.log_message("Đã khởi động monitoring.")

    def _create_collector(self) -> BaseCollector:
        source = self.source_combo.currentText()
        collectors: dict[str, type[BaseCollector]] = {
            "Windows Security Log": WindowsEventCollector,
            "macOS Unified Log": MacOSUnifiedLogCollector,
        }
        collector_cls = collectors.get(source)
        if collector_cls is None:
            raise ValueError(f"Nguồn log không được hỗ trợ: {source}")
        return collector_cls(self.enqueue_event, self.enqueue_status, self.enqueue_error)

    def stop_monitoring(self) -> None:
        if self.collector:
            self.collector.stop()
            self.collector = None
        self.log_message("Đã gửi tín hiệu dừng monitoring.")

    def export_alerts(self) -> None:
        if not self.alert_results:
            self.show_info("Export alerts", "Chưa có cảnh báo nào để export.")
            return
        export_path = EXPORT_DIR / f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with export_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "time",
                    "user",
                    "src_host",
                    "destination_host",
                    "result",
                    "model_name",
                    "anomaly_score",
                    "raw_score",
                    "reason",
                ],
            )
            writer.writeheader()
            for item in self.alert_results:
                writer.writerow(
                    {
                        "time": item.display_time,
                        "user": item.username,
                        "src_host": item.src_host,
                        "destination_host": item.destination_host,
                        "result": item.result,
                        "model_name": item.model_name,
                        "anomaly_score": f"{item.anomaly_score:.4f}",
                        "raw_score": f"{item.raw_score:.4f}",
                        "reason": item.reason,
                    }
                )
        self.log_message(f"Đã export cảnh báo ra {export_path}")
        self.show_info("Export alerts", f"Đã lưu file:\n{export_path}")

    def drain_runtime_queue(self) -> None:
        while True:
            try:
                message_type, payload = self.runtime_queue.get_nowait()
            except queue.Empty:
                break

            if message_type == "event":
                self.handle_event(payload)  # type: ignore[arg-type]
            elif message_type == "status":
                self.log_message(str(payload))
            elif message_type == "error":
                self.log_message(str(payload))
                self.show_error("Runtime error", str(payload))

    def handle_event(self, event: NormalizedEvent) -> None:
        try:
            features = self.feature_engine.transform(event)
            result = self.detector.score(event, features)
            self.recent_results.appendleft(result)
            if result.is_alert:
                self.alert_results.appendleft(result)
                self.last_alert_card.set_value(f"{result.username} | {result.anomaly_score:.2f}")

            self.events_card.set_value(str(len(self.recent_results)))
            self.alerts_card.set_value(str(len(self.alert_results)))
            self.model_card.set_value(self.detector.model_name)

            self._append_row(
                self.events_table,
                [
                    result.display_time,
                    str(event.event_id),
                    result.username,
                    result.src_host,
                    result.destination_host,
                    result.result,
                    f"{result.anomaly_score:.2f}",
                    result.model_name,
                ],
                prepend=True,
            )
            if result.is_alert:
                self._append_row(
                    self.alerts_table,
                    [
                        result.display_time,
                        result.username,
                        result.src_host,
                        f"{result.anomaly_score:.2f}",
                        result.reason,
                        result.model_name,
                    ],
                    prepend=True,
                )
            self.refresh_summary(result)
        except Exception as exc:
            details = traceback.format_exc(limit=3)
            self.log_message(f"Lỗi xử lý event: {exc}\n{details}")

    def _append_row(self, table: QTableWidget, values: list[str], prepend: bool = False) -> None:
        row = 0 if prepend else table.rowCount()
        table.insertRow(row)
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            table.setItem(row, column, item)
        while table.rowCount() > TABLE_MAX_ROWS:
            table.removeRow(table.rowCount() - 1)

    def refresh_summary(self, latest: DetectionResult) -> None:
        lines = [
            f"Model hiện tại: {self.detector.model_name}",
            f"Score mới nhất: {latest.anomaly_score:.2f}",
            f"User: {latest.username}",
            f"Nguồn: {latest.src_host}",
            f"Đích: {latest.destination_host}",
            f"Kết quả: {latest.result}",
            f"Lý do: {latest.reason}",
            "",
            "Top 5 cảnh báo gần nhất:",
        ]
        if self.alert_results:
            for idx, item in enumerate(list(self.alert_results)[:5], start=1):
                lines.append(
                    f"{idx}. {item.display_time} | {item.username} | {item.anomaly_score:.2f} | {item.reason}"
                )
        else:
            lines.append("Chưa có cảnh báo.")
        self.summary_text.setPlainText("\n".join(lines))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.collector:
            self.collector.stop()
        super().closeEvent(event)


def run() -> None:
    ensure_runtime_dirs()
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    window = AuthAnomalyDesktopApp()
    window.show()
    if owns_app:
        app.exec()


if __name__ == "__main__":
    run()
