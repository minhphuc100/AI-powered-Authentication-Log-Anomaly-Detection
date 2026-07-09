from pathlib import Path
from typing import Iterable

import pandas as pd


RAW_PATH = Path("data/raw/auth.txt")
REDTEAM_PATH = Path("data/raw/redteam.txt")
PROCESSED_PATH = Path("data/processed/parsed_auth_logs.csv")

AUTH_COLUMNS = [
    "time",
    "src_user",
    "dst_user",
    "src_computer",
    "dst_computer",
    "auth_type",
    "logon_type",
    "auth_orientation",
    "result",
]

REDTEAM_COLUMNS = ["time", "src_user", "src_computer", "dst_computer"]

UNKNOWN_MARKERS = {"?", "-", ""}
CHUNK_SIZE = 1_000_000


def _display_path(path: Path) -> str:
    return str(path).encode("ascii", errors="backslashreplace").decode("ascii")


def _detect_encoding(path: Path) -> str:
    with path.open("rb") as handle:
        bom = handle.read(4)
    if bom.startswith(b"\xff\xfe") or bom.startswith(b"\xfe\xff"):
        return "utf-16"
    if bom.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = AUTH_COLUMNS

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].astype("int64")

    text_cols = [c for c in AUTH_COLUMNS if c != "time"]
    for col in text_cols:
        df[col] = _normalize_text(df[col])

    df = df.replace({marker: pd.NA for marker in UNKNOWN_MARKERS})

    df["is_success"] = (df["result"] == "Success").astype("int8")
    df["is_fail"] = (df["result"] == "Fail").astype("int8")
    df["is_logon"] = (df["auth_orientation"] == "LogOn").astype("int8")
    df["is_logoff"] = (df["auth_orientation"] == "LogOff").astype("int8")
    df["is_tgs"] = (df["auth_orientation"] == "TGS").astype("int8")
    df["is_machine_account"] = (
        df["src_user"].fillna("").str.contains(r"\$@", regex=True)
    ).astype("int8")
    df["event_type"] = df["result"].map({"Success": "Success", "Fail": "Failure"}).fillna("Unknown")
    df["event_id"] = df["result"].map({"Success": 4624, "Fail": 4625}).fillna(0).astype("int16")

    return df


def _to_team_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "time": "timestamp",
            "src_user": "src_username",
            "dst_user": "username",
            "src_computer": "src_host",
            "dst_computer": "destination_host",
        }
    )

    columns = [
        "timestamp",
        "username",
        "src_username",
        "src_host",
        "destination_host",
        "event_type",
        "event_id",
        "auth_type",
        "logon_type",
        "auth_orientation",
        "result",
        "is_success",
        "is_fail",
        "is_logon",
        "is_logoff",
        "is_tgs",
        "is_machine_account",
        "label",
    ]
    return df[columns]


def load_redteam(path: Path = REDTEAM_PATH) -> pd.DataFrame:
    if not path.exists():
        print(f"[parser] Redteam labels not found at {_display_path(path)}; all labels will be 0")
        return pd.DataFrame(columns=REDTEAM_COLUMNS + ["label"])

    redteam = pd.read_csv(path, header=None, names=REDTEAM_COLUMNS, encoding=_detect_encoding(path))
    redteam["time"] = pd.to_numeric(redteam["time"], errors="coerce")
    redteam = redteam.dropna(subset=["time"])
    redteam["time"] = redteam["time"].astype("int64")

    for col in REDTEAM_COLUMNS:
        if col != "time":
            redteam[col] = _normalize_text(redteam[col])

    redteam = redteam.drop_duplicates()
    redteam["label"] = 1
    print(f"[parser] Loaded {len(redteam):,} redteam labels from {_display_path(path)}")
    return redteam


def iter_auth_chunks(path: Path, chunksize: int = CHUNK_SIZE) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        path,
        header=None,
        names=AUTH_COLUMNS,
        chunksize=chunksize,
        dtype="string",
        encoding=_detect_encoding(path),
    )


def parse_file(
    raw_path: Path = RAW_PATH,
    redteam_path: Path = REDTEAM_PATH,
    output_path: Path = PROCESSED_PATH,
    chunksize: int = CHUNK_SIZE,
) -> pd.DataFrame | None:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Auth log not found: {_display_path(raw_path)}. Put the LANL auth log there or pass raw_path."
        )

    redteam = load_redteam(redteam_path)
    label_keys = REDTEAM_COLUMNS
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_header = False
    total_rows = 0
    total_labels = 0
    last_time = None

    for chunk_index, chunk in enumerate(iter_auth_chunks(raw_path, chunksize), start=1):
        chunk = _clean_chunk(chunk)

        if not redteam.empty:
            chunk = chunk.merge(redteam[label_keys + ["label"]], on=label_keys, how="left")
            chunk["label"] = chunk["label"].fillna(0).astype("int8")
        else:
            chunk["label"] = 0

        if last_time is not None and not chunk.empty and chunk["time"].iloc[0] < last_time:
            print("[parser] Warning: input is not globally sorted by time; rolling features may need sorting.")
        if not chunk.empty:
            last_time = int(chunk["time"].iloc[-1])

        chunk = _to_team_format(chunk)
        chunk.to_csv(output_path, mode="w" if not wrote_header else "a", header=not wrote_header, index=False)
        wrote_header = True

        total_rows += len(chunk)
        total_labels += int(chunk["label"].sum())
        print(
            f"[parser] chunk {chunk_index:,}: rows={len(chunk):,}, "
            f"labels={int(chunk['label'].sum()):,}, total={total_rows:,}"
        )

    print(f"[parser] Saved {total_rows:,} rows -> {_display_path(output_path)}")
    print(f"[parser] Matched redteam labels: {total_labels:,}")
    return None


def load(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[parser] Loaded parsed data: {len(df):,} rows")
    return df


def run() -> None:
    parse_file()


if __name__ == "__main__":
    run()
