#!/usr/bin/env python3
"""
Merge and deduplicate hourly parquet files into consolidated datasets.

Reads all hourly files under the configured data root (defaults to
data-pull/hourly_data) for the last N hours and writes single parquet outputs
to <data_root>/merged/.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_CONFIG_PATH = THIS_DIR / "hourly_config.json"
ARCHIVE_DIR = REPO_ROOT / "archive" / "data_store"

DATA_TARGETS = {
    "ratings": THIS_DIR / "data" / "ratings" / "ratings.parquet",
    "watches": THIS_DIR / "data" / "watches" / "watches.parquet",
    "movies": THIS_DIR / "data" / "meta" / "movies.parquet",
    "users": THIS_DIR / "data" / "meta" / "users.parquet",
}

FULL_SNAPSHOT_TARGETS = {
    "movies": THIS_DIR / "data" / "meta" / "movies_full.parquet",
    "users": THIS_DIR / "data" / "meta" / "users_full.parquet",
}

KIND_SETTINGS = {
    "ratings": {
        "subset": ["timestamp", "user_id", "movie_id", "rating"],
    },
    "watches": {
        "subset": ["timestamp", "user_id", "movie_id", "minute"],
    },
    "movies": {
        "subset": ["movie_id"],
    },
    "users": {
        "subset": ["user_id"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge hourly parquet files.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to hourly_config.json.")
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help="Number of recent hours to include (defaults to merge_hours in config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for merged output directory (defaults to <data_root>/merged).",
    )
    return parser.parse_args()


def load_config_values(config_path: Path) -> Dict:
    if not config_path.exists():
        logging.warning("Config file %s not found; using defaults.", config_path)
        return {
            "data_root": (THIS_DIR / "hourly_data").resolve(),
            "merge_hours": 24,
        }
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    raw["data_root"] = Path(raw.get("data_root", THIS_DIR / "hourly_data")).expanduser().resolve()
    raw["merge_hours"] = int(raw.get("merge_hours", 24))
    return raw


def list_hourly_files(base_dir: Path, kind: str, hours: int) -> List[Path]:
    files: List[Path] = []
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    kind_dir = base_dir / kind
    if not kind_dir.exists():
        return files

    for date_dir in sorted(kind_dir.glob("date=*")):
        try:
            date_str = date_dir.name.split("=", 1)[1]
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue
        for hour_dir in sorted(date_dir.glob("hour=*")):
            try:
                hour_str = hour_dir.name.split("=", 1)[1]
                hour_int = int(hour_str)
            except Exception:
                continue
            file_ts = datetime.combine(date_obj, datetime.min.time()).replace(hour=hour_int)
            if file_ts < cutoff:
                continue
            parquet_path = hour_dir / f"{kind}.parquet"
            if parquet_path.exists():
                files.append(parquet_path)
    return files


def merge_files(files: List[Path], subset: Optional[List[str]]) -> Optional[pd.DataFrame]:
    if not files:
        return None
    frames: List[pd.DataFrame] = []
    for path in files:
        frames.append(pd.read_parquet(path))
    combined = pd.concat(frames, ignore_index=True)
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")
    return combined


def prepare_dataset(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    if kind != "watches":
        return df

    df = df.copy()
    if {"timestamp_end", "minutes_watched"}.issubset(df.columns):
        for col in ["timestamp_start", "timestamp_end"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        if "minutes_watched" in df.columns:
            df["minutes_watched"] = pd.to_numeric(df["minutes_watched"], errors="coerce").fillna(0).astype(int)
        return df

    required_cols = {"timestamp", "user_id", "movie_id", "minute"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Watches dataset missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    aggregated = (
        df.groupby(["user_id", "movie_id"], as_index=False)
        .agg(
            timestamp_start=("timestamp", "min"),
            timestamp_end=("timestamp", "max"),
            minutes_watched=("minute", pd.Series.nunique),
        )
    )
    aggregated["minutes_watched"] = aggregated["minutes_watched"].fillna(0).astype(int)
    return aggregated


def merge_with_full_snapshot(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    target = FULL_SNAPSHOT_TARGETS[kind]
    subset = KIND_SETTINGS[kind]["subset"]
    if target.exists():
        existing = pd.read_parquet(target)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=subset, keep="last")
    else:
        combined = df.copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(target, index=False)
    return combined


def filter_snapshot_by_ids(kind: str, df: pd.DataFrame, active_ids: Set[str]) -> pd.DataFrame:
    if not active_ids:
        return df.iloc[0:0]
    key = "movie_id" if kind == "movies" else "user_id"
    return df[df[key].astype(str).isin(active_ids)]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config_values(Path(args.config))
    data_root = cfg["data_root"]
    hours_to_merge = args.hours if args.hours is not None else cfg["merge_hours"]
    if hours_to_merge <= 0:
        raise ValueError("merge_hours must be greater than zero.")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    active_user_ids: Set[str] = set()
    active_movie_ids: Set[str] = set()

    merged_frames: Dict[str, pd.DataFrame] = {}

    for kind, settings in KIND_SETTINGS.items():
        files = list_hourly_files(data_root, kind, hours_to_merge)
        logging.info("Found %d %s hourly files to merge (last %d hours).", len(files), kind, hours_to_merge)
        df = merge_files(files, settings["subset"])
        if df is None:
            logging.info("Skipping %s (no data).", kind)
            merged_frames[kind] = pd.DataFrame()
            continue
        df = prepare_dataset(kind, df)

        if kind in {"ratings", "watches"}:
            if "user_id" in df.columns:
                active_user_ids.update(df["user_id"].dropna().astype(str))
            if "movie_id" in df.columns:
                active_movie_ids.update(df["movie_id"].dropna().astype(str))
        merged_frames[kind] = df

    # Ensure we have full snapshots before filtering
    for kind in ["movies", "users"]:
        base_df = merged_frames.get(kind)
        if base_df is None or base_df.empty:
            base_df = pd.DataFrame(columns=[c for c in KIND_SETTINGS[kind]["subset"]])
        full_df = merge_with_full_snapshot(kind, merged_frames.get(kind, pd.DataFrame()))
        active_ids = active_movie_ids if kind == "movies" else active_user_ids
        merged_frames[kind] = filter_snapshot_by_ids(kind, full_df, active_ids)

    for kind, df in merged_frames.items():
        logging.info("Merged %s dataset (%d rows).", kind, len(df))
        write_primary_copy(kind, df)
        write_archive_copy(kind, df, timestamp)


def write_primary_copy(kind: str, df: pd.DataFrame) -> None:
    target_path = DATA_TARGETS[kind]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_path, index=False)
    logging.info("Updated %s with %d rows.", target_path, len(df))


def write_archive_copy(kind: str, df: pd.DataFrame, timestamp: str) -> None:
    archive_path = ARCHIVE_DIR / f"{kind}_{timestamp}.parquet"
    df.to_parquet(archive_path, index=False)
    logging.info("Archived snapshot -> %s", archive_path)


if __name__ == "__main__":
    main()
