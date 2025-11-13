#!/usr/bin/env python3
"""
Merge and deduplicate hourly parquet files into consolidated datasets.

Reads all hourly files under the configured data root (defaults to
data-pull/hourly_data) for the last N days and writes single parquet outputs
to <data_root>/merged/.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Optional

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
        "--days",
        type=int,
        default=None,
        help="Number of recent days to include (defaults to merge_days in config).",
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
            "merge_days": 2,
        }
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    raw["data_root"] = Path(raw.get("data_root", THIS_DIR / "hourly_data")).expanduser().resolve()
    raw["merge_days"] = int(raw.get("merge_days", 2))
    return raw


def list_hourly_files(base_dir: Path, kind: str, days: int) -> List[Path]:
    files: List[Path] = []
    cutoff = (datetime.utcnow() - timedelta(days=days)).date()
    kind_dir = base_dir / kind
    if not kind_dir.exists():
        return files

    for date_dir in sorted(kind_dir.glob("date=*")):
        try:
            date_str = date_dir.name.split("=", 1)[1]
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue
        if date_obj < cutoff:
            continue
        for hour_dir in sorted(date_dir.glob("hour=*")):
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


def merge_with_existing_snapshot(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    """Union the fresh merge with the previous snapshot to retain historical rows."""
    target_path = DATA_TARGETS[kind]
    if not target_path.exists():
        return df
    existing = pd.read_parquet(target_path)
    combined = pd.concat([existing, df], ignore_index=True)
    subset = KIND_SETTINGS[kind].get("subset")
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

    # Raw watch events -> aggregate into schema expected by training pipeline
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config_values(Path(args.config))
    data_root = cfg["data_root"]
    days_to_merge = args.days if args.days is not None else cfg["merge_days"]
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    for kind, settings in KIND_SETTINGS.items():
        files = list_hourly_files(data_root, kind, days_to_merge)
        logging.info("Found %d %s hourly files to merge (last %d days).", len(files), kind, days_to_merge)
        df = merge_files(files, settings["subset"])
        if df is None:
            logging.info("Skipping %s (no data).", kind)
            continue
        df = prepare_dataset(kind, df)
        if kind in {"movies", "users"}:
            df = merge_with_existing_snapshot(kind, df)
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
