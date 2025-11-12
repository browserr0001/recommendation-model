#!/usr/bin/env python3
"""
Continuous Kafka puller that writes immutable hourly parquet files.

Configuration is provided via data-pull/hourly_config.json. The script reuses
the parsing helpers from data_pull.py but stores each flush into folders like:

    data/hourly/ratings/date=2025-02-09/hour=13/*.parquet
    data/hourly/watches/date=2025-02-09/hour=13/*.parquet

Run it indefinitely on the VM (tmux/systemd) as long as the Kafka tunnel is up.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import json

import pandas as pd
from confluent_kafka import Consumer, KafkaError, KafkaException, TopicPartition

import data_pull  # reuse parsing utilities


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = BASE_DIR / "hourly_data"
DEFAULT_CONFIG_PATH = BASE_DIR / "hourly_config.json"


@dataclass(frozen=True)
class Config:
    bootstrap_servers: str = "localhost:9092"
    topic: str = "movielog8"
    group_id: str = "hourly-data-consumer"
    poll_timeout: float = 1.0
    kafka_batch_size: int = 5000
    flush_message_count: int = 20_000
    flush_interval_seconds: float = 30.0
    auto_offset_reset: str = "earliest"
    data_root: Path = DEFAULT_DATA_ROOT
    base_metadata_url: str = data_pull.CONFIG.base_metadata_url
    metadata_workers: int = data_pull.CONFIG.metadata_workers
    request_timeout: float = data_pull.CONFIG.request_timeout


def build_consumer(config: Config) -> Consumer:
    conf = {
        "bootstrap.servers": config.bootstrap_servers,
        "group.id": config.group_id,
        "auto.offset.reset": config.auto_offset_reset,
        "enable.auto.commit": False,
        "session.timeout.ms": 45000,
    }
    consumer = Consumer(conf)
    consumer.subscribe([config.topic])
    return consumer


# --------------------------------------------------------------------------- #
# Ingestion + storage
# --------------------------------------------------------------------------- #


def ensure_directories(root: Path) -> None:
    for sub in ["ratings", "watches", "movies", "users"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.utcnow())
    return df


def partition_dataframe(df: pd.DataFrame) -> Iterable[Tuple[str, str, pd.DataFrame]]:
    if df.empty:
        return []
    df = normalize_timestamps(df)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["timestamp"].dt.strftime("%H")
    grouped = df.groupby(["date", "hour"], as_index=False)
    return [(date, hour, group.drop(columns=["date", "hour"])) for (date, hour), group in grouped]


class HourlyIngestor:
    def __init__(self, config: Config):
        self.config = config
        ensure_directories(self.config.data_root)
        self.consumer = build_consumer(config)
        self.buffer_messages: List = []
        self.last_flush_ts = time.monotonic()
        self.shutdown_flag = threading.Event()
        self.state_dir = self.config.data_root / "metadata_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.movies_seen = self._load_seen_ids(self.state_dir / "movies_seen.json")
        self.users_seen = self._load_seen_ids(self.state_dir / "users_seen.json")
        self.metadata_config = data_pull.Config(
            base_metadata_url=self.config.base_metadata_url,
            metadata_workers=self.config.metadata_workers,
            request_timeout=self.config.request_timeout,
        )

    def run(self) -> None:
        logging.info(
            "Starting hourly data pull | topic=%s group=%s data_root=%s",
            self.config.topic,
            self.config.group_id,
            self.config.data_root,
        )
        logging.info(
            "Consumer settings | offset_reset=%s batch=%d flush_every=%d msgs or %.1fs",
            self.config.auto_offset_reset,
            self.config.kafka_batch_size,
            self.config.flush_message_count,
            self.config.flush_interval_seconds,
        )
        try:
            while not self.shutdown_flag.is_set():
                messages = self.consumer.consume(
                    num_messages=self.config.kafka_batch_size,
                    timeout=self.config.poll_timeout,
                )
                if not messages:
                    self._maybe_flush(force=False)
                    continue

                for msg in messages:
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        logging.error("Kafka error: %s", msg.error())
                        continue
                    self.buffer_messages.append(msg)

                self._maybe_flush(force=False)
        except KeyboardInterrupt:
            logging.info("Interrupt received; flushing buffers before exit.")
            self.shutdown_flag.set()
            self._maybe_flush(force=True)
        finally:
            self.consumer.close()
            logging.info("Consumer closed. Shutdown complete.")

    def _maybe_flush(self, force: bool) -> None:
        should_flush = force
        if not should_flush and self.buffer_messages:
            if len(self.buffer_messages) >= self.config.flush_message_count:
                should_flush = True
            elif (time.monotonic() - self.last_flush_ts) >= self.config.flush_interval_seconds:
                should_flush = True

        if not should_flush:
            return

        if not self.buffer_messages:
            self.last_flush_ts = time.monotonic()
            return

        batch_age = time.monotonic() - self.last_flush_ts
        logging.info(
            "Flush start | messages=%d buffer_age=%.1fs",
            len(self.buffer_messages),
            batch_age,
        )
        ratings_df, watch_df = data_pull.decode_messages(self.buffer_messages)
        offsets = self._capture_offsets(self.buffer_messages)
        self._persist_frames("ratings", ratings_df, offsets)
        self._persist_frames("watches", watch_df, offsets)
        logging.info(
            "Flush complete | ratings_rows=%d watches_rows=%d partitions=%d duration=%.2fs",
            len(ratings_df),
            len(watch_df),
            len(offsets),
            time.monotonic() - self.last_flush_ts,
        )
        self._commit_offsets(self.buffer_messages)
        self.buffer_messages.clear()
        self.last_flush_ts = time.monotonic()

    def _persist_frames(self, kind: str, df: pd.DataFrame, offsets: Dict[Tuple[str, int], Dict[str, int]]) -> None:
        if df.empty:
            return

        subset_map = {
            "ratings": ["timestamp", "user_id", "movie_id", "rating"],
            "watches": ["timestamp", "user_id", "movie_id", "minute"],
        }

        metadata_requests: Dict[str, List[Tuple[str, str, Set]]] = {"movies": [], "users": []}

        for date_str, hour_str, chunk in partition_dataframe(df):
            dest_dir = self.config.data_root / kind / f"date={date_str}" / f"hour={hour_str}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f"{kind}.parquet"

            chunk = chunk.drop_duplicates(subset=subset_map[kind])
            persisted = self._merge_with_existing(dest_file, chunk, subset_map[kind])
            if persisted is None:
                continue

            file_rows = len(persisted)
            persisted.to_parquet(dest_file, index=False)
            self._write_manifest(dest_dir, kind, dest_file.name, file_rows, offsets)
            logging.info(
                "[%s] date=%s hour=%s rows=%d path=%s",
                kind,
                date_str,
                hour_str,
                file_rows,
                dest_file,
            )

            chunk_movies = set(chunk.get("movie_id", pd.Series(dtype=str)).dropna().astype(str))
            chunk_users = set(chunk.get("user_id", pd.Series(dtype=int)).dropna().astype(int))
            if chunk_movies:
                metadata_requests["movies"].append((date_str, hour_str, chunk_movies))
            if chunk_users:
                metadata_requests["users"].append((date_str, hour_str, chunk_users))

        self._handle_metadata_requests(metadata_requests)

    def _merge_with_existing(
        self,
        dest_file: Path,
        chunk: pd.DataFrame,
        subset: List[str],
    ) -> Optional[pd.DataFrame]:
        """Append chunk to destination, dropping duplicates."""
        frames = [chunk]
        if dest_file.exists():
            frames.insert(0, pd.read_parquet(dest_file))
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=subset, keep="last")
        if combined.empty:
            return None
        return combined

    def _handle_metadata_requests(self, requests: Dict[str, List[Tuple[str, str, Set]]]) -> None:
        if not any(requests.values()):
            return

        for kind, entries in requests.items():
            if not entries:
                continue
            for date_str, hour_str, ids in entries:
                self._write_metadata_chunk(kind, ids, date_str, hour_str)

    def _write_metadata_chunk(self, kind: str, ids: Set, date: str, hour: str) -> None:
        subset_map = {"movies": ["movie_id"], "users": ["user_id"]}
        seen_set = self.movies_seen if kind == "movies" else self.users_seen
        new_ids = set(ids) - seen_set
        if not new_ids:
            return

        dest_dir = self.config.data_root / kind / f"date={date}" / f"hour={hour}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{kind}.parquet"

        if kind == "movies":
            df = data_pull.fetch_movies(sorted(new_ids), self.metadata_config)
        else:
            df = data_pull.fetch_users(sorted(new_ids), self.metadata_config)

        if df.empty:
            return

        df = self._prepare_metadata(kind, df)
        df["fetched_at"] = datetime.utcnow().replace(tzinfo=timezone.utc)
        combined = self._merge_with_existing(dest_file, df, subset_map[kind])
        if combined is None:
            return
        combined.to_parquet(dest_file, index=False)
        logging.info(
            "[metadata:%s] date=%s hour=%s new_rows=%d total_seen=%d path=%s",
            kind,
            date,
            hour,
            len(df),
            len(seen_set),
            dest_file,
        )

        seen_set.update(new_ids)
        state_path = self.state_dir / f"{kind}_seen.json"
        self._save_seen_ids(state_path, seen_set)

    def _write_manifest(
        self,
        dest_dir: Path,
        kind: str,
        filename: str,
        rows: int,
        offsets: Dict[Tuple[str, int], Dict[str, int]],
    ) -> None:
        manifest_path = dest_dir / "manifest.jsonl"
        entry = {
            "kind": kind,
            "file": filename,
            "rows": rows,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "offsets": [
                {
                    "topic": topic,
                    "partition": partition,
                    "start_offset": info["start"],
                    "end_offset": info["end"],
                }
                for (topic, partition), info in offsets.items()
            ],
        }
        with open(manifest_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _capture_offsets(self, messages: Sequence) -> Dict[Tuple[str, int], Dict[str, int]]:
        offsets: Dict[Tuple[str, int], Dict[str, int]] = {}
        for msg in messages:
            key = (msg.topic(), msg.partition())
            offsets.setdefault(key, {"start": msg.offset(), "end": msg.offset()})
            offsets[key]["end"] = msg.offset()
        return offsets

    def _commit_offsets(self, messages: Sequence) -> None:
        topic_offsets: Dict[Tuple[str, int], TopicPartition] = {}
        for msg in messages:
            topic = msg.topic()
            partition = msg.partition()
            topic_offsets[(topic, partition)] = TopicPartition(topic, partition, msg.offset() + 1)
        try:
            if topic_offsets:
                self.consumer.commit(offsets=list(topic_offsets.values()), asynchronous=False)
        except KafkaException as exc:
            logging.error("Offset commit failed: %s", exc)

    def _load_seen_ids(self, path: Path) -> Set:
        if not path.exists():
            return set()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return set(data)
        except Exception:
            return set()

    def _save_seen_ids(self, path: Path, values: Set) -> None:
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(sorted(values)), encoding="utf-8")
        tmp_path.replace(path)
    def _prepare_metadata(self, kind: str, df: pd.DataFrame) -> pd.DataFrame:
        if kind == "movies":
            drop_cols = [col for col in ["belongs_to_collection"] if col in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)
        return df


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    defaults = Config()
    if not path.exists():
        logging.warning("Config file %s not found; using defaults.", path)
        return defaults

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    data_root = Path(raw.get("data_root", defaults.data_root)).expanduser()
    auto_offset_reset = raw.get(
        "auto_offset_reset",
        "latest" if raw.get("start_latest") else defaults.auto_offset_reset,
    )

    return Config(
        bootstrap_servers=raw.get("bootstrap_servers", defaults.bootstrap_servers),
        topic=raw.get("topic", defaults.topic),
        group_id=raw.get("group_id", defaults.group_id),
        poll_timeout=raw.get("poll_timeout", defaults.poll_timeout),
        kafka_batch_size=raw.get("kafka_batch_size", defaults.kafka_batch_size),
        flush_message_count=raw.get("flush_message_count", defaults.flush_message_count),
        flush_interval_seconds=raw.get("flush_interval_seconds", defaults.flush_interval_seconds),
        auto_offset_reset=auto_offset_reset,
        data_root=data_root.resolve(),
        base_metadata_url=raw.get("base_metadata_url", defaults.base_metadata_url),
        metadata_workers=raw.get("metadata_workers", defaults.metadata_workers),
        request_timeout=raw.get("request_timeout", defaults.request_timeout),
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    configure_logging()
    config = load_config()
    ingestor = HourlyIngestor(config)

    def handle_sigterm(signum, frame):  # pragma: no cover - signal handler
        logging.info("Signal %s received. Stopping...", signum)
        ingestor.shutdown_flag.set()
        ingestor._maybe_flush(force=True)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    ingestor.run()


if __name__ == "__main__":
    main()
