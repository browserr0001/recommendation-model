#!/usr/bin/env python3
"""
Continuous data puller that mirrors the logic in data-pull.ipynb.

The script:
* Streams Kafka messages indefinitely until manually stopped.
* Parses rating and watch events, persisting them to parquet files.
* Maintains an aggregated watches parquet derived from raw watch events.
* Fetches movie and user metadata for any newly observed ids.

All configuration lives in the constants section below; no external files required.
"""

from __future__ import annotations

import logging
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from confluent_kafka import Consumer, KafkaError, KafkaException, TopicPartition

# --------------------------------------------------------------------------- #
# Constants & defaults
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RATINGS_PATH = DATA_DIR / "ratings" / "ratings.parquet"
WATCH_AGG_PATH = DATA_DIR / "watches" / "watches.parquet"
MOVIES_PATH = DATA_DIR / "meta" / "movies.parquet"
USERS_PATH = DATA_DIR / "meta" / "users.parquet"


# --------------------------------------------------------------------------- #
# Settings & Consumer construction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    bootstrap_servers: str = "localhost:9092"
    topic: str = "movielog8"
    group_id: str = "ratings-consumer"
    poll_timeout: float = 1.0
    kafka_batch_size: int = 5000
    flush_message_count: int = 20_000
    flush_interval_seconds: float = 30.0
    base_metadata_url: str = "http://128.2.220.241:8080"
    metadata_workers: int = 16
    request_timeout: float = 10.0


CONFIG = Config()


def build_consumer(config: Config) -> Consumer:
    conf = {
        "bootstrap.servers": config.bootstrap_servers,
        "group.id": config.group_id,
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
        "session.timeout.ms": 45000,
    }

    consumer = Consumer(conf)
    consumer.subscribe([config.topic])
    return consumer


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def parse_rating_line(line: str) -> Optional[Dict]:
    try:
        timestamp, user_id, request = line.split(",", 2)
        match = re.search(r"/rate/(.+)=(\d+)", request)
        if match:
            return {
                "timestamp": timestamp,
                "user_id": int(user_id),
                "movie_id": match.group(1),
                "rating": int(match.group(2)),
            }
    except Exception:
        return None
    return None


def parse_watch_line(line: str) -> Optional[Dict]:
    try:
        timestamp, user_id, request = line.split(",", 2)
        match = re.search(r"/data/m/(.+)/(\d+)\.mpg", request)
        if match:
            return {
                "timestamp": timestamp,
                "user_id": int(user_id),
                "movie_id": match.group(1),
                "minute": int(match.group(2)),
            }
    except Exception:
        return None
    return None


def decode_messages(messages: Sequence) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rating_records: List[Dict] = []
    watch_records: List[Dict] = []

    for msg in messages:
        if msg.error():
            # log and skip errors
            logging.error("Consumer error: %s", msg.error())
            continue
        try:
            payload = msg.value().decode("utf-8")
        except Exception as exc:
            logging.warning("Failed to decode message: %s", exc)
            continue

        rating = parse_rating_line(payload)
        if rating:
            rating_records.append(rating)
            continue

        watch = parse_watch_line(payload)
        if watch:
            watch_records.append(watch)

    ratings_df = pd.DataFrame(rating_records)
    if not ratings_df.empty:
        ratings_df["timestamp"] = pd.to_datetime(ratings_df["timestamp"], errors="coerce")

    watch_df = pd.DataFrame(watch_records)
    if not watch_df.empty:
        watch_df["timestamp"] = pd.to_datetime(watch_df["timestamp"], errors="coerce")

    return ratings_df, watch_df


# --------------------------------------------------------------------------- #
# Storage & metadata helpers
# --------------------------------------------------------------------------- #


def ensure_directories() -> None:
    for directory in [
        RATINGS_PATH.parent,
        WATCH_AGG_PATH.parent,
        MOVIES_PATH.parent,
        USERS_PATH.parent,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def append_dataframe(path: Path, df: pd.DataFrame, subset: Optional[List[str]] = None) -> None:
    if df.empty:
        return
    engine_kwargs = {"engine": "pyarrow"}
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")
    combined.to_parquet(path, index=False, **engine_kwargs)


def load_existing_ids(path: Path, column: str) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_parquet(path, columns=[column])
    except Exception:
        return set()
    return set(df[column].dropna().unique().tolist())


def fetch_metadata_item(url: str, timeout: float) -> Optional[Dict]:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logging.warning("Metadata fetch error for %s: %s", url, exc)
    return None


def fetch_movies(movie_ids: Iterable[str], config: Config) -> pd.DataFrame:
    if not movie_ids:
        return pd.DataFrame()

    records: List[Dict] = []
    with ThreadPoolExecutor(max_workers=config.metadata_workers) as executor:
        future_to_movie = {
            executor.submit(
                fetch_metadata_item,
                f"{config.base_metadata_url}/movie/{movie_id}",
                config.request_timeout,
            ): movie_id
            for movie_id in movie_ids
        }
        for future in as_completed(future_to_movie):
            movie_id = future_to_movie[future]
            data = future.result()
            if data:
                data["movie_id"] = movie_id
                records.append(data)
    return pd.DataFrame(records)


def fetch_users(user_ids: Iterable[int], config: Config) -> pd.DataFrame:
    if not user_ids:
        return pd.DataFrame()

    records: List[Dict] = []
    with ThreadPoolExecutor(max_workers=config.metadata_workers) as executor:
        future_to_user = {
            executor.submit(
                fetch_metadata_item,
                f"{config.base_metadata_url}/user/{user_id}",
                config.request_timeout,
            ): user_id
            for user_id in user_ids
        }
        for future in as_completed(future_to_user):
            user_id = future_to_user[future]
            data = future.result()
            if data:
                data["user_id"] = int(data.get("user_id", user_id))
                records.append(data)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Core run loop
# --------------------------------------------------------------------------- #


class DataIngestor:
    def __init__(self, config: Config):
        self.config = config
        ensure_directories()
        self.consumer = build_consumer(config)
        self.buffer_messages: List = []
        self.last_flush_ts = time.monotonic()
        self.shutdown_flag = threading.Event()

        self.existing_movie_ids = load_existing_ids(MOVIES_PATH, "movie_id")
        self.existing_user_ids = load_existing_ids(USERS_PATH, "user_id")

    def run(self) -> None:
        logging.info("Starting continuous data pull. Press Ctrl+C to stop.")
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

        logging.info("Flushing %d messages.", len(self.buffer_messages))
        ratings_df, watch_df = decode_messages(self.buffer_messages)
        self._persist_data(ratings_df, watch_df)
        self._commit_offsets(self.buffer_messages)
        self.buffer_messages.clear()
        self.last_flush_ts = time.monotonic()

    def _persist_data(self, ratings_df: pd.DataFrame, watch_df: pd.DataFrame) -> None:
        # Ratings
        if not ratings_df.empty:
            ratings_df = ratings_df.drop_duplicates(subset=["timestamp", "user_id", "movie_id", "rating"])
            append_dataframe(
                RATINGS_PATH,
                ratings_df,
                subset=["timestamp", "user_id", "movie_id", "rating"],
            )
            logging.info("Appended %d rating rows.", len(ratings_df))

        # Watch events
        if not watch_df.empty:
            watch_df = watch_df.drop_duplicates(subset=["timestamp", "user_id", "movie_id", "minute"])
            collapsed = (
                watch_df.groupby(["user_id", "movie_id"], as_index=False)
                .agg(
                    timestamp_start=("timestamp", "min"),
                    timestamp_end=("timestamp", "max"),
                    minutes_watched=("minute", pd.Series.nunique),
                )
            )

            if WATCH_AGG_PATH.exists():
                existing = pd.read_parquet(WATCH_AGG_PATH)
                combined = pd.concat([existing, collapsed], ignore_index=True)
                aggregated = (
                    combined.groupby(["user_id", "movie_id"], as_index=False)
                    .agg(
                        timestamp_start=("timestamp_start", "min"),
                        timestamp_end=("timestamp_end", "max"),
                        minutes_watched=("minutes_watched", "sum"),
                    )
                )
            else:
                aggregated = collapsed

            aggregated["timestamp_start"] = pd.to_datetime(aggregated["timestamp_start"], errors="coerce")
            aggregated["timestamp_end"] = pd.to_datetime(aggregated["timestamp_end"], errors="coerce")
            aggregated["minutes_watched"] = aggregated["minutes_watched"].fillna(0).astype(int)
            aggregated.to_parquet(WATCH_AGG_PATH, index=False)
            logging.info("Updated watches.parquet with %d new rows.", len(collapsed))

        # Metadata
        new_movie_ids = set(ratings_df.get("movie_id", pd.Series(dtype=str)).dropna().unique()) | set(
            watch_df.get("movie_id", pd.Series(dtype=str)).dropna().unique()
        )
        new_user_ids = set(ratings_df.get("user_id", pd.Series(dtype=int)).dropna().unique()) | set(
            watch_df.get("user_id", pd.Series(dtype=int)).dropna().unique()
        )
        missing_movies = sorted(new_movie_ids - self.existing_movie_ids)
        missing_users = sorted(new_user_ids - self.existing_user_ids)

        if missing_movies:
            movies_df = fetch_movies(missing_movies, self.config)
            if not movies_df.empty:
                append_dataframe(MOVIES_PATH, movies_df, subset=["movie_id"])
                self.existing_movie_ids.update(movies_df["movie_id"].unique().tolist())
                logging.info("Fetched %d new movie metadata rows.", len(movies_df))

        if missing_users:
            users_df = fetch_users(missing_users, self.config)
            if not users_df.empty:
                append_dataframe(USERS_PATH, users_df, subset=["user_id"])
                self.existing_user_ids.update(users_df["user_id"].unique().tolist())
                logging.info("Fetched %d new user metadata rows.", len(users_df))

    def _commit_offsets(self, messages: Sequence) -> None:
        offsets: Dict[Tuple[str, int], TopicPartition] = {}
        for msg in messages:
            topic = msg.topic()
            partition = msg.partition()
            offsets[(topic, partition)] = TopicPartition(topic, partition, msg.offset() + 1)
        try:
            if offsets:
                self.consumer.commit(offsets=list(offsets.values()), asynchronous=False)
        except KafkaException as exc:
            logging.error("Offset commit failed: %s", exc)

# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    configure_logging()
    ingestor = DataIngestor(CONFIG)

    def handle_sigterm(signum, frame):  # pragma: no cover - signal handler
        logging.info("Signal %s received. Stopping...", signum)
        ingestor.shutdown_flag.set()
        ingestor._maybe_flush(force=True)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, signal.default_int_handler)  # allow KeyboardInterrupt

    ingestor.run()


if __name__ == "__main__":
    main()
