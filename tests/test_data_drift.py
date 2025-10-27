import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pandas as pd
import pytest

if "confluent_kafka" not in sys.modules:
    class _DummyKafkaError(Exception):
        def __init__(self, code=0):
            self._code = code

        def code(self):
            return self._code

    class _DummyKafkaException(Exception):
        pass

    class _DummyConsumer:
        def __init__(self, *args, **kwargs):
            self._topics = []

        def subscribe(self, topics):
            self._topics = topics

        def consume(self, *args, **kwargs):
            return []

        def close(self):
            pass

        def commit(self, *args, **kwargs):
            pass

    sys.modules["confluent_kafka"] = type(
        "confluent_kafka",
        (),
        {
            "Consumer": _DummyConsumer,
            "KafkaError": _DummyKafkaError,
            "KafkaException": _DummyKafkaException,
            "TopicPartition": lambda topic, partition, offset: type(
                "TopicPartition", (), {"topic": topic, "partition": partition, "offset": offset}
            )(),
        },
    )

if "data_pull" not in sys.modules:
    data_pull_module = SourceFileLoader(
        "data_pull",
        str(Path(__file__).resolve().parents[1] / "data-pull" / "data_pull.py"),
    ).load_module()
    sys.modules["data_pull"] = data_pull_module

if "confluent_kafka" not in sys.modules:
    sys.modules["confluent_kafka"] = type(
        "confluent_kafka",
        (),
        {
            "KafkaError": type("KafkaError", (), {"_PARTITION_EOF": 1}),
        },
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "data-pull" / "test_data_drift.py"
module = SourceFileLoader("test_data_drift_module", str(MODULE_PATH)).load_module()


def test_get_data_stats_no_files(tmp_path, monkeypatch):
    stats = module.get_data_stats(
        tmp_path / "ratings.parquet",
        tmp_path / "watches.parquet",
        tmp_path / "movies.parquet",
        tmp_path / "users.parquet",
    )
    assert stats["ratings"]["rows"] == 0
    assert stats["ratings"]["exists"] is False


def test_get_data_stats_with_data(tmp_path, monkeypatch):
    ratings_path = tmp_path / "ratings.parquet"
    watches_path = tmp_path / "watches.parquet"
    movies_path = tmp_path / "movies.parquet"
    users_path = tmp_path / "users.parquet"

    pd.DataFrame(
        [
            {"user_id": 1, "movie_id": "m1", "rating": 4, "timestamp": pd.Timestamp("2025-01-01")},
            {"user_id": 1, "movie_id": "m2", "rating": 2, "timestamp": pd.Timestamp("2025-01-02")},
        ]
    ).to_parquet(ratings_path, index=False)
    pd.DataFrame(
        [
            {"user_id": 1, "movie_id": "m1", "minutes_watched": 10, "timestamp": pd.Timestamp("2025-01-03")},
            {"user_id": 2, "movie_id": "m2", "minutes_watched": 20, "timestamp": pd.Timestamp("2025-01-03")},
        ]
    ).to_parquet(watches_path, index=False)
    pd.DataFrame(
        [
            {"movie_id": "m1", "title": "Movie 1"},
            {"movie_id": "m2", "title": "Movie 2"},
        ]
    ).to_parquet(movies_path, index=False)
    pd.DataFrame(
        [
            {"user_id": 1, "age": 25, "gender": "M"},
            {"user_id": 2, "age": 35, "gender": "F"},
        ]
    ).to_parquet(users_path, index=False)

    stats = module.get_data_stats(ratings_path, watches_path, movies_path, users_path)
    assert stats["ratings"]["rows"] == 2
    assert stats["watches"]["rows"] == 2
    assert stats["movies"]["rows"] == 2
    assert stats["users"]["rows"] == 2
    assert "M_20" in stats["users"]["demographics_stats"]


def test_save_statistics(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.json"
    monkeypatch.setattr(module, "STATS_PATH", stats_path)
    stats_path.write_text(json.dumps({"existing": {}}))

    module.save_statistics({"value": 1})

    saved = json.loads(stats_path.read_text())
    assert len(saved) == 2  # existing and new timestamp


def test_run_data_collection_short(monkeypatch, tmp_path):
    stats_path = tmp_path / "stats.json"
    monkeypatch.setattr(module, "STATS_PATH", stats_path)

    ratings_path = tmp_path / "ratings" / "ratings.parquet"
    watches_path = tmp_path / "watches" / "watches.parquet"
    movies_path = tmp_path / "meta" / "movies.parquet"
    users_path = tmp_path / "meta" / "users.parquet"
    for p in [ratings_path, watches_path, movies_path, users_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "RATINGS_PATH", ratings_path, raising=False)
    monkeypatch.setattr(module, "WATCH_AGG_PATH", watches_path, raising=False)
    monkeypatch.setattr(module, "MOVIES_PATH", movies_path, raising=False)
    monkeypatch.setattr(module, "USERS_PATH", users_path, raising=False)

    class DummyConsumer:
        def __init__(self):
            self.consume = lambda num_messages, timeout: []

        def close(self):
            self.closed = True

        def commit(self, *args, **kwargs):
            pass

    class DummyFlag:
        def is_set(self):
            return True

    class DummyIngestor:
        def __init__(self, cfg):
            self.config = cfg
            self.consumer = DummyConsumer()
            self.shutdown_flag = DummyFlag()
            self.buffer_messages = []

        def _maybe_flush(self, force):
            pass

    monkeypatch.setattr(module, "DataIngestor", DummyIngestor)
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    monkeypatch.setattr(module.time, "time", lambda: 0)

    module.run_data_collection(duration_minutes=0.01)
    assert stats_path.exists()
