import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

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
            pass

        def subscribe(self, topics):
            self.topics = topics

        def consume(self, *args, **kwargs):
            return []

        def close(self):
            pass

        def commit(self, *args, **kwargs):
            pass

    dummy_module = SimpleNamespace(
        Consumer=_DummyConsumer,
        KafkaError=_DummyKafkaError,
        KafkaException=_DummyKafkaException,
        TopicPartition=lambda topic, partition, offset: SimpleNamespace(
            topic=topic, partition=partition, offset=offset
        ),
    )
    sys.modules["confluent_kafka"] = dummy_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PULL_PATH = PROJECT_ROOT / "data-pull" / "data_pull.py"

data_pull = SourceFileLoader("data_pull", str(DATA_PULL_PATH)).load_module()


class DummyMessage:
    def __init__(self, topic="movielog8", partition=0, offset=0, error=None, value=b""):
        self._topic = topic
        self._partition = partition
        self._offset = offset
        self._error = error
        self._value = value

    def topic(self):
        return self._topic

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset

    def error(self):
        return self._error

    def value(self):
        return self._value


def _make_ingestor(monkeypatch, tmp_path, **config_overrides):
    class DummyConsumer:
        def __init__(self, conf):
            self.conf = conf
            self.topics = None
            self.closed = False
            self.committed = False
            self.offsets = None
            self.consume_response = []

        def subscribe(self, topics):
            self.topics = topics

        def consume(self, num_messages, timeout):
            if callable(self.consume_response):
                return self.consume_response()
            return self.consume_response

        def close(self):
            self.closed = True

        def commit(self, offsets, asynchronous):
            self.committed = True
            self.offsets = offsets

    class DummyTopicPartition:
        def __init__(self, topic, partition, offset):
            self.topic = topic
            self.partition = partition
            self.offset = offset

    monkeypatch.setattr(data_pull, "Consumer", DummyConsumer)
    monkeypatch.setattr(data_pull, "TopicPartition", DummyTopicPartition)

    ratings_path = tmp_path / "ratings" / "ratings.parquet"
    watches_path = tmp_path / "watches" / "watches.parquet"
    movies_path = tmp_path / "meta" / "movies.parquet"
    users_path = tmp_path / "meta" / "users.parquet"

    monkeypatch.setattr(data_pull, "RATINGS_PATH", ratings_path, raising=False)
    monkeypatch.setattr(data_pull, "WATCH_AGG_PATH", watches_path, raising=False)
    monkeypatch.setattr(data_pull, "MOVIES_PATH", movies_path, raising=False)
    monkeypatch.setattr(data_pull, "USERS_PATH", users_path, raising=False)

    data_pull.ensure_directories()

    monkeypatch.setattr(
        data_pull,
        "fetch_movies",
        lambda movie_ids, cfg: pd.DataFrame({"movie_id": list(movie_ids), "title": ["t"] * len(movie_ids)}),
    )
    monkeypatch.setattr(
        data_pull,
        "fetch_users",
        lambda user_ids, cfg: pd.DataFrame({"user_id": list(user_ids), "name": ["n"] * len(user_ids)}),
    )

    cfg_kwargs = {"metadata_workers": 1, "flush_message_count": 2, "flush_interval_seconds": 0}
    cfg_kwargs.update(config_overrides)
    cfg = data_pull.Config(**cfg_kwargs)

    ingestor = data_pull.DataIngestor(cfg)
    return ingestor, ingestor.consumer


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def test_parse_rating_line_valid():
    line = "2025-09-18T20:39:43,19832,GET /rate/saving+private+ryan+1998=4"
    parsed = data_pull.parse_rating_line(line)
    assert parsed == {
        "timestamp": "2025-09-18T20:39:43",
        "user_id": 19832,
        "movie_id": "saving+private+ryan+1998",
        "rating": 4,
    }


def test_parse_rating_line_invalid():
    line = "some,malformed,line"
    assert data_pull.parse_rating_line(line) is None


def test_parse_watch_line_valid():
    line = "2025-09-18T20:39:43,19832,GET /data/m/avatar+2009/42.mpg"
    parsed = data_pull.parse_watch_line(line)
    assert parsed == {
        "timestamp": "2025-09-18T20:39:43",
        "user_id": 19832,
        "movie_id": "avatar+2009",
        "minute": 42,
    }


def test_parse_watch_line_invalid():
    line = "2025-09-18T20:39:43,19832,GET /other/endpoint"
    assert data_pull.parse_watch_line(line) is None


# ---------------------------------------------------------------------------
# Message decoding
# ---------------------------------------------------------------------------


class DummyKafkaMessage:
    """Minimal stub to emulate confluent_kafka.Message."""

    def __init__(self, value: bytes | None, error: Exception | None = None):
        self._value = value
        self._error = error

    def error(self):
        return self._error

    def value(self):
        return self._value


def test_decode_messages_builds_dataframes():
    messages = [
        DummyKafkaMessage(b"2025-09-18T20:39:43,1,GET /rate/movie+1=5"),
        DummyKafkaMessage(b"2025-09-18T20:39:45,1,GET /data/m/movie+2/12.mpg"),
    ]

    ratings_df, watches_df = data_pull.decode_messages(messages)

    assert len(ratings_df) == 1
    assert len(watches_df) == 1
    assert pd.api.types.is_datetime64_any_dtype(ratings_df["timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(watches_df["timestamp"])


def test_decode_messages_skips_errors():
    faulty = DummyKafkaMessage(b"bad payload")
    faulty._value = b"bad"
    messages = [
        DummyKafkaMessage(None, error=Exception("boom")),
        faulty,
    ]

    ratings_df, watches_df = data_pull.decode_messages(messages)
    assert ratings_df.empty and watches_df.empty


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def test_append_dataframe_deduplicates(tmp_path):
    path = tmp_path / "ratings.parquet"
    df1 = pd.DataFrame(
        [
            {"timestamp": "2025-01-01", "user_id": 1, "movie_id": "a", "rating": 4},
            {"timestamp": "2025-01-02", "user_id": 2, "movie_id": "b", "rating": 5},
        ]
    )
    df2 = pd.DataFrame(
        [
            {"timestamp": "2025-01-02", "user_id": 2, "movie_id": "b", "rating": 5},
            {"timestamp": "2025-01-03", "user_id": 3, "movie_id": "c", "rating": 3},
        ]
    )

    subset = ["timestamp", "user_id", "movie_id", "rating"]
    data_pull.append_dataframe(path, df1, subset=subset)
    data_pull.append_dataframe(path, df2, subset=subset)

    combined = pd.read_parquet(path)
    assert len(combined) == 3
    assert set(combined["user_id"]) == {1, 2, 3}


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def test_fetch_metadata_item_success(monkeypatch):
    class DummyResp:
        status_code = 200

        def json(self):
            return {"title": "foo"}

    monkeypatch.setattr(
        data_pull.requests, "get", lambda url, timeout: DummyResp()
    )

    result = data_pull.fetch_metadata_item("http://example.com/movie/1", timeout=5)
    assert result == {"title": "foo"}


def test_fetch_metadata_item_failure(monkeypatch):
    class DummyResp:
        status_code = 500

        def json(self):
            return {}

    monkeypatch.setattr(
        data_pull.requests, "get", lambda url, timeout: DummyResp()
    )

    assert data_pull.fetch_metadata_item("http://example.com/movie/1", timeout=5) is None


def test_fetch_movies_uses_helper(monkeypatch):
    calls = []

    def fake_fetch(url, timeout):
        calls.append(url)
        return {"name": f"movie-{url.split('/')[-1]}"}  # no movie_id key

    monkeypatch.setattr(data_pull, "fetch_metadata_item", fake_fetch)

    movie_ids = ["m1", "m2", "m3"]
    df = data_pull.fetch_movies(movie_ids, data_pull.Config(metadata_workers=2))

    assert len(df) == len(movie_ids)
    assert set(df["movie_id"]) == set(movie_ids)
    assert len(calls) == len(movie_ids)


def test_fetch_users_uses_helper(monkeypatch):
    calls = []

    def fake_fetch(url, timeout):
        calls.append(url)
        return {"name": f"user-{url.split('/')[-1]}"}  # missing user_id on purpose

    monkeypatch.setattr(data_pull, "fetch_metadata_item", fake_fetch)

    user_ids = [1, 2, 3]
    df = data_pull.fetch_users(user_ids, data_pull.Config(metadata_workers=2))

    assert len(df) == len(user_ids)
    assert set(df["user_id"]) == set(user_ids)
    assert len(calls) == len(user_ids)


def test_fetch_movies_empty_returns_empty():
    cfg = data_pull.Config()
    df = data_pull.fetch_movies([], cfg)
    assert df.empty


def test_fetch_users_empty_returns_empty():
    cfg = data_pull.Config()
    df = data_pull.fetch_users([], cfg)
    assert df.empty


# ---------------------------------------------------------------------------
# Infrastructure helpers and DataIngestor
# ---------------------------------------------------------------------------


def test_build_consumer_uses_config(monkeypatch):
    class DummyConsumer:
        def __init__(self, conf):
            self.conf = conf
            self.topics = None

        def subscribe(self, topics):
            self.topics = topics

    monkeypatch.setattr(data_pull, "Consumer", DummyConsumer)
    cfg = data_pull.Config(bootstrap_servers="broker:9092", topic="movielogX")

    consumer = data_pull.build_consumer(cfg)

    assert consumer.conf["bootstrap.servers"] == "broker:9092"
    assert consumer.topics == ["movielogX"]


def test_ensure_directories_creates_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(data_pull, "RATINGS_PATH", tmp_path / "ratings" / "ratings.parquet", raising=False)
    monkeypatch.setattr(data_pull, "WATCH_AGG_PATH", tmp_path / "watches" / "watches.parquet", raising=False)
    monkeypatch.setattr(data_pull, "MOVIES_PATH", tmp_path / "meta" / "movies.parquet", raising=False)
    monkeypatch.setattr(data_pull, "USERS_PATH", tmp_path / "meta" / "users.parquet", raising=False)

    data_pull.ensure_directories()

    assert (tmp_path / "ratings").exists()
    assert (tmp_path / "watches").exists()
    assert (tmp_path / "meta").exists()


def test_load_existing_ids_with_parquet(tmp_path, monkeypatch):
    path = tmp_path / "ratings.parquet"
    df = pd.DataFrame({"user_id": [1, 2, 2, None]})
    df.to_parquet(path, index=False)

    ids = data_pull.load_existing_ids(path, "user_id")
    assert ids == {1, 2}


def test_data_ingestor_persist_and_commit(tmp_path, monkeypatch):
    ingestor, consumer = _make_ingestor(monkeypatch, tmp_path)
    ratings_path = data_pull.RATINGS_PATH
    watches_path = data_pull.WATCH_AGG_PATH

    ratings_df = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2025-01-01"), "user_id": 1, "movie_id": "m1", "rating": 4},
            {"timestamp": pd.Timestamp("2025-01-02"), "user_id": 2, "movie_id": "m2", "rating": 5},
        ]
    )
    watch_df = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2025-01-03"), "user_id": 1, "movie_id": "m1", "minute": 10},
            {"timestamp": pd.Timestamp("2025-01-04"), "user_id": 2, "movie_id": "m2", "minute": 20},
        ]
    )

    ingestor._persist_data(ratings_df, watch_df)

    stored_ratings = pd.read_parquet(ratings_path)
    stored_watches = pd.read_parquet(watches_path)
    assert len(stored_ratings) == 2
    assert len(stored_watches) == 2
    assert "minutes_watched" in stored_watches.columns

    # second call to cover aggregation with existing parquet
    watch_df_2 = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2025-01-05"), "user_id": 1, "movie_id": "m1", "minute": 5},
        ]
    )
    ingestor._persist_data(pd.DataFrame(), watch_df_2)

    ingestor._commit_offsets([DummyMessage(offset=5)])
    assert consumer.committed


def test_maybe_flush_skips_when_not_ready(tmp_path, monkeypatch):
    ingestor, consumer = _make_ingestor(monkeypatch, tmp_path, flush_message_count=5, flush_interval_seconds=1000)
    ingestor.buffer_messages = []
    ingestor._maybe_flush(force=False)
    assert not consumer.committed


def test_maybe_flush_processes_buffer(tmp_path, monkeypatch):
    ingestor, consumer = _make_ingestor(monkeypatch, tmp_path, flush_message_count=1)

    ratings_df = pd.DataFrame({"timestamp": ["2025"], "user_id": [1], "movie_id": ["m1"], "rating": [5]})
    watches_df = pd.DataFrame({"timestamp": ["2025"], "user_id": [1], "movie_id": ["m1"], "minute": [1]})

    monkeypatch.setattr(data_pull, "decode_messages", lambda msgs: (ratings_df, watches_df))

    persist_called = {"called": False}
    commit_called = {"called": False}

    def fake_persist(self, r_df, w_df):
        persist_called["called"] = True

    def fake_commit(self, msgs):
        commit_called["called"] = True

    monkeypatch.setattr(data_pull.DataIngestor, "_persist_data", fake_persist)
    monkeypatch.setattr(data_pull.DataIngestor, "_commit_offsets", fake_commit)

    ingestor.buffer_messages = [DummyMessage(offset=1)]
    ingestor._maybe_flush(force=False)

    assert persist_called["called"]
    assert commit_called["called"]
    assert ingestor.buffer_messages == []


def test_persist_data_without_watch(tmp_path, monkeypatch):
    ingestor, _ = _make_ingestor(monkeypatch, tmp_path)
    ratings_df = pd.DataFrame({"timestamp": ["2025"], "user_id": [1], "movie_id": ["m1"], "rating": [4]})
    watch_df = pd.DataFrame()

    ingestor.existing_movie_ids = {"m1"}
    ingestor.existing_user_ids = {1}

    def fail_fetch(*args, **kwargs):  # should not be called
        raise AssertionError("Metadata fetch should not be called")

    monkeypatch.setattr(data_pull, "fetch_movies", fail_fetch)
    monkeypatch.setattr(data_pull, "fetch_users", fail_fetch)

    ingestor._persist_data(ratings_df, watch_df)

    stored = pd.read_parquet(data_pull.RATINGS_PATH)
    assert len(stored) == 1


def test_commit_offsets_handles_exception(tmp_path, monkeypatch):
    ingestor, consumer = _make_ingestor(monkeypatch, tmp_path)

    class DummyKafkaException(Exception):
        pass

    monkeypatch.setattr(data_pull, "KafkaException", DummyKafkaException)

    def fail_commit(offsets, asynchronous):
        raise DummyKafkaException("boom")

    consumer.commit = fail_commit

    ingestor._commit_offsets([DummyMessage(offset=2)])  # should not raise


def test_run_handles_keyboardinterrupt(tmp_path, monkeypatch):
    ingestor, consumer = _make_ingestor(monkeypatch, tmp_path)

    def raise_interrupt(num_messages, timeout):
        raise KeyboardInterrupt

    consumer.consume = raise_interrupt

    flush_calls = []

    def fake_flush(force):
        flush_calls.append(force)

    monkeypatch.setattr(ingestor, "_maybe_flush", fake_flush)

    ingestor.run()

    assert consumer.closed
    assert flush_calls == [True]
