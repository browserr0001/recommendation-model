from importlib.machinery import SourceFileLoader
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PULL_PATH = PROJECT_ROOT / "data-pull" / "data_pull.py"

data_pull = SourceFileLoader("data_pull_module", str(DATA_PULL_PATH)).load_module()


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
