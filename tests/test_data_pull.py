from importlib.machinery import SourceFileLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PULL_PATH = PROJECT_ROOT / "data-pull" / "data_pull.py"

data_pull = SourceFileLoader("data_pull_module", str(DATA_PULL_PATH)).load_module()


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
    parsed = data_pull.parse_rating_line(line)
    assert parsed is None


def test_parse_watch_line_valid():
    line = "2025-09-18T20:39:43,19832,GET /data/m/avatar+2009/42.mpg"
    parsed = data_pull.parse_watch_line(line)
    assert parsed == {
        "timestamp": "2025-09-18T20:39:43",
        "user_id": 19832,
        "movie_id": "avatar+2009",
        "minute": 42,
    }
