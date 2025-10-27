"""
Comprehensive unit tests for Online Evaluation pipeline (app/model/online_eval.py)
Covers:
- Time parsing, cohort classification, metric computation
- CTR/diversity aggregation and persistence
- Regex-based log parsing and state updates (reviewability)
"""

import pytest
import numpy as np
import time
import json
import os
import csv
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from importlib.machinery import SourceFileLoader

# ------- LOAD TARGET MODULE DYNAMICALLY --------------

# Compute project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to your online evaluation module
ONLINE_EVAL_PATH = PROJECT_ROOT / "app" / "model" / "online.py"

# Dynamically load the module
online_eval = SourceFileLoader("online_eval_module", str(ONLINE_EVAL_PATH)).load_module()

# Import functions and globals from dynamically loaded module
parse_time = online_eval.parse_time
cohort_for_rec = online_eval.cohort_for_rec
process_line = online_eval.process_line
evaluate_user = online_eval.evaluate_user
compute_ctr_by_cohort = online_eval.compute_ctr_by_cohort
diversity_coverage_by_cohort = online_eval.diversity_coverage_by_cohort
aggregate_metrics_all = online_eval.aggregate_metrics_all
persist_metrics = online_eval.persist_metrics

metrics_buffer = online_eval.metrics_buffer
user_watched = online_eval.user_watched
user_rated = online_eval.user_rated
user_rating_counts = online_eval.user_rating_counts
user_minutes_watched = online_eval.user_minutes_watched
recommendations = online_eval.recommendations
MODEL_KNOWN_USERS = online_eval.MODEL_KNOWN_USERS

# ----------- HELPER FIXTURES / RESET STATE -----------

@pytest.fixture(autouse=True)
def reset_state():
    """Reset global mutable state between tests."""
    metrics_buffer.clear()
    for c in ["all", "warm", "cold"]:
        metrics_buffer[c] = defaultdict(list)
    user_watched.clear()
    user_rated.clear()
    user_rating_counts.clear()
    user_minutes_watched.clear()
    recommendations.clear()
    MODEL_KNOWN_USERS.clear()
    yield


# ------------------ UNIT TESTS -----------------------

def test_parse_time_valid_formats():
    """Ensure different time formats parse to epoch correctly."""
    ts1 = "2025-10-26 17:25:57"
    ts2 = "Oct 26 2025 5:25PM"
    epoch1 = parse_time(ts1)
    epoch2 = parse_time(ts2)
    # Allow tolerance for timezone differences
    assert abs(epoch1 - epoch2) < 120


def test_parse_time_invalid_fallback():
    """Invalid timestamp returns current time."""
    now = int(time.time())
    ts = "not-a-timestamp"
    parsed = parse_time(ts)
    assert abs(parsed - now) < 3


def test_cohort_for_rec_known_user():
    """Known training users must be warm."""
    MODEL_KNOWN_USERS.add(99)
    assert cohort_for_rec(99) == "warm"


def test_cohort_for_rec_behavioral_thresholds():
    """Users with enough activity should be warm."""
    user_rating_counts[1] = 6
    user_minutes_watched[1] = 10
    assert cohort_for_rec(1) == "warm"

    user_rating_counts[2] = 0
    user_minutes_watched[2] = 45
    assert cohort_for_rec(2) == "warm"

    user_rating_counts[3] = 2
    user_minutes_watched[3] = 5
    assert cohort_for_rec(3) == "cold"


def test_evaluate_user_precision_recall_ndcg_hitrate():
    """Verify metric math for a simple case."""
    uid = 10
    recs = ["A", "B", "C"]
    rec_time = 100
    user_watched[uid] = [(120, "A"), (150, "D")]  # watched one recommended item
    evaluate_user(uid, recs, "all", rec_time, k=3)

    # Get last computed metrics
    p = np.mean(metrics_buffer["all"]["precision"])
    r = np.mean(metrics_buffer["all"]["recall"])
    n = np.mean(metrics_buffer["all"]["ndcg"])
    h = np.mean(metrics_buffer["all"]["hit_rate"])

    assert round(p, 2) == 0.33
    assert round(r, 2) == 0.5
    assert n > 0 and n <= 1
    assert h == 1.0


def test_evaluate_user_no_hits_returns_zero():
    """If user watches nothing after rec time, buffer remains unchanged."""
    uid = 11
    recs = ["X", "Y"]
    rec_time = 200
    user_watched[uid] = [(50, "A")]  # all before rec_time
    evaluate_user(uid, recs, "all", rec_time)
    if metrics_buffer["all"]["precision"]:
        assert metrics_buffer["all"]["precision"][-1] == 0
    else:
        assert True


def test_ctr_computation_within_window():
    """CTR counts only recs with watch within 30 min (1800s)."""
    now = time.time()
    uid = 1
    rec_time = now - 1000
    recommendations.append((rec_time, uid, ["M1", "M2"]))
    user_watched[uid] = [(now - 900, "M1")]  # within window
    ctrs = compute_ctr_by_cohort(now)
    assert "all" in ctrs
    ctr, total, clicked = ctrs["all"]
    assert clicked == 1 and total == 1 and round(ctr, 2) == 1.0


def test_ctr_computation_outside_window():
    """CTR excludes watches outside 30 min window."""
    now = time.time()
    uid = 2
    rec_time = now - 4000
    recommendations.append((rec_time, uid, ["M1", "M2"]))
    user_watched[uid] = [(now - 2000, "M1")]  # too old
    ctrs = compute_ctr_by_cohort(now)
    assert ctrs["all"][0] == 0.0


def test_diversity_and_coverage_basic():
    """Ensure diversity and coverage values are computed correctly."""
    uid = 1
    recs1 = ["A", "B", "C"]
    recs2 = ["A", "B", "D"]
    recommendations.append((time.time(), uid, recs1))
    recommendations.append((time.time(), uid, recs2))
    out = diversity_coverage_by_cohort()
    div, cov = out["all"]
    assert 0 < div <= 1
    assert 0 < cov <= 1


def test_aggregate_metrics_all_output_structure():
    """Ensure output JSON has expected keys and proper types."""
    for c in ["all", "warm", "cold"]:
        metrics_buffer[c]["precision"].append(0.5)
        metrics_buffer[c]["recall"].append(0.7)
        metrics_buffer[c]["ndcg"].append(0.6)
        metrics_buffer[c]["hit_rate"].append(1.0)
        metrics_buffer[c]["ctr"].append(0.1)
        metrics_buffer[c]["diversity"].append(0.3)
        metrics_buffer[c]["coverage"].append(0.2)
        metrics_buffer[c]["inference_time"].extend([100, 120])

    result = aggregate_metrics_all()
    for c in ["all", "warm", "cold"]:
        assert set(result[c].keys()) == {
            "precision@K",
            "recall@K",
            "ndcg@K",
            "hit_rate@K",
            "ctr",
            "diversity",
            "coverage",
            "avg_inference_time_ms",
            "std_inference_time_ms",
        }
        assert isinstance(result[c]["precision@K"], float)
        assert isinstance(result[c]["avg_inference_time_ms"], float)
    assert "timestamp" in result
    assert "users_tracked" in result


def test_process_line_recommendation_updates_metrics(monkeypatch):
    """Mock recommendation log line and verify metrics update."""
    test_line = "2025-10-26 17:25:57,42,recommendation request ... result: M1, M2, 120 ms"

    monkeypatch.setattr("online_eval_module.cohort_for_rec", lambda x: "warm")

    process_line(test_line)
    assert len(metrics_buffer["warm"]["inference_time"]) == 1
    assert isinstance(metrics_buffer["warm"]["inference_time"][0], int)


def test_process_line_watch_and_rating_events():
    """Ensure watch and rating lines correctly update state."""
    watch_line = "2025-10-26 17:25:57,5,GET /data/m/M123/25.mpg"
    rating_line = "2025-10-26 17:25:57,5,GET /rate/M123=4"

    process_line(watch_line)
    process_line(rating_line)

    assert len(user_watched[5]) == 1
    assert len(user_rated[5]) == 1
    assert user_rating_counts[5] == 1
    assert user_minutes_watched[5] == 1


def test_invalid_line_does_not_crash():
    """Unknown log formats should be safely ignored."""
    bad_line = "This is a malformed log line"
    try:
        process_line(bad_line)
    except Exception as e:
        pytest.fail(f"process_line raised unexpected error: {e}")


def test_aggregate_metrics_all_avg_rating_computation(monkeypatch):
    """Validate avg_rating calculation."""
    monkeypatch.setattr("online_eval_module.rating_sum", 40)
    monkeypatch.setattr("online_eval_module.rating_count", 10)

    for c in ["all", "warm", "cold"]:
        metrics_buffer[c]["precision"].append(1.0)

    result = aggregate_metrics_all()
    assert result["avg_rating"] == 4.0
    assert isinstance(result["timestamp"], str)


# ---------------- PERSISTENCE TESTS ------------------

def test_persist_metrics_creates_json_and_csv(tmp_path):
    """Ensure persist_metrics() writes to both JSON and CSV files."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    metrics = {
        "timestamp": "2025-10-27 15:00:00",
        "all": {"precision@K": 0.1, "recall@K": 0.2, "ctr": 0.05, "diversity": 0.3, "coverage": 0.1},
        "warm": {"precision@K": 0.15, "recall@K": 0.25, "ctr": 0.06, "diversity": 0.31, "coverage": 0.11},
        "cold": {"precision@K": 0.12, "recall@K": 0.22, "ctr": 0.04, "diversity": 0.29, "coverage": 0.09},
        "avg_rating": 3.8,
        "users_tracked": 10
    }

    cwd = os.getcwd()
    os.chdir(metrics_dir.parent)

    persist_metrics(metrics)

    json_path = Path("metrics/online_evaluation_history.json")
    csv_path = Path("metrics/online_evaluation_history.csv")
    assert json_path.exists()
    assert csv_path.exists()

    data = json.loads(json_path.read_text())
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["all"]["precision@K"] == 0.1

    csv_lines = csv_path.read_text().strip().splitlines()
    assert len(csv_lines) >= 2

    os.chdir(cwd)


# ------------- PARSER REVIEWABILITY TEST -------------


def test_process_line_parsers_with_canned_logs():
    """Validate regex-based parsing with canned logs."""
    rec_line = "2025-10-26 17:25:57,42,recommendation request ... result: M1, M2, 150 ms"
    process_line(rec_line)
    assert any(metrics_buffer[c]["inference_time"] for c in ["all", "warm", "cold"])

    watch_line = "2025-10-26 17:30:00,42,GET /data/m/M1/25.mpg"
    process_line(watch_line)
    assert 42 in user_watched

    rating_line = "2025-10-26 17:32:00,42,GET /rate/M1=5"
    process_line(rating_line)
    assert user_rating_counts[42] == 1
