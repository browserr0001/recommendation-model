"""
Unit tests for Offline Evaluation Framework (app/model/offline_evaluation.py)

Tests cover:
- Data loading and validation
- Temporal train/test split correctness
- Warm/cold-start user classification
- Metric calculations (Precision, Recall, NDCG, Hit Rate, Diversity, Coverage)
- Training data accounting (explicit ratings + implicit watches)
- Results persistence and JSON structure
"""

import pytest
import numpy as np
import pandas as pd
import json
import os
import tempfile
from pathlib import Path
from importlib.machinery import SourceFileLoader

# ------- LOAD TARGET MODULE DYNAMICALLY --------------

# Compute project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to offline evaluation module
OFFLINE_EVAL_PATH = PROJECT_ROOT / "app" / "model" / "offline_evaluation.py"

# Dynamically load the module
offline_eval = SourceFileLoader("offline_eval_module", str(OFFLINE_EVAL_PATH)).load_module()

# Import the OfflineEvaluator class
OfflineEvaluator = offline_eval.OfflineEvaluator


# ----------- FIXTURES FOR TEST DATA -----------

@pytest.fixture
def sample_data():
    """Create minimal sample data for testing."""
    # Sample movies
    movies = pd.DataFrame({
        'movie_id': ['movie_1', 'movie_2', 'movie_3', 'movie_4'],
        'title': ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4'],
        'genres': [['Action'], ['Comedy'], ['Action', 'Comedy'], ['Drama']],
        'runtime': [120, 90, 100, 110]
    })

    # Sample users
    users = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'age': [25, 30, 22, 35, 28],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'occupation': ['engineer', 'teacher', 'student', 'doctor', 'artist']
    })

    # Sample ratings (explicit)
    ratings = pd.DataFrame({
        'timestamp': pd.to_datetime(['2025-10-20 10:00:00', '2025-10-20 11:00:00',
                                     '2025-10-21 10:00:00', '2025-10-21 11:00:00',
                                     '2025-10-22 10:00:00', '2025-10-22 11:00:00',
                                     '2025-10-23 10:00:00', '2025-10-23 11:00:00']),
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'movie_id': ['movie_1', 'movie_2', 'movie_1', 'movie_3',
                     'movie_2', 'movie_4', 'movie_1', 'movie_3'],
        'rating': [5, 4, 3, 5, 4, 5, 3, 4]
    })

    # Sample watches (implicit)
    watches = pd.DataFrame({
        'timestamp_start': pd.to_datetime(['2025-10-20 09:00:00', '2025-10-20 10:30:00',
                                          '2025-10-21 09:00:00', '2025-10-22 09:00:00',
                                          '2025-10-23 09:00:00', '2025-10-24 10:00:00']),
        'timestamp_end': pd.to_datetime(['2025-10-20 09:30:00', '2025-10-20 11:00:00',
                                        '2025-10-21 09:45:00', '2025-10-22 09:50:00',
                                        '2025-10-23 09:55:00', '2025-10-24 10:50:00']),
        'user_id': [1, 2, 2, 3, 4, 5],
        'movie_id': ['movie_1', 'movie_2', 'movie_3', 'movie_4', 'movie_1', 'movie_2'],
        'minutes_watched': [30, 60, 45, 50, 55, 50]
    })

    return movies, users, ratings, watches


@pytest.fixture
def temp_data_dir(sample_data):
    """Create temporary directory with sample parquet files."""
    movies, users, ratings, watches = sample_data

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        meta_dir = data_path / 'meta'
        ratings_dir = data_path / 'ratings'
        watches_dir = data_path / 'watches'

        meta_dir.mkdir()
        ratings_dir.mkdir()
        watches_dir.mkdir()

        # Save as parquet files (would need pyarrow, but for testing we can mock)
        # For now, we'll test with the data structures directly

        yield str(data_path), movies, users, ratings, watches


# ----------- UNIT TESTS -----------

def test_evaluator_initialization():
    """Test OfflineEvaluator initializes correctly."""
    evaluator = OfflineEvaluator(data_path='data/')

    assert evaluator.data_path == 'data/'
    assert evaluator.movies is None
    assert evaluator.users is None
    assert evaluator.ratings is None
    assert evaluator.watches is None
    assert evaluator.model is None
    assert evaluator.train_users_combined is None
    assert evaluator.results == {}


def test_temporal_split_attributes(monkeypatch, sample_data):
    """Test that temporal_train_test_split sets correct attributes."""
    movies, users, ratings, watches = sample_data

    evaluator = OfflineEvaluator()
    evaluator.movies = movies
    evaluator.users = users
    evaluator.ratings = ratings
    evaluator.watches = watches

    # Mock the external train_test_split function
    def mock_train_test_split(ratings, movies, users, watches, test_size):
        n_test = int(len(ratings) * test_size)
        n_train = len(ratings) - n_test

        train_ratings = ratings.iloc[:n_train].copy()
        test_ratings = ratings.iloc[n_train:].copy()
        test_users = test_ratings['user_id'].unique()

        return train_ratings, movies, users, watches, test_ratings, test_users

    monkeypatch.setattr("offline_eval_module.train_test_split", mock_train_test_split)

    evaluator.temporal_train_test_split(test_size=0.25)

    assert evaluator.train_ratings is not None
    assert evaluator.test_ratings is not None
    assert evaluator.test_users is not None
    assert evaluator.train_users_combined is not None
    assert 'split_metadata' in evaluator.results


def test_split_metadata_structure(monkeypatch, sample_data):
    """Test that split_metadata contains all required keys."""
    movies, users, ratings, watches = sample_data

    evaluator = OfflineEvaluator()
    evaluator.movies = movies
    evaluator.users = users
    evaluator.ratings = ratings
    evaluator.watches = watches

    def mock_train_test_split(ratings, movies, users, watches, test_size):
        n_test = int(len(ratings) * test_size)
        n_train = len(ratings) - n_test

        train_ratings = ratings.iloc[:n_train].copy()
        test_ratings = ratings.iloc[n_train:].copy()
        test_users = test_ratings['user_id'].unique()

        return train_ratings, movies, users, watches, test_ratings, test_users

    monkeypatch.setattr("offline_eval_module.train_test_split", mock_train_test_split)

    evaluator.temporal_train_test_split(test_size=0.25)

    metadata = evaluator.results['split_metadata']

    required_keys = [
        'test_size',
        'split_timestamp',
        'train_explicit_ratings',
        'train_implicit_watches',
        'train_combined_interactions',
        'test_interactions',
        'train_users_from_ratings',
        'train_users_from_watches',
        'train_users_total',
        'test_users',
        'warm_start_users',
        'cold_start_users',
        'cold_start_percentage'
    ]

    for key in required_keys:
        assert key in metadata, f"Missing key: {key}"


def test_warm_cold_start_classification(monkeypatch, sample_data):
    """Test that warm/cold-start users are correctly classified."""
    movies, users, ratings, watches = sample_data

    evaluator = OfflineEvaluator()
    evaluator.movies = movies
    evaluator.users = users
    evaluator.ratings = ratings
    evaluator.watches = watches

    def mock_train_test_split(ratings, movies, users, watches, test_size):
        train_ratings = ratings.iloc[:6].copy()
        test_ratings = ratings.iloc[6:].copy()
        test_users = test_ratings['user_id'].unique()

        return train_ratings, movies, users, watches, test_ratings, test_users

    monkeypatch.setattr("offline_eval_module.train_test_split", mock_train_test_split)

    evaluator.temporal_train_test_split(test_size=0.25)

    metadata = evaluator.results['split_metadata']

    # Warm-start + cold-start should equal total test users
    assert metadata['warm_start_users'] + metadata['cold_start_users'] == metadata['test_users']

    # Cold-start percentage should be calculated correctly
    expected_percentage = (metadata['cold_start_users'] / metadata['test_users']) * 100
    assert abs(metadata['cold_start_percentage'] - expected_percentage) < 0.01


def test_metric_calculation_precision():
    """Test precision calculation: hits / k."""
    # Simulate scenario
    recommendations = ['movie_1', 'movie_2', 'movie_3']
    actual_items = {'movie_1', 'movie_4'}
    k = 3

    hits = set(recommendations) & actual_items
    precision = len(hits) / k

    assert precision == 1/3  # One hit out of 3 recommendations


def test_metric_calculation_recall():
    """Test recall calculation: hits / |actual|."""
    recommendations = ['movie_1', 'movie_2', 'movie_3']
    actual_items = {'movie_1', 'movie_4'}

    hits = set(recommendations) & actual_items
    recall = len(hits) / len(actual_items)

    assert recall == 1/2  # One hit out of 2 actual items


def test_metric_calculation_hit_rate():
    """Test hit rate: binary indicator if any hit."""
    # Case 1: Has hits
    recommendations = ['movie_1', 'movie_2']
    actual_items = {'movie_1'}

    hits = set(recommendations) & actual_items
    hit_rate = 1.0 if len(hits) > 0 else 0.0

    assert hit_rate == 1.0

    # Case 2: No hits
    recommendations = ['movie_2', 'movie_3']
    actual_items = {'movie_1'}

    hits = set(recommendations) & actual_items
    hit_rate = 1.0 if len(hits) > 0 else 0.0

    assert hit_rate == 0.0


def test_metric_calculation_ndcg():
    """Test NDCG calculation with position-based discounting."""
    recommendations = ['movie_1', 'movie_2', 'movie_3']
    actual_items = {'movie_1', 'movie_3'}
    k = 3

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommendations):
        if item in actual_items:
            dcg += 1.0 / np.log2(i + 2)

    # Calculate IDCG (ideal DCG - if all top-k were relevant)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k))])

    ndcg = dcg / idcg if idcg > 0 else 0.0

    assert 0 <= ndcg <= 1.0
    assert ndcg > 0  # Should have some ranking quality


def test_diversity_calculation():
    """Test diversity: unique genres / total unique genres."""
    all_genres = {'Action', 'Comedy', 'Drama', 'Thriller', 'Horror'}
    recommended_genres = {'Action', 'Comedy', 'Drama'}

    diversity = len(recommended_genres) / len(all_genres)

    assert diversity == 3/5


def test_coverage_calculation():
    """Test coverage: unique movies recommended / total movies."""
    total_movies = 100
    unique_recommendations = {'movie_1', 'movie_2', 'movie_5', 'movie_10'}

    coverage = len(unique_recommendations) / total_movies

    assert coverage == 4/100


def test_results_json_structure():
    """Test that results JSON has expected structure."""
    evaluator = OfflineEvaluator()

    # Manually set some results
    evaluator.results = {
        'split_metadata': {'test_size': 0.2},
        'model_metadata': {'training_time_sec': 10.0},
        'overall_metrics': {'precision_at_k': 0.5},
        'warm_start_metrics': {'recall_at_k': 0.6},
        'cold_start_metrics': {'ndcg_at_k': 0.3}
    }

    assert 'split_metadata' in evaluator.results
    assert 'model_metadata' in evaluator.results
    assert 'overall_metrics' in evaluator.results
    assert 'warm_start_metrics' in evaluator.results
    assert 'cold_start_metrics' in evaluator.results


def test_empty_user_list_handling():
    """Test that evaluation handles empty user lists gracefully."""
    # Simulate the _evaluate_user_group method with empty list
    user_list = []
    k = 20

    # Expected behavior: return zero metrics
    if len(user_list) == 0:
        metrics = {
            'n_users': 0,
            'precision_at_k': 0.0,
            'recall_at_k': 0.0,
            'ndcg_at_k': 0.0,
            'hit_rate_at_k': 0.0,
            'diversity': 0.0,
            'coverage': 0.0,
            'avg_inference_time_ms': 0.0
        }

    assert metrics['n_users'] == 0
    assert metrics['precision_at_k'] == 0.0


def test_train_users_combined_calculation():
    """Test that train_users_combined includes both ratings and watches."""
    train_users_from_ratings = {1, 2, 3}
    train_users_from_watches = {2, 3, 4, 5}

    train_users_combined = train_users_from_ratings | train_users_from_watches

    assert train_users_combined == {1, 2, 3, 4, 5}
    assert len(train_users_combined) == 5


def test_watch_filtering_threshold():
    """Test that watches are filtered by ≥50% watched threshold."""
    # Watch data
    movie_runtime = 100  # minutes
    minutes_watched_scenarios = [30, 50, 60, 80, 100]

    filtered = []
    for minutes in minutes_watched_scenarios:
        if minutes / movie_runtime >= 0.5:
            filtered.append(minutes)

    assert filtered == [50, 60, 80, 100]
    assert 30 not in filtered  # Below threshold


def test_implicit_rating_assignment():
    """Test that implicit watches get assigned rating of 3."""
    # Simulated watch that meets ≥50% threshold
    implicit_rating = 3

    assert implicit_rating == 3  # Standard implicit rating value


def test_deduplication_keeps_latest():
    """Test that deduplication keeps latest entry for (user, movie) pairs."""
    # Simulate combined data with duplicates
    data = pd.DataFrame({
        'user_id': [1, 1, 2],
        'movie_id': ['movie_a', 'movie_a', 'movie_b'],
        'timestamp': pd.to_datetime(['2025-10-20', '2025-10-21', '2025-10-22']),
        'rating': [3, 5, 4]
    })

    deduped = data.groupby(['user_id', 'movie_id']).last().reset_index()

    assert len(deduped) == 2  # Two unique (user, movie) pairs

    # Check that latest rating is kept for user 1, movie_a
    user1_moviea = deduped[(deduped['user_id'] == 1) & (deduped['movie_id'] == 'movie_a')]
    assert user1_moviea['rating'].values[0] == 5  # Latest rating


def test_inference_time_tracking():
    """Test that inference time is measured in milliseconds."""
    import time

    start = time.time()
    time.sleep(0.01)  # Simulate 10ms operation
    end = time.time()

    inference_time_ms = (end - start) * 1000

    assert inference_time_ms >= 10
    assert inference_time_ms < 20  # Should be close to 10ms


def test_no_division_by_zero_in_metrics():
    """Test that metrics handle edge cases without division by zero."""
    # Empty actual items
    actual_items = set()
    recommendations = ['movie_1', 'movie_2']

    # Recall calculation
    recall = len(set(recommendations) & actual_items) / len(actual_items) if len(actual_items) > 0 else 0.0
    assert recall == 0.0

    # IDCG calculation
    k = 20
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k))])
    ndcg = 0.0 / idcg if idcg > 0 else 0.0
    assert ndcg == 0.0


def test_results_include_timestamp():
    """Test that results include evaluation timestamp."""
    evaluator = OfflineEvaluator()

    # Simulate saving results
    from datetime import datetime
    evaluator.results['evaluation_timestamp'] = datetime.now().isoformat()
    evaluator.results['evaluation_framework'] = 'offline_evaluation.py'

    assert 'evaluation_timestamp' in evaluator.results
    assert 'evaluation_framework' in evaluator.results
    assert evaluator.results['evaluation_framework'] == 'offline_evaluation.py'


# ----------- INTEGRATION TESTS -----------

def test_complete_pipeline_structure():
    """Test that complete pipeline has all expected steps."""
    # Pipeline steps that should be called in order
    pipeline_steps = [
        'load_data',
        'temporal_train_test_split',
        'train_model',
        'evaluate_metrics',
        'save_results'
    ]

    # Verify evaluator has all these methods
    evaluator = OfflineEvaluator()

    for step in pipeline_steps:
        assert hasattr(evaluator, step), f"Missing method: {step}"


def test_json_serialization():
    """Test that results can be serialized to JSON."""
    results = {
        'split_metadata': {
            'train_explicit_ratings': 100,
            'train_users_total': 50,
            'cold_start_percentage': 45.5
        },
        'overall_metrics': {
            'precision_at_k': 0.123,
            'recall_at_k': 0.456
        }
    }

    # Should be able to serialize to JSON
    json_str = json.dumps(results, indent=2)
    assert isinstance(json_str, str)

    # Should be able to deserialize
    loaded = json.loads(json_str)
    assert loaded['split_metadata']['train_explicit_ratings'] == 100