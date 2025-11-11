"""
Unit tests for content_based.py recommendation system.

Tests cover:
- Data loading and preprocessing
- Model training and saving
- Recommendation generation
- Cold-start user handling
- Edge cases and error handling
"""

import os
import pickle
import tempfile
import shutil
from pathlib import Path
from importlib.machinery import SourceFileLoader

import pytest
import pandas as pd
import numpy as np

# Load the content_based module dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTENT_BASED_PATH = PROJECT_ROOT / "app" / "model" / "src" / "content_based.py"
content_based = SourceFileLoader("content_based_module", str(CONTENT_BASED_PATH)).load_module()

# Get fixtures path
FIXTURES_PATH = Path(__file__).parent / 'fixtures'


# ============================================================================
# Tests for read_data function
# ============================================================================

def test_read_data_correct_number_of_rows():
    """Test that read_data reads the correct number of rows from fixtures"""
    movies, users, ratings, watches = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    # Check that data was loaded
    assert isinstance(movies, pd.DataFrame)
    assert isinstance(users, pd.DataFrame)
    assert isinstance(ratings, pd.DataFrame)
    assert isinstance(watches, pd.DataFrame)
    
    # Check that we got some data
    assert len(movies) > 0, "Movies dataframe should not be empty"
    assert len(users) > 0, "Users dataframe should not be empty"
    assert len(ratings) > 0, "Ratings dataframe should not be empty"
    assert len(watches) > 0, "Watches dataframe should not be empty"


def test_read_data_has_required_columns():
    """Test that read_data returns dataframes with required columns"""
    movies, users, ratings, watches = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    # Check movies columns
    required_movie_cols = ['movie_id', 'title', 'adult', 'budget', 'genres', 
                          'original_language', 'overview', 'popularity', 
                          'production_companies', 'production_countries', 
                          'release_date', 'revenue', 'runtime', 'vote_average', 
                          'vote_count', 'release_year']
    for col in required_movie_cols:
        assert col in movies.columns, f"Movies missing column: {col}"
    
    # Check users columns
    assert 'user_id' in users.columns
    assert 'age' in users.columns
    assert 'gender' in users.columns
    
    # Check ratings columns
    assert 'user_id' in ratings.columns
    assert 'movie_id' in ratings.columns
    assert 'rating' in ratings.columns
    assert 'timestamp' in ratings.columns
    
    # Check watches columns
    assert 'user_id' in watches.columns
    assert 'movie_id' in watches.columns
    assert 'timestamp_end' in watches.columns
    assert 'minutes_watched' in watches.columns


def test_read_data_genres_processed():
    """Test that genres are properly processed into lists"""
    movies, _, _, _ = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    # Check that genres column exists and contains lists
    assert 'genres' in movies.columns
    
    # Check at least one non-null genre entry
    non_null_genres = movies['genres'].dropna()
    if len(non_null_genres) > 0:
        first_genre = non_null_genres.iloc[0]
        assert isinstance(first_genre, list), "Genres should be a list"


def test_read_data_numeric_features_are_numeric():
    """Test that numeric features are properly converted to numeric types"""
    movies, _, _, _ = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    numeric_features = ['budget', 'popularity', 'revenue', 'runtime', 
                       'vote_average', 'vote_count']
    
    for feature in numeric_features:
        # Check that the column dtype is numeric (allowing for NaN)
        assert pd.api.types.is_numeric_dtype(movies[feature]), \
            f"{feature} should be numeric type"


def test_read_data_adult_is_boolean():
    """Test that adult column is converted to boolean"""
    movies, _, _, _ = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    assert pd.api.types.is_bool_dtype(movies['adult']) or \
           movies['adult'].dtype == bool


def test_read_data_invalid_path_raises_error():
    """Test that read_data raises an error for invalid path"""
    try:
        content_based.read_data('nonexistent_path/')
        assert False, "Should have raised an exception"
    except Exception:
        pass  # Expected


# ============================================================================
# Tests for train_model_full_data function
# ============================================================================

def test_train_model_outputs_file():
    """Test that train_model_full_data creates a model file"""
    movies, users, ratings, watches = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'test_model.pkl')
        
        content_based.train_model_full_data(
            movies, 
            users, 
            ratings, 
            watches, 
            path=output_path
        )
        
        # Check that file was created
        assert os.path.exists(output_path), "Model file should be created"
        
        # Check that file is not empty
        assert os.path.getsize(output_path) > 0, "Model file should not be empty"


def test_trained_model_can_be_loaded():
    """Test that the trained model can be loaded and used"""
    movies, users, ratings, watches = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'test_model_loadable.pkl')
        
        content_based.train_model_full_data(
            movies, 
            users, 
            ratings, 
            watches, 
            path=output_path
        )
        
        # Load the model
        with open(output_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Check that it's a ContentBasedRecommender
        assert isinstance(loaded_model, content_based.ContentBasedRecommender)
        
        # Check that it has required attributes
        assert loaded_model.user_profiles is not None
        assert loaded_model.group_profiles is not None


def test_trained_model_has_group_profiles():
    """Test that the trained model has group profiles for cold start"""
    movies, users, ratings, watches = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'test_model_groups.pkl')
        
        content_based.train_model_full_data(
            movies, 
            users, 
            ratings, 
            watches, 
            path=output_path
        )
        
        with open(output_path, 'rb') as f:
            model = pickle.load(f)
        
        # Check that group_profiles exists and has Unknown key
        assert model.group_profiles is not None
        assert 'Unknown' in model.group_profiles
        assert -1 in model.group_profiles['Unknown']


# ============================================================================
# Tests for ContentBasedRecommender class
# ============================================================================

@pytest.fixture(scope='module')
def trained_model():
    """Fixture that creates a trained model for testing"""
    movies, users, ratings, watches = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    model = content_based.ContentBasedRecommender()
    model.fit(movies, users, ratings, watches)
    
    return {
        'model': model,
        'movies': movies,
        'users': users,
        'ratings': ratings,
        'watches': watches
    }


def test_get_recommendations_for_user_in_profiles(trained_model):
    """Test recommendations for a user in user_profiles"""
    model = trained_model['model']
    movies = trained_model['movies']
    
    # Get a user that should be in user_profiles
    if len(model.user_profiles) > 0:
        user_id = list(model.user_profiles.keys())[0]
        
        recommendations, inference_time, metadata = model.get_recommendations(user_id, top_n=10)
        
        # Check that we got recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) == 10
        assert inference_time > 0
        
        # Check that recommendations are valid movie IDs
        for movie_id in recommendations:
            assert movie_id in movies['movie_id'].values
    else:
        pytest.skip("No users in user_profiles")


def test_get_recommendations_for_user_not_in_profiles(trained_model):
    """Test recommendations for cold-start user (not in user_profiles)"""
    model = trained_model['model']
    users = trained_model['users']
    
    # Find a user_id that's in all_users but not in user_profiles
    all_user_ids = set(users['user_id'].values)
    profile_user_ids = set(model.user_profiles.keys())
    cold_start_users = all_user_ids - profile_user_ids
    
    if len(cold_start_users) > 0:
        user_id = list(cold_start_users)[0]
        
        recommendations, inference_time, metadata = model.get_recommendations(user_id, top_n=10)
        
        # Check that we got recommendations from group_profiles
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert len(recommendations) <= 10
        assert inference_time > 0
    else:
        pytest.skip("No cold start users available")


def test_get_recommendations_respects_top_n(trained_model):
    """Test that get_recommendations respects top_n parameter"""
    model = trained_model['model']
    
    if len(model.user_profiles) > 0:
        user_id = list(model.user_profiles.keys())[0]
        
        for n in [5, 10, 20]:
            recommendations, _, _ = model.get_recommendations(user_id, top_n=n)
            assert len(recommendations) == n
    else:
        pytest.skip("No users in user_profiles")


def test_get_recommendations_for_negative_user_id(trained_model):
    """Test recommendations for negative user ID (should use Unknown group)"""
    model = trained_model['model']
    movies = trained_model['movies']
    
    user_id = -999
    
    recommendations, inference_time, metadata = model.get_recommendations(user_id, top_n=10)
    
    # Should still get recommendations from Unknown group
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert len(recommendations) <= 10
    assert inference_time > 0
    
    # Check that recommendations are valid
    for movie_id in recommendations:
        assert movie_id in movies['movie_id'].values


def test_get_recommendations_for_very_large_user_id(trained_model):
    """Test recommendations for very large user ID (should use Unknown group)"""
    model = trained_model['model']
    movies = trained_model['movies']
    
    user_id = 100000000000000000
    
    recommendations, inference_time, metadata = model.get_recommendations(user_id, top_n=10)
    
    # Should still get recommendations from Unknown group
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert len(recommendations) <= 10
    assert inference_time > 0
    
    # Check that recommendations are valid
    for movie_id in recommendations:
        assert movie_id in movies['movie_id'].values


def test_model_has_unknown_group_profile(trained_model):
    """Test that model has Unknown group profile for fallback"""
    model = trained_model['model']
    
    assert 'Unknown' in model.group_profiles
    assert -1 in model.group_profiles['Unknown']
    
    # Check that Unknown profile has recommendations
    unknown_recs = model.group_profiles['Unknown'][-1]
    assert isinstance(unknown_recs, np.ndarray)
    assert len(unknown_recs) > 0


def test_get_model_size(trained_model):
    """Test that get_model_size returns a positive value"""
    model = trained_model['model']
    
    model_size = model.get_model_size()
    
    assert isinstance(model_size, int)
    assert model_size > 0


def test_model_training_time_recorded(trained_model):
    """Test that training time is recorded"""
    model = trained_model['model']
    
    assert hasattr(model, 'training_time')
    assert model.training_time > 0


def test_recommendations_top_n_parameter(trained_model):
    """Test that top_n parameter works correctly"""
    model = trained_model['model']
    
    user_id = list(model.user_profiles.keys())[0] if model.user_profiles else -999
    
    for n in [5, 10, 20]:
        recommendations, _, _ = model.get_recommendations(user_id, top_n=n)
        assert len(recommendations) <= n


def test_movie_profiles_shape(trained_model):
    """Test that movie profiles have correct shape"""
    model = trained_model['model']
    movies = trained_model['movies']
    
    assert model.movie_profiles is not None
    assert model.movie_profiles.shape[0] == len(movies)
    assert model.movie_profiles.shape[1] > 0


def test_user_profiles_not_empty(trained_model):
    """Test that some user profiles were created"""
    model = trained_model['model']
    
    assert model.user_profiles is not None
    assert len(model.user_profiles) > 0


# ============================================================================
# Edge cases and error handling
# ============================================================================

def test_model_with_empty_movies_raises_error():
    """Test that empty movies dataframe raises an error"""
    _, users, ratings, watches = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    model = content_based.ContentBasedRecommender()
    empty_movies = pd.DataFrame(columns=['movie_id', 'title', 'adult', 'budget', 'genres', 
                                         'original_language', 'overview', 'popularity', 
                                         'production_companies', 'production_countries', 
                                         'release_date', 'revenue', 'runtime', 'vote_average', 
                                         'vote_count', 'release_year'])
    
    try:
        model.fit(empty_movies, users, ratings, watches)
        assert False, "Should have raised ValueError for empty movies"
    except ValueError as e:
        assert "movies_df cannot be empty" in str(e)


def test_model_with_empty_users_raises_error():
    """Test that empty users dataframe raises an error"""
    movies, _, ratings, watches = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    model = content_based.ContentBasedRecommender()
    empty_users = pd.DataFrame(columns=['user_id', 'age', 'gender'])
    
    try:
        model.fit(movies, empty_users, ratings, watches)
        assert False, "Should have raised ValueError for empty users"
    except ValueError as e:
        assert "users_df cannot be empty" in str(e)


def test_model_with_empty_ratings_and_watches_raises_error():
    """Test that empty ratings AND watches raises an error"""
    movies, users, _, _ = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    model = content_based.ContentBasedRecommender()
    empty_ratings = pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'timestamp'])
    empty_watches = pd.DataFrame(columns=['user_id', 'movie_id', 'timestamp_end', 'minutes_watched'])
    
    try:
        model.fit(movies, users, empty_ratings, empty_watches)
        assert False, "Should have raised ValueError for empty ratings and watches"
    except ValueError as e:
        assert "At least one of ratings_df or watches_df must be non-empty" in str(e)


def test_model_with_empty_ratings():
    """Test model behavior with empty ratings but valid watches"""
    movies, users, _, watches = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    model = content_based.ContentBasedRecommender()
    empty_ratings = pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    # Should succeed with watches data
    model.fit(movies, users, empty_ratings, watches)
    
    assert model.movie_profiles is not None
    assert model.group_profiles is not None


def test_model_with_empty_watches():
    """Test model behavior with empty watches but valid ratings"""
    movies, users, ratings, _ = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    model = content_based.ContentBasedRecommender()
    empty_watches = pd.DataFrame(columns=['user_id', 'movie_id', 'timestamp_end', 'minutes_watched'])
    
    # Should succeed with ratings data
    model.fit(movies, users, ratings, empty_watches)
    
    assert model.movie_profiles is not None
    assert model.group_profiles is not None


def test_recommendations_consistency():
    """Test that recommendations are consistent across calls"""
    movies, users, ratings, watches = content_based.read_data(str(FIXTURES_PATH) + '/')
    
    model = content_based.ContentBasedRecommender()
    model.fit(movies, users, ratings, watches)
    
    user_id = -999  # Unknown user
    
    recs1, _, _ = model.get_recommendations(user_id, top_n=10)
    recs2, _, _ = model.get_recommendations(user_id, top_n=10)
    
    # Should get same recommendations for same user
    assert recs1 == recs2
