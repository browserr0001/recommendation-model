import numpy as np
import pandas as pd
import pickle
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from content_based import ContentBasedRecommender, read_data


def train_test_split(ratings_df, movies_df, users_df, watches_df, test_size=0.2, mid_rating_watch_over=0.5):
    """
    Split the ratings data and watches data into training and test sets based on timestamp.
    For movies and users, include only those present in the training set.

    Parameters:
        ratings_df: DataFrame with user ratings
        movies_df: DataFrame with movie metadata
        users_df: DataFrame with user metadata
        watches_df: DataFrame with user watch history
        test_size: Proportion of data to use for testing
        mid_rating_watch_over: Threshold for considering a watch as a rating (not used in current implementation)

    Returns:
        train_ratings: Training set of ratings
        train_movies: Movies present in the training set
        train_users: Users present in the training set
        train_watches: Training set of watches
        test_ratings: Test set of ratings (includes watches)
        test_users: Users present in the test set
    """
    # Get the split timestamp based on percentage of all timestamps
    ratings_df = ratings_df.sort_values('timestamp').reset_index(drop=True)
    n_total = len(ratings_df)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test
    train_ratings = ratings_df.iloc[:n_train].reset_index(drop=True)
    test_ratings = ratings_df.iloc[n_train:].reset_index(drop=True)

    split_timestamp = ratings_df.iloc[n_train]['timestamp']
    print(f"Train-test Split timestamp: {split_timestamp}")

    # Get users and movies in the training set
    train_user_ids = train_ratings['user_id'].unique()
    train_movie_ids = train_ratings['movie_id'].unique()

    train_users = users_df[users_df['user_id'].isin(train_user_ids)].reset_index(drop=True)
    train_movies = movies_df[movies_df['movie_id'].isin(train_movie_ids)].reset_index(drop=True)

    train_watches = watches_df[watches_df['timestamp_start'] <= split_timestamp].reset_index(drop=True)
    test_watches = watches_df[watches_df['timestamp_start'] > split_timestamp].reset_index(drop=True)

    test_watch_subset = test_watches[['timestamp_start', 'user_id', 'movie_id']]
    test_ratings = pd.concat([test_watch_subset, test_ratings]).groupby(['user_id', 'movie_id']).last().reset_index()
    test_users = test_ratings['user_id'].unique()
    print(f"Train ratings: {train_ratings.shape}, Test ratings: {test_ratings.shape}")
    print(f"Train users: {train_users.shape}, Train movies: {train_movies.shape}")

    return train_ratings, train_movies, train_users, train_watches, test_ratings, test_users


class OfflineEvaluator:
    """
    Comprehensive offline evaluation framework for recommendation models.

    Addresses common evaluation pitfalls:
    1. Temporal ordering: Uses time-based train/test split
    2. Data leakage prevention: No future data in training
    3. Cold-start evaluation: Separate metrics for known vs unknown users
    4. Realistic metrics: Multiple complementary metrics
    """

    def __init__(self, data_path: str = 'data/'):
        """
        Initialize evaluator with data path.

        Args:
            data_path: Path to the data directory containing parquet files
        """
        self.data_path = data_path
        self.movies = None
        self.users = None
        self.ratings = None
        self.watches = None
        self.model = None

        # Split data
        self.train_ratings = None
        self.train_movies = None
        self.train_users = None
        self.train_watches = None
        self.test_ratings = None
        self.test_users = None
        self.train_users_combined = None

        # Results
        self.results = {}

    def load_data(self):
        """Load all data from parquet files."""
        print("Loading data...")
        self.movies, self.users, self.ratings, self.watches = read_data(self.data_path)

        print(f"Loaded {len(self.movies)} movies")
        print(f"Loaded {len(self.users)} users")
        print(f"Loaded {len(self.ratings)} ratings")
        print(f"Loaded {len(self.watches)} watch records")

    def temporal_train_test_split(self, test_size: float = 0.2):
        """
        Perform temporal train/test split to avoid data leakage.

        STRATEGY TO AVOID PITFALLS:
        1. Temporal ordering: Split by timestamp, not random
        2. No data leakage: All test interactions happen AFTER training period
        3. Realistic cold-start: Some test users may not be in training set

        Args:
            test_size: Proportion of data for testing (default 0.2 = 20%)
        """
        print(f"\n{'='*80}")
        print("TEMPORAL TRAIN/TEST SPLIT")
        print(f"{'='*80}")

        # Use the existing train_test_split function
        self.train_ratings, self.train_movies, self.train_users, self.train_watches, \
        self.test_ratings, self.test_users = train_test_split(
            self.ratings, self.movies, self.users, self.watches,
            test_size=test_size
        )

        # Calculate split statistics
        split_timestamp = self.ratings.sort_values('timestamp').iloc[
            int(len(self.ratings) * (1 - test_size))
        ]['timestamp']

        # Simulate what model.fit() does to get actual training interactions
        percent_watched = pd.merge(
            self.train_watches[['movie_id', 'user_id', 'timestamp_end', 'minutes_watched']],
            self.train_movies[['movie_id', 'runtime']],
            on='movie_id'
        )
        percent_watched = percent_watched[percent_watched['minutes_watched']/percent_watched['runtime'] >= 0.5]
        percent_watched['rating'] = 3
        percent_watched.rename(columns={'timestamp_end': 'timestamp'}, inplace=True)
        watch_subset = percent_watched[['timestamp', 'user_id', 'movie_id', 'rating']]

        # Combine watches with explicit ratings
        combined_train = pd.concat([watch_subset, self.train_ratings])
        combined_train_dedup = combined_train.groupby(['user_id', 'movie_id']).last().reset_index()

        # Get all training users (from both ratings and watches)
        train_users_from_ratings = set(self.train_ratings['user_id'].unique())
        train_users_from_watches = set(percent_watched['user_id'].unique())
        train_users_combined = train_users_from_ratings | train_users_from_watches

        # Get test users
        test_user_set = set(self.test_users)

        # Calculate warm/cold start based on ALL training data (not just explicit ratings)
        warm_start_users = train_users_combined & test_user_set
        cold_start_users = test_user_set - train_users_combined

        print(f"\nSplit timestamp: {split_timestamp}")

        print(f"\nTrain set:")
        print(f"  - Explicit ratings: {len(self.train_ratings)}")
        print(f"  - Implicit watches (≥50% watched): {len(percent_watched)}")
        print(f"  - Combined interactions (after deduplication): {len(combined_train_dedup)}")
        print(f"  - Users (from explicit ratings only): {len(train_users_from_ratings)}")
        print(f"  - Users (from implicit watches): {len(train_users_from_watches)}")
        print(f"  - Total unique users (ratings OR watches): {len(train_users_combined)}")
        print(f"  - Movies: {len(self.train_ratings['movie_id'].unique())}")

        print(f"\nTest set:")
        print(f"  - Interactions (ratings + watches): {len(self.test_ratings)}")
        print(f"  - Users: {len(test_user_set)}")
        print(f"  - Movies: {len(self.test_ratings['movie_id'].unique())}")

        print(f"\nUser categorization:")
        print(f"  - Warm-start users (in both train & test): {len(warm_start_users)}")
        print(f"  - Cold-start users (only in test): {len(cold_start_users)}")
        print(f"  - Percentage cold-start: {len(cold_start_users)/len(test_user_set)*100:.2f}%")

        # Store metadata with corrected values
        self.results['split_metadata'] = {
            'test_size': test_size,
            'split_timestamp': str(split_timestamp),
            'train_explicit_ratings': len(self.train_ratings),
            'train_implicit_watches': len(percent_watched),
            'train_combined_interactions': len(combined_train_dedup),
            'test_interactions': len(self.test_ratings),
            'train_users_from_ratings': len(train_users_from_ratings),
            'train_users_from_watches': len(train_users_from_watches),
            'train_users_total': len(train_users_combined),
            'test_users': len(test_user_set),
            'warm_start_users': len(warm_start_users),
            'cold_start_users': len(cold_start_users),
            'cold_start_percentage': len(cold_start_users)/len(test_user_set)*100
        }

        # Store for use in evaluation
        self.train_users_combined = train_users_combined

    def train_model(self):
        """Train the recommendation model on training data."""
        print(f"\n{'='*80}")
        print("MODEL TRAINING")
        print(f"{'='*80}")

        self.model = ContentBasedRecommender()
        self.model.fit(
            self.train_movies,
            self.users,  # Use all users for cold-start handling
            self.train_ratings,
            self.train_watches
        )

        # Store model metadata
        model_size_bytes = self.model.get_model_size()
        self.results['model_metadata'] = {
            'training_time_sec': self.model.training_time,
            'model_size_mb': model_size_bytes / (1024 * 1024),
            'n_movie_features': self.model.num_total_features
        }

        print(f"\nModel trained successfully")
        print(f"  - Training time: {self.model.training_time:.2f} seconds")
        print(f"  - Model size: {model_size_bytes / (1024*1024):.2f} MB")

    def evaluate_metrics(self, k: int = 20) -> Dict:
        """
        Evaluate model using multiple complementary metrics.

        METRICS DEFINED (3-STEP FORMAT):

        1. PRECISION@K
           - Metric: Proportion of recommended items that user actually interacted with
           - Data: Top-k recommendations vs. actual test interactions
           - Operationalization: |recommended ∩ actual| / k

        2. RECALL@K
           - Metric: Proportion of user's actual items that were recommended
           - Data: Top-k recommendations vs. all test interactions
           - Operationalization: |recommended ∩ actual| / |actual|

        3. NDCG@K (Normalized Discounted Cumulative Gain)
           - Metric: Ranking quality with position-based discounting
           - Data: Position of relevant items in recommendation list
           - Operationalization: DCG@k / IDCG@k where DCG = Σ(1/log2(i+2)) for hits

        4. HIT RATE@K
           - Metric: Percentage of users with at least one hit in top-k
           - Data: Binary indicator if any recommendation matches actual
           - Operationalization: |users with ≥1 hit| / |users|

        5. DIVERSITY
           - Metric: Genre coverage in recommendations
           - Data: Unique genres across all recommendations
           - Operationalization: |unique genres in recs| / |total genres|

        6. COVERAGE
           - Metric: Catalog coverage - how many movies get recommended
           - Data: Unique movies recommended across all users
           - Operationalization: |unique movies recommended| / |total movies|

        Args:
            k: Number of recommendations to evaluate (default 20)

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*80}")
        print(f"OFFLINE EVALUATION (k={k})")
        print(f"{'='*80}")

        # Separate warm-start and cold-start users based on ACTUAL training data
        # Use train_users_combined which includes both ratings and watches
        test_user_set = set(self.test_users)
        warm_start_users = list(self.train_users_combined & test_user_set)
        cold_start_users = list(test_user_set - self.train_users_combined)

        print(f"\nUser classification (based on ALL training data):")
        print(f"  - Warm-start users: {len(warm_start_users)}")
        print(f"  - Cold-start users: {len(cold_start_users)}")

        # Evaluate both groups
        print("\nEvaluating warm-start users...")
        warm_metrics = self._evaluate_user_group(warm_start_users, k, "warm_start")

        print("\nEvaluating cold-start users...")
        cold_metrics = self._evaluate_user_group(cold_start_users, k, "cold_start")

        # Evaluate all users together
        print("\nEvaluating all users...")
        all_metrics = self._evaluate_user_group(list(test_user_set), k, "all_users")

        # Store results
        self.results['warm_start_metrics'] = warm_metrics
        self.results['cold_start_metrics'] = cold_metrics
        self.results['overall_metrics'] = all_metrics

        return {
            'warm_start': warm_metrics,
            'cold_start': cold_metrics,
            'overall': all_metrics
        }

    def _evaluate_user_group(self, user_list: List, k: int, group_name: str) -> Dict:
        """
        Evaluate a specific group of users.

        Args:
            user_list: List of user IDs to evaluate
            k: Number of recommendations
            group_name: Name of the group (for logging)

        Returns:
            Dictionary of metrics for this group
        """
        if len(user_list) == 0:
            return {
                'n_users': 0,
                'precision_at_k': 0.0,
                'recall_at_k': 0.0,
                'ndcg_at_k': 0.0,
                'hit_rate_at_k': 0.0,
                'diversity': 0.0,
                'coverage': 0.0,
                'avg_inference_time_ms': 0.0
            }

        # Initialize metric collectors
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        hit_rates = []
        diversity_scores = []
        inference_times = []
        all_recommendations = []

        # Evaluate each user
        for user_id in user_list:
            # Get test interactions for this user
            user_test = self.test_ratings[self.test_ratings['user_id'] == user_id]
            if len(user_test) == 0:
                continue

            actual_items = set(user_test['movie_id'].values)

            # Get recommendations with timing
            start_time = time.time()
            recommendations, _ = self.model.get_recommendations(user_id, top_n=k)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)

            all_recommendations.append(recommendations)

            # Calculate hits
            recommended_set = set(recommendations)
            hits = recommended_set & actual_items
            n_hits = len(hits)

            # Precision@K
            precision = n_hits / k if k > 0 else 0.0
            precision_scores.append(precision)

            # Recall@K
            recall = n_hits / len(actual_items) if len(actual_items) > 0 else 0.0
            recall_scores.append(recall)

            # Hit Rate (binary: did we get at least one hit?)
            hit_rate = 1.0 if n_hits > 0 else 0.0
            hit_rates.append(hit_rate)

            # NDCG@K
            dcg = 0.0
            for i, item in enumerate(recommendations):
                if item in actual_items:
                    dcg += 1.0 / np.log2(i + 2)  # +2 because i starts at 0

            # Ideal DCG (if all top-k were relevant)
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k))])
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

            # Diversity (genre coverage for this user)
            user_genres = set()
            for movie_id in recommendations:
                movie_data = self.train_movies[self.train_movies['movie_id'] == movie_id]
                if not movie_data.empty and movie_data['genres'].iloc[0] is not None:
                    genres = movie_data['genres'].iloc[0]
                    if isinstance(genres, list):
                        user_genres.update(genres)

            diversity = len(user_genres) / k if k > 0 else 0.0
            diversity_scores.append(diversity)

        # Calculate coverage (catalog coverage)
        unique_recommendations = set()
        for recs in all_recommendations:
            unique_recommendations.update(recs)

        total_movies = len(self.train_movies)
        coverage = len(unique_recommendations) / total_movies if total_movies > 0 else 0.0

        # Calculate overall diversity (all genres covered)
        all_genres = set()
        for genres in self.train_movies['genres'].dropna():
            if isinstance(genres, list):
                all_genres.update(genres)

        recommended_genres = set()
        for recs in all_recommendations:
            for movie_id in recs:
                movie_data = self.train_movies[self.train_movies['movie_id'] == movie_id]
                if not movie_data.empty and movie_data['genres'].iloc[0] is not None:
                    genres = movie_data['genres'].iloc[0]
                    if isinstance(genres, list):
                        recommended_genres.update(genres)

        overall_diversity = len(recommended_genres) / len(all_genres) if len(all_genres) > 0 else 0.0

        # Aggregate metrics
        metrics = {
            'n_users': len(user_list),
            'n_users_evaluated': len(precision_scores),
            'precision_at_k': float(np.mean(precision_scores)) if precision_scores else 0.0,
            'recall_at_k': float(np.mean(recall_scores)) if recall_scores else 0.0,
            'ndcg_at_k': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            'hit_rate_at_k': float(np.mean(hit_rates)) if hit_rates else 0.0,
            'diversity': overall_diversity,
            'coverage': coverage,
            'avg_inference_time_ms': float(np.mean(inference_times)) if inference_times else 0.0,
            'std_inference_time_ms': float(np.std(inference_times)) if inference_times else 0.0
        }

        # Print results
        print(f"\n{group_name.upper()} Results (k={k}):")
        print(f"  Users evaluated: {metrics['n_users_evaluated']}/{metrics['n_users']}")
        print(f"  Precision@{k}: {metrics['precision_at_k']:.4f}")
        print(f"  Recall@{k}: {metrics['recall_at_k']:.4f}")
        print(f"  NDCG@{k}: {metrics['ndcg_at_k']:.4f}")
        print(f"  Hit Rate@{k}: {metrics['hit_rate_at_k']:.4f}")
        print(f"  Diversity: {metrics['diversity']:.4f}")
        print(f"  Coverage: {metrics['coverage']:.4f}")
        print(f"  Avg Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")

        return metrics

    def save_results(self, output_path: str = 'app/model/offline_evaluation_results.json'):
        """
        Save evaluation results to JSON file.

        Args:
            output_path: Path to save results
        """
        print(f"\n{'='*80}")
        print(f"SAVING RESULTS")
        print(f"{'='*80}")

        # Add timestamp
        self.results['evaluation_timestamp'] = datetime.now().isoformat()
        self.results['evaluation_framework'] = 'offline_evaluation.py'

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    def run_complete_evaluation(self, test_size: float = 0.2, k: int = 20):
        """
        Run the complete evaluation pipeline.

        Args:
            test_size: Proportion of data for testing
            k: Number of recommendations to evaluate
        """
        print(f"\n{'='*80}")
        print("COMPLETE OFFLINE EVALUATION PIPELINE")
        print(f"{'='*80}")

        # Step 1: Load data
        self.load_data()

        # Step 2: Split data
        self.temporal_train_test_split(test_size=test_size)

        # Step 3: Train model
        self.train_model()

        # Step 4: Evaluate
        self.evaluate_metrics(k=k)

        # Step 5: Save results
        self.save_results()

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print("\nOutput:")
        print("  - JSON results: app/model/offline_evaluation_results.json")


if __name__ == "__main__":
    # Run complete evaluation
    # Use relative path from project root to data directory
    evaluator = OfflineEvaluator(data_path='data-pull/data/')
    evaluator.run_complete_evaluation(test_size=0.2, k=20)
