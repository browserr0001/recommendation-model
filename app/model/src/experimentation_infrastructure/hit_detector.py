"""
Hit Detector for A/B Testing
=============================
This module detects "hits" by joining recommendations with user interactions.
A hit occurs when a user interacts (rates or watches) with a recommended movie
within a specified time window.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from datetime import timedelta
from tqdm import tqdm


class HitDetector:
    """Detect hits by joining predictions with user interactions"""

    def __init__(self, time_window_days: int = 7):
        """
        Initialize hit detector

        Args:
            time_window_days: Number of days after recommendation to count interactions as hits
        """
        self.time_window_days = time_window_days
        self.time_window = timedelta(days=time_window_days)

    def detect_hits_for_prediction(self,
                                   user_id: int,
                                   recommendation_timestamp: pd.Timestamp,
                                   recommended_movies: List[str],
                                   user_interactions: pd.DataFrame) -> Tuple[int, List[str]]:
        """
        Detect hits for a single prediction

        Args:
            user_id: User ID
            recommendation_timestamp: When recommendations were made
            recommended_movies: List of recommended movie IDs
            user_interactions: DataFrame of interactions for this user

        Returns:
            Tuple of (hit_count, list of hit movie IDs)
        """
        # Define time window
        window_end = recommendation_timestamp + self.time_window

        # Filter interactions within time window
        valid_interactions = user_interactions[
            (user_interactions['interaction_timestamp'] > recommendation_timestamp) &
            (user_interactions['interaction_timestamp'] <= window_end)
        ]

        # Find which recommended movies were interacted with
        hit_movies = []
        for movie_id in recommended_movies:
            if movie_id in valid_interactions['movie_id'].values:
                hit_movies.append(movie_id)

        return len(hit_movies), hit_movies

    def detect_all_hits(self,
                       predictions_df: pd.DataFrame,
                       interactions_df: pd.DataFrame,
                       show_progress: bool = True) -> pd.DataFrame:
        """
        Detect hits for all predictions

        Args:
            predictions_df: DataFrame with predictions (from data_loader)
            interactions_df: DataFrame with user interactions (from data_loader)
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with added columns:
                - total_recommendations (always 20)
                - hits_count (number of hits)
                - hit_movies (list of movie IDs that were hits)
                - hit_rate (hits_count / 20)
        """
        print(f"\nDetecting hits with {self.time_window_days}-day time window...")

        # Create a dictionary for faster lookup of user interactions
        print("  Indexing interactions by user...")
        user_interactions_dict = {}
        for user_id in interactions_df['user_id'].unique():
            user_interactions_dict[user_id] = interactions_df[
                interactions_df['user_id'] == user_id
            ]

        print(f"  Indexed interactions for {len(user_interactions_dict):,} users")

        # Detect hits for each prediction
        results = []

        iterator = predictions_df.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=len(predictions_df),
                          desc="  Detecting hits")

        for idx, row in iterator:
            user_id = row['user_id']

            # Get user's interactions (if any)
            user_interactions = user_interactions_dict.get(user_id, pd.DataFrame())

            if user_interactions.empty:
                # User has no interactions - no hits
                hits_count = 0
                hit_movies = []
            else:
                # Detect hits
                hits_count, hit_movies = self.detect_hits_for_prediction(
                    user_id=user_id,
                    recommendation_timestamp=row['recommendation_timestamp'],
                    recommended_movies=row['recommended_movies'],
                    user_interactions=user_interactions
                )

            results.append({
                'user_id': user_id,
                'recommendation_timestamp': row['recommendation_timestamp'],
                'trained_at': row['trained_at'],
                'model_tag': row['model_tag'],
                'recommended_movies': row['recommended_movies'],
                'total_recommendations': len(row['recommended_movies']),
                'hits_count': hits_count,
                'hit_movies': hit_movies,
                'hit_rate': hits_count / len(row['recommended_movies']) if len(row['recommended_movies']) > 0 else 0
            })

        hits_df = pd.DataFrame(results)

        # Print summary statistics
        print(f"\nHit Detection Summary:")
        print(f"  Total predictions analyzed: {len(hits_df):,}")
        print(f"  Predictions with at least 1 hit: {(hits_df['hits_count'] > 0).sum():,} ({(hits_df['hits_count'] > 0).mean()*100:.2f}%)")
        print(f"  Total hits detected: {hits_df['hits_count'].sum():,}")
        print(f"  Average hits per prediction: {hits_df['hits_count'].mean():.4f}")
        print(f"  Average hit rate: {hits_df['hit_rate'].mean():.4f}")

        print(f"\nBy model:")
        for model in hits_df['trained_at'].unique():
            model_data = hits_df[hits_df['trained_at'] == model]
            print(f"\n  Model: {model}")
            print(f"    Predictions: {len(model_data):,}")
            print(f"    Total hits: {model_data['hits_count'].sum():,}")
            print(f"    Avg hits per prediction: {model_data['hits_count'].mean():.4f}")
            print(f"    Avg hit rate: {model_data['hit_rate'].mean():.4f}")
            print(f"    Predictions with hits: {(model_data['hits_count'] > 0).sum():,} ({(model_data['hits_count'] > 0).mean()*100:.2f}%)")

        return hits_df

    def calculate_precision_at_k(self,
                                 hits_df: pd.DataFrame,
                                 k: int) -> pd.DataFrame:
        """
        Calculate Precision@K for each prediction

        Precision@K = (number of hits in top K) / K

        Args:
            hits_df: DataFrame with hit detection results
            k: Number of top recommendations to consider

        Returns:
            DataFrame with added precision@k column
        """
        print(f"\nCalculating Precision@{k}...")

        precisions = []

        for idx, row in hits_df.iterrows():
            # Get top K recommendations
            top_k_recommendations = row['recommended_movies'][:k]

            # Count how many of the top K were hits
            hits_in_top_k = sum(1 for movie in top_k_recommendations
                               if movie in row['hit_movies'])

            # Calculate precision
            precision = hits_in_top_k / k if k > 0 else 0
            precisions.append(precision)

        hits_df[f'precision@{k}'] = precisions

        print(f"  Average Precision@{k}: {hits_df[f'precision@{k}'].mean():.4f}")
        print(f"  By model:")
        for model in hits_df['trained_at'].unique():
            model_data = hits_df[hits_df['trained_at'] == model]
            print(f"    {model}: {model_data[f'precision@{k}'].mean():.4f}")

        return hits_df

    def get_user_level_metrics(self,
                               hits_df: pd.DataFrame,
                               k: int = 20) -> pd.DataFrame:
        """
        Aggregate metrics at the user level (averaging multiple predictions per user)

        Args:
            hits_df: DataFrame with hit detection and precision results
            k: K value for precision metric

        Returns:
            DataFrame with one row per (user, model) combination:
                - user_id
                - trained_at (model)
                - num_predictions (number of predictions for this user-model)
                - avg_precision@k
                - total_hits
        """
        print(f"\nAggregating user-level metrics...")

        user_metrics = hits_df.groupby(['user_id', 'trained_at']).agg({
            f'precision@{k}': 'mean',
            'hits_count': 'sum',
            'user_id': 'count'  # Count number of predictions
        }).reset_index()

        user_metrics.columns = ['user_id', 'trained_at', f'avg_precision@{k}',
                               'total_hits', 'num_predictions']

        print(f"  User-model combinations: {len(user_metrics):,}")
        print(f"  Unique users: {user_metrics['user_id'].nunique():,}")

        # Check for users with multiple models
        users_per_model = user_metrics.groupby('user_id')['trained_at'].nunique()
        users_with_both_models = (users_per_model > 1).sum()

        print(f"  Users with predictions from both models: {users_with_both_models:,}")

        return user_metrics


if __name__ == "__main__":
    print("Hit Detector Module for A/B Testing")
    print("=" * 80)
    print("\nThis module detects 'hits' by matching recommendations with user interactions.")
    print("\nKey features:")
    print("  - Time window filtering (default: 7 days)")
    print("  - Supports both ratings and watches as hits")
    print("  - Calculates Precision@K metrics")
    print("  - User-level metric aggregation")
