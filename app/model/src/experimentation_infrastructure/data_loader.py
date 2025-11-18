"""
Data Loader for Production A/B Testing
======================================
This module loads prediction logs and user interaction data for A/B testing.
"""

import json
import logging
import pandas as pd
import numpy as np
import pytz
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionLogLoader:
    """Load and parse prediction logs from JSONL files"""

    def __init__(self):
        self.predictions_df = None

    def load_predictions(self,
                        jsonl_path: str,
                        model_timestamps: List[str]) -> pd.DataFrame:
        """
        Load predictions from JSONL log file, filtering for specific models

        Args:
            jsonl_path: Path to predictions JSONL file
            model_timestamps: List of trained_at timestamps to filter for

        Returns:
            DataFrame with columns:
                - user_id
                - recommendation_timestamp
                - recommended_movies (list of movie IDs)
                - trained_at (model identifier)
                - model_tag

        Raises:
            FileNotFoundError: If jsonl_path doesn't exist
            ValueError: If no predictions found for specified models
        """
        # Validate inputs
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {jsonl_path}")

        if not model_timestamps:
            raise ValueError("model_timestamps cannot be empty")

        print(f"Loading predictions from {jsonl_path}...")
        print(f"Filtering for models: {model_timestamps}")

        predictions = []
        total_lines = 0
        matched_lines = 0
        decode_errors = 0
        missing_metadata_count = 0
        missing_key_errors = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                total_lines += 1
                try:
                    data = json.loads(line)

                    # Extract model metadata
                    if 'model_metadata' not in data:
                        missing_metadata_count += 1
                        if missing_metadata_count <= 5:  # Log first 5 examples
                            logger.warning(f"Line {total_lines}: Missing 'model_metadata' field")
                        continue

                    trained_at = data['model_metadata'].get('trained_at')

                    # Filter for specified models
                    if trained_at not in model_timestamps:
                        continue

                    matched_lines += 1

                    # Parse prediction data
                    # Ensure timezone consistency - convert to UTC aware
                    timestamp = pd.to_datetime(data['timestamp'])
                    if timestamp.tzinfo is None:
                        # Assume naive timestamps are UTC
                        timestamp = timestamp.replace(tzinfo=pytz.UTC)

                    predictions.append({
                        'user_id': data['user_id'],
                        'recommendation_timestamp': timestamp,
                        'recommended_movies': data['recommendations'],
                        'trained_at': trained_at,
                        'model_tag': data['model_metadata'].get('model_tag', 'unknown')
                    })

                except json.JSONDecodeError as e:
                    decode_errors += 1
                    if decode_errors <= 5:  # Log first 5 examples
                        logger.warning(f"Line {total_lines}: JSON decode error - {e}")
                    continue
                except KeyError as e:
                    missing_key = str(e)
                    if missing_key not in [err[0] for err in missing_key_errors]:
                        missing_key_errors.append((missing_key, total_lines))
                    continue

                # Progress update every 100k lines
                if total_lines % 100000 == 0:
                    print(f"  Processed {total_lines:,} lines, matched {matched_lines:,} predictions...")

        print(f"Loaded {matched_lines:,} predictions from {total_lines:,} total log entries")

        # Log data quality issues
        if decode_errors > 0:
            logger.warning(f"Found {decode_errors} JSON decode errors (shown first 5)")
        if missing_metadata_count > 0:
            logger.warning(f"Found {missing_metadata_count} lines without model_metadata")
        if missing_key_errors:
            logger.warning(f"Found missing keys: {dict(missing_key_errors)}")

        # Log overall data quality
        quality_rate = (matched_lines / total_lines * 100) if total_lines > 0 else 0
        logger.info(f"Data quality: {quality_rate:.2f}% of lines successfully parsed and matched")

        if matched_lines == 0:
            raise ValueError(f"No predictions found for specified models: {model_timestamps}")

        self.predictions_df = pd.DataFrame(predictions)

        # Add metadata
        print("\nPredictions Summary:")
        print(f"  Total predictions: {len(self.predictions_df):,}")
        print(f"  Unique users: {self.predictions_df['user_id'].nunique():,}")
        print(f"  Date range: {self.predictions_df['recommendation_timestamp'].min()} to {self.predictions_df['recommendation_timestamp'].max()}")
        print("\nPredictions by model:")
        print(self.predictions_df.groupby('trained_at').size())

        return self.predictions_df


class InteractionLoader:
    """Load and combine user interaction data from ratings and watches"""

    def __init__(self):
        self.interactions_df = None

    def load_interactions(self,
                         ratings_path: str,
                         watches_path: str,
                         start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load and combine ratings and watches into unified interactions DataFrame

        Args:
            ratings_path: Path to ratings.parquet file
            watches_path: Path to watches.parquet file
            start_date: Optional filter for interactions after this date

        Returns:
            DataFrame with columns:
                - user_id
                - movie_id
                - interaction_timestamp
                - interaction_type ('rating' or 'watch')

        Raises:
            FileNotFoundError: If ratings_path or watches_path doesn't exist
        """
        # Validate inputs
        if not Path(ratings_path).exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
        if not Path(watches_path).exists():
            raise FileNotFoundError(f"Watches file not found: {watches_path}")

        print(f"\nLoading user interactions...")

        # Load ratings
        print(f"  Loading ratings from {ratings_path}...")
        ratings_df = pd.read_parquet(ratings_path)

        # Normalize ratings to interaction format
        ratings_interactions = pd.DataFrame({
            'user_id': ratings_df['user_id'],
            'movie_id': ratings_df['movie_id'],
            'interaction_timestamp': pd.to_datetime(ratings_df['timestamp']),
            'interaction_type': 'rating'
        })

        print(f"    Loaded {len(ratings_interactions):,} ratings")

        # Load watches
        print(f"  Loading watches from {watches_path}...")
        watches_df = pd.read_parquet(watches_path)

        # Normalize watches to interaction format (use timestamp_start)
        watches_interactions = pd.DataFrame({
            'user_id': watches_df['user_id'],
            'movie_id': watches_df['movie_id'],
            'interaction_timestamp': pd.to_datetime(watches_df['timestamp_start']),
            'interaction_type': 'watch'
        })

        print(f"    Loaded {len(watches_interactions):,} watches")

        # Combine both types of interactions
        self.interactions_df = pd.concat([ratings_interactions, watches_interactions],
                                         ignore_index=True)

        # Filter by start date if provided
        if start_date is not None:
            # Ensure start_date is timezone-aware to match interactions_df
            if start_date.tzinfo is None:
                # Assume naive timestamps are UTC
                start_date = start_date.replace(tzinfo=pytz.UTC)

            print(f"  Filtering interactions after {start_date}...")
            before_filter = len(self.interactions_df)
            self.interactions_df = self.interactions_df[
                self.interactions_df['interaction_timestamp'] >= start_date
            ]
            print(f"    Kept {len(self.interactions_df):,} / {before_filter:,} interactions")

        # Sort by timestamp for efficiency
        self.interactions_df = self.interactions_df.sort_values('interaction_timestamp')

        print(f"\nTotal interactions loaded: {len(self.interactions_df):,}")
        print(f"  Unique users: {self.interactions_df['user_id'].nunique():,}")
        print(f"  Unique movies: {self.interactions_df['movie_id'].nunique():,}")
        print(f"  Date range: {self.interactions_df['interaction_timestamp'].min()} to {self.interactions_df['interaction_timestamp'].max()}")
        print(f"\nInteractions by type:")
        print(self.interactions_df['interaction_type'].value_counts())

        return self.interactions_df

    def get_interactions_for_user(self, user_id: int) -> pd.DataFrame:
        """Get all interactions for a specific user"""
        if self.interactions_df is None:
            raise ValueError("No interactions loaded. Call load_interactions() first.")

        return self.interactions_df[self.interactions_df['user_id'] == user_id]


def load_all_data(predictions_log: str,
                  ratings_path: str,
                  watches_path: str,
                  model_timestamps: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all data needed for A/B testing

    Args:
        predictions_log: Path to predictions JSONL file
        ratings_path: Path to ratings parquet file
        watches_path: Path to watches parquet file
        model_timestamps: List of model timestamps to compare

    Returns:
        Tuple of (predictions_df, interactions_df)
    """
    # Load predictions
    pred_loader = PredictionLogLoader()
    predictions_df = pred_loader.load_predictions(predictions_log, model_timestamps)

    # Get earliest prediction timestamp to filter interactions
    earliest_prediction = predictions_df['recommendation_timestamp'].min()

    # Load interactions (only those after first prediction to avoid data leakage)
    interaction_loader = InteractionLoader()
    interactions_df = interaction_loader.load_interactions(
        ratings_path,
        watches_path,
        start_date=earliest_prediction
    )

    return predictions_df, interactions_df


if __name__ == "__main__":
    # Example usage
    print("Data Loader Module for Production A/B Testing")
    print("=" * 80)

    # Example paths
    predictions_log = "app/logs/predictions_20251115.jsonl"
    ratings_path = "data-pull/data/ratings/ratings.parquet"
    watches_path = "data-pull/data/watches/watches.parquet"

    # Model timestamps to compare
    model_a_timestamp = "2025-11-13T22:02:06.510993"
    model_b_timestamp = "2025-11-14T23:45:05.172431"

    print(f"\nExample: Loading data for models:")
    print(f"  Model A: {model_a_timestamp}")
    print(f"  Model B: {model_b_timestamp}")
    print("\nUse load_all_data() function to load data for A/B testing")
