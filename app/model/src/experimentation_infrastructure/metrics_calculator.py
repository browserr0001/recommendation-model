"""
Metrics Calculator for A/B Testing
===================================
This module calculates various recommendation metrics for model comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelMetrics:
    """Container for model-level metrics"""
    model_id: str
    model_tag: str
    num_predictions: int
    num_users: int
    num_users_with_hits: int
    total_hits: int

    # Precision metrics
    mean_precision: float
    median_precision: float
    std_precision: float

    # Hit rate
    overall_hit_rate: float
    user_hit_rate: float  # Percentage of users with at least 1 hit

    # Distribution percentiles
    p25_precision: float
    p75_precision: float

    # For statistical testing
    user_level_precisions: np.ndarray  # Array of per-user precision values


class MetricsCalculator:
    """Calculate recommendation metrics for A/B testing"""

    def __init__(self, k: int = 20):
        """
        Initialize metrics calculator

        Args:
            k: Number of recommendations to consider for Precision@K
        """
        self.k = k

    def calculate_model_metrics(self,
                                hits_df: pd.DataFrame,
                                model_id: str) -> ModelMetrics:
        """
        Calculate comprehensive metrics for a single model

        Args:
            hits_df: DataFrame with hit detection results (from HitDetector)
            model_id: trained_at timestamp identifying the model

        Returns:
            ModelMetrics object with all calculated metrics
        """
        # Filter for this model
        model_data = hits_df[hits_df['trained_at'] == model_id].copy()

        if len(model_data) == 0:
            raise ValueError(f"No data found for model {model_id}")

        # Basic counts
        num_predictions = len(model_data)
        num_users = model_data['user_id'].nunique()

        # Precision metric (must be calculated already in hits_df)
        precision_col = f'precision@{self.k}'
        if precision_col not in model_data.columns:
            raise ValueError(f"{precision_col} not found in hits_df. Run HitDetector.calculate_precision_at_k() first.")

        # Aggregate to user level (average precision per user)
        user_metrics = model_data.groupby('user_id').agg({
            precision_col: 'mean',
            'hits_count': 'sum'
        }).reset_index()

        user_metrics.columns = ['user_id', 'avg_precision', 'total_hits']

        # Calculate metrics
        mean_precision = user_metrics['avg_precision'].mean()
        median_precision = user_metrics['avg_precision'].median()
        std_precision = user_metrics['avg_precision'].std()

        # Hit statistics
        num_users_with_hits = (user_metrics['total_hits'] > 0).sum()
        total_hits = user_metrics['total_hits'].sum()

        # Hit rates
        overall_hit_rate = total_hits / (num_predictions * self.k) if num_predictions > 0 else 0
        user_hit_rate = num_users_with_hits / num_users if num_users > 0 else 0

        # Percentiles
        p25_precision = user_metrics['avg_precision'].quantile(0.25)
        p75_precision = user_metrics['avg_precision'].quantile(0.75)

        # Get model tag
        model_tag = model_data['model_tag'].iloc[0] if 'model_tag' in model_data.columns else 'unknown'

        return ModelMetrics(
            model_id=model_id,
            model_tag=model_tag,
            num_predictions=num_predictions,
            num_users=num_users,
            num_users_with_hits=num_users_with_hits,
            total_hits=total_hits,
            mean_precision=mean_precision,
            median_precision=median_precision,
            std_precision=std_precision,
            overall_hit_rate=overall_hit_rate,
            user_hit_rate=user_hit_rate,
            p25_precision=p25_precision,
            p75_precision=p75_precision,
            user_level_precisions=user_metrics['avg_precision'].values
        )

    def compare_two_models(self,
                          hits_df: pd.DataFrame,
                          model_a_id: str,
                          model_b_id: str) -> Dict:
        """
        Calculate and compare metrics for two models

        Args:
            hits_df: DataFrame with hit detection results
            model_a_id: trained_at timestamp for Model A (baseline)
            model_b_id: trained_at timestamp for Model B (challenger)

        Returns:
            Dictionary with metrics for both models and comparison
        """
        print(f"\nCalculating metrics for model comparison...")
        print(f"  Model A (baseline): {model_a_id}")
        print(f"  Model B (challenger): {model_b_id}")

        # Calculate metrics for both models
        metrics_a = self.calculate_model_metrics(hits_df, model_a_id)
        metrics_b = self.calculate_model_metrics(hits_df, model_b_id)

        # Calculate differences
        precision_diff = metrics_b.mean_precision - metrics_a.mean_precision
        precision_pct_change = (precision_diff / metrics_a.mean_precision * 100) if metrics_a.mean_precision > 0 else 0

        hit_rate_diff = metrics_b.overall_hit_rate - metrics_a.overall_hit_rate
        hit_rate_pct_change = (hit_rate_diff / metrics_a.overall_hit_rate * 100) if metrics_a.overall_hit_rate > 0 else 0

        # Print summary
        print(f"\nModel A (Baseline) Metrics:")
        print(f"  Predictions: {metrics_a.num_predictions:,}")
        print(f"  Users: {metrics_a.num_users:,}")
        print(f"  Mean Precision@{self.k}: {metrics_a.mean_precision:.6f}")
        print(f"  Median Precision@{self.k}: {metrics_a.median_precision:.6f}")
        print(f"  Overall Hit Rate: {metrics_a.overall_hit_rate:.6f}")
        print(f"  User Hit Rate: {metrics_a.user_hit_rate:.4f} ({metrics_a.user_hit_rate*100:.2f}%)")

        print(f"\nModel B (Challenger) Metrics:")
        print(f"  Predictions: {metrics_b.num_predictions:,}")
        print(f"  Users: {metrics_b.num_users:,}")
        print(f"  Mean Precision@{self.k}: {metrics_b.mean_precision:.6f}")
        print(f"  Median Precision@{self.k}: {metrics_b.median_precision:.6f}")
        print(f"  Overall Hit Rate: {metrics_b.overall_hit_rate:.6f}")
        print(f"  User Hit Rate: {metrics_b.user_hit_rate:.4f} ({metrics_b.user_hit_rate*100:.2f}%)")

        print(f"\nDifferences (B - A):")
        print(f"  Precision@{self.k}: {precision_diff:+.6f} ({precision_pct_change:+.2f}%)")
        print(f"  Hit Rate: {hit_rate_diff:+.6f} ({hit_rate_pct_change:+.2f}%)")

        return {
            'model_a': metrics_a,
            'model_b': metrics_b,
            'precision_difference': precision_diff,
            'precision_percent_change': precision_pct_change,
            'hit_rate_difference': hit_rate_diff,
            'hit_rate_percent_change': hit_rate_pct_change,
            'k': self.k
        }

    def get_summary_table(self, comparison: Dict) -> pd.DataFrame:
        """
        Create a summary table comparing the two models

        Args:
            comparison: Output from compare_two_models()

        Returns:
            DataFrame with comparison table
        """
        metrics_a = comparison['model_a']
        metrics_b = comparison['model_b']

        data = {
            'Metric': [
                'Predictions',
                'Unique Users',
                f'Mean Precision@{self.k}',
                f'Median Precision@{self.k}',
                f'Std Precision@{self.k}',
                'Overall Hit Rate',
                'User Hit Rate (%)',
                'Total Hits',
                'Users with Hits',
                f'25th Percentile Precision@{self.k}',
                f'75th Percentile Precision@{self.k}'
            ],
            'Model A (Baseline)': [
                f"{metrics_a.num_predictions:,}",
                f"{metrics_a.num_users:,}",
                f"{metrics_a.mean_precision:.6f}",
                f"{metrics_a.median_precision:.6f}",
                f"{metrics_a.std_precision:.6f}",
                f"{metrics_a.overall_hit_rate:.6f}",
                f"{metrics_a.user_hit_rate*100:.2f}%",
                f"{metrics_a.total_hits:,}",
                f"{metrics_a.num_users_with_hits:,}",
                f"{metrics_a.p25_precision:.6f}",
                f"{metrics_a.p75_precision:.6f}"
            ],
            'Model B (Challenger)': [
                f"{metrics_b.num_predictions:,}",
                f"{metrics_b.num_users:,}",
                f"{metrics_b.mean_precision:.6f}",
                f"{metrics_b.median_precision:.6f}",
                f"{metrics_b.std_precision:.6f}",
                f"{metrics_b.overall_hit_rate:.6f}",
                f"{metrics_b.user_hit_rate*100:.2f}%",
                f"{metrics_b.total_hits:,}",
                f"{metrics_b.num_users_with_hits:,}",
                f"{metrics_b.p25_precision:.6f}",
                f"{metrics_b.p75_precision:.6f}"
            ]
        }

        return pd.DataFrame(data)


if __name__ == "__main__":
    print("Metrics Calculator Module for A/B Testing")
    print("=" * 80)
    print("\nThis module calculates recommendation metrics for model comparison.")
    print("\nKey metrics:")
    print("  - Precision@K (mean, median, std)")
    print("  - Hit Rate (overall and per-user)")
    print("  - Distribution statistics (percentiles)")
    print("  - User-level aggregations for statistical testing")
