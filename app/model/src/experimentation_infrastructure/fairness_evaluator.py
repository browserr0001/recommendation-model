"""
Fairness Evaluator: Gender-Based Recall@20 Disparity
=====================================================
Evaluates if recommendation model has significant gender-based disparity
in Recall@20 (threshold: < 5% difference between Men and Women)

Usage:
    python3 fairness_evaluator.py
    python3 fairness_evaluator.py --predictions app/logs/predictions_20251201.jsonl
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import argparse
import os
import pytz


def load_data(predictions_path, users_path, ratings_path, watches_path):
    """
    Load all required data files

    Args:
        predictions_path: Path to predictions JSONL file
        users_path: Path to users metadata parquet
        ratings_path: Path to ratings parquet
        watches_path: Path to watches parquet

    Returns:
        predictions_df: DataFrame with prediction logs
        users_df: DataFrame with user demographics (gender)
        interactions_df: Combined ratings + watches
    """

    # 1.1 Load predictions from JSONL
    print(f"  Loading predictions from {predictions_path}...")
    predictions = []
    with open(predictions_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            predictions.append({
                'user_id': data['user_id'],
                'timestamp': pd.to_datetime(data['timestamp']),
                'recommendations': data['recommendations'],
                'trained_at': data['model_metadata']['trained_at']
            })
    predictions_df = pd.DataFrame(predictions)
    print(f"    Loaded {len(predictions_df):,} predictions")

    # 1.2 Load user metadata (gender)
    print(f"  Loading user metadata from {users_path}...")
    users_df = pd.read_parquet(users_path)
    # Keep only: user_id, gender
    users_df = users_df[['user_id', 'gender']]
    print(f"    Loaded {len(users_df):,} users")

    # 1.3 Load ratings
    print(f"  Loading ratings from {ratings_path}...")
    ratings_df = pd.read_parquet(ratings_path)
    # Filter: rating >= 0.5 (positive interactions)
    ratings_df = ratings_df[ratings_df['rating'] >= 0.5]
    ratings_df['interaction_type'] = 'rating'
    print(f"    Loaded {len(ratings_df):,} ratings (rating >= 0.5)")

    # 1.4 Load watches
    print(f"  Loading watches from {watches_path}...")
    watches_df = pd.read_parquet(watches_path)
    watches_df = watches_df.rename(columns={'timestamp_start': 'timestamp'})
    watches_df['interaction_type'] = 'watch'
    print(f"    Loaded {len(watches_df):,} watches")

    # 1.5 Combine interactions
    interactions_df = pd.concat([
        ratings_df[['user_id', 'movie_id', 'timestamp', 'interaction_type']],
        watches_df[['user_id', 'movie_id', 'timestamp', 'interaction_type']]
    ], ignore_index=True)
    print(f"    Combined: {len(interactions_df):,} total interactions")

    # 1.6 Ensure timezone consistency
    if predictions_df['timestamp'].dt.tz is None:
        predictions_df['timestamp'] = predictions_df['timestamp'].dt.tz_localize(pytz.UTC)
    else:
        predictions_df['timestamp'] = predictions_df['timestamp'].dt.tz_convert(pytz.UTC)

    if interactions_df['timestamp'].dt.tz is None:
        interactions_df['timestamp'] = interactions_df['timestamp'].dt.tz_localize(pytz.UTC)
    else:
        interactions_df['timestamp'] = interactions_df['timestamp'].dt.tz_convert(pytz.UTC)

    return predictions_df, users_df, interactions_df


def add_gender_to_predictions(predictions_df, users_df):
    """
    Merge predictions with user gender

    Args:
        predictions_df: DataFrame with predictions
        users_df: DataFrame with user demographics

    Returns:
        DataFrame with gender column added
    """

    # Left join to keep all predictions
    predictions_with_gender = predictions_df.merge(
        users_df[['user_id', 'gender']],
        on='user_id',
        how='left'
    )

    # Check for missing gender data
    missing_gender = predictions_with_gender['gender'].isnull().sum()
    if missing_gender > 0:
        print(f"  ⚠️  Warning: {missing_gender:,} users missing gender data (will be dropped)")
        # Drop rows with missing gender
        predictions_with_gender = predictions_with_gender.dropna(subset=['gender'])

    # Validate gender values
    valid_genders = {'M', 'F'}
    invalid = ~predictions_with_gender['gender'].isin(valid_genders)
    if invalid.any():
        print(f"  ⚠️  Warning: {invalid.sum():,} users with invalid gender values (will be dropped)")
        predictions_with_gender = predictions_with_gender[~invalid]

    print(f"  ✓ {len(predictions_with_gender):,} predictions with valid gender data")

    return predictions_with_gender


def build_user_relevant_items(predictions_df, interactions_df, time_window_days=7):
    """
    For each user in predictions, find relevant items (interactions within time window)

    Args:
        predictions_df: DataFrame with predictions
        interactions_df: DataFrame with user interactions
        time_window_days: Number of days after prediction to count interactions

    Returns:
        Dictionary: {(user_id, timestamp): set of relevant movie_ids}
    """

    user_relevant_items = {}

    # Group interactions by user for efficiency
    interactions_by_user = {
        user_id: group
        for user_id, group in interactions_df.groupby('user_id')
    }

    total_predictions = len(predictions_df)
    progress_interval = max(1, total_predictions // 10)  # Report every 10%

    for idx, row in predictions_df.iterrows():
        user_id = row['user_id']
        pred_time = row['timestamp']

        # Define time window: [pred_time, pred_time + 7 days]
        window_start = pred_time
        window_end = pred_time + pd.Timedelta(days=time_window_days)

        # Get user's interactions
        if user_id in interactions_by_user:
            user_interactions = interactions_by_user[user_id]

            # Filter by time window
            relevant = user_interactions[
                (user_interactions['timestamp'] >= window_start) &
                (user_interactions['timestamp'] <= window_end)
            ]

            # Extract unique movie IDs
            user_relevant_items[(user_id, pred_time)] = set(relevant['movie_id'].unique())
        else:
            # User has no interactions
            user_relevant_items[(user_id, pred_time)] = set()

        # Progress reporting
        if (idx + 1) % progress_interval == 0:
            print(f"    Processed {idx + 1:,} / {total_predictions:,} predictions ({(idx+1)/total_predictions*100:.0f}%)")

    print(f"  ✓ Built relevant items for {len(user_relevant_items):,} prediction instances")

    return user_relevant_items


def calculate_recall_at_k(predictions_df, user_relevant_items, k=20):
    """
    Calculate Recall@K for each prediction

    Args:
        predictions_df: DataFrame with predictions
        user_relevant_items: Dictionary mapping (user_id, timestamp) to relevant items
        k: Number of recommendations to consider

    Returns:
        DataFrame with recall@k column added
    """

    recalls = []
    num_relevant = []
    num_hits = []

    for idx, row in predictions_df.iterrows():
        user_id = row['user_id']
        pred_time = row['timestamp']
        recommendations = row['recommendations'][:k]  # Top K

        # Get relevant items for this user at this time
        relevant_items = user_relevant_items.get((user_id, pred_time), set())

        if len(relevant_items) == 0:
            # No relevant items → recall = 0
            recall = 0.0
            hits = 0
        else:
            # Count hits in top K
            recommended_set = set(recommendations)
            hits_set = recommended_set & relevant_items
            hits = len(hits_set)

            # Calculate recall
            recall = hits / len(relevant_items)

        recalls.append(recall)
        num_relevant.append(len(relevant_items))
        num_hits.append(hits)

    # Add columns to dataframe
    predictions_df = predictions_df.copy()
    predictions_df[f'recall@{k}'] = recalls
    predictions_df['num_relevant_items'] = num_relevant
    predictions_df['num_hits'] = num_hits

    print(f"  ✓ Calculated Recall@{k} for {len(predictions_df):,} predictions")
    print(f"    Mean Recall@{k}: {np.mean(recalls):.4f}")
    print(f"    Users with interactions: {sum(1 for r in num_relevant if r > 0):,} ({sum(1 for r in num_relevant if r > 0)/len(num_relevant)*100:.1f}%)")

    return predictions_df


def aggregate_by_gender(predictions_df, k=20):
    """
    Calculate aggregate metrics by gender

    Args:
        predictions_df: DataFrame with predictions and recall
        k: K value for recall metric

    Returns:
        Dictionary with metrics for each gender
    """

    metrics = {}
    recall_col = f'recall@{k}'

    for gender in ['M', 'F']:
        gender_data = predictions_df[predictions_df['gender'] == gender]

        if len(gender_data) == 0:
            print(f"  ⚠️  Warning: No data for gender '{gender}'")
            continue

        metrics[gender] = {
            'n_users': len(gender_data),
            'mean_recall': float(gender_data[recall_col].mean()),
            'median_recall': float(gender_data[recall_col].median()),
            'std_recall': float(gender_data[recall_col].std()),
            'min_recall': float(gender_data[recall_col].min()),
            'max_recall': float(gender_data[recall_col].max()),
            'p25_recall': float(gender_data[recall_col].quantile(0.25)),
            'p75_recall': float(gender_data[recall_col].quantile(0.75)),

            # Additional context
            'users_with_interactions': int((gender_data['num_relevant_items'] > 0).sum()),
            'mean_relevant_items': float(gender_data['num_relevant_items'].mean()),
            'mean_hits': float(gender_data['num_hits'].mean())
        }

    print(f"  ✓ Aggregated metrics:")
    for gender in ['M', 'F']:
        if gender in metrics:
            print(f"    {gender}: {metrics[gender]['n_users']:,} users, mean recall = {metrics[gender]['mean_recall']:.4f}")

    return metrics


def evaluate_fairness(metrics, predictions_df, k=20, threshold=0.05):
    """
    Calculate disparity and determine if fairness requirement is met

    Args:
        metrics: Dictionary with metrics by gender
        predictions_df: DataFrame with predictions (for statistical test)
        k: K value for recall metric
        threshold: Disparity threshold for fairness

    Returns:
        Dictionary with fairness evaluation results
    """

    if 'M' not in metrics or 'F' not in metrics:
        return {
            'error': 'Missing data for one or both genders',
            'passes_fairness': False,
            'verdict': 'FAIL ✗ (insufficient data)'
        }

    recall_M = metrics['M']['mean_recall']
    recall_F = metrics['F']['mean_recall']

    # Absolute disparity
    absolute_disparity = abs(recall_M - recall_F)

    # Relative disparity (percentage of larger value)
    max_recall = max(recall_M, recall_F)
    relative_disparity = absolute_disparity / max_recall if max_recall > 0 else 0

    # Fairness check
    passes_fairness = absolute_disparity < threshold

    # Statistical significance (Welch's t-test)
    recall_col = f'recall@{k}'
    male_recalls = predictions_df[predictions_df['gender'] == 'M'][recall_col]
    female_recalls = predictions_df[predictions_df['gender'] == 'F'][recall_col]

    t_stat, p_value = stats.ttest_ind(male_recalls, female_recalls, equal_var=False)
    is_significant = p_value < 0.05

    results = {
        'recall_men': float(recall_M),
        'recall_women': float(recall_F),
        'absolute_disparity': float(absolute_disparity),
        'relative_disparity': float(relative_disparity),
        'threshold': float(threshold),
        'passes_fairness': bool(passes_fairness),
        'statistical_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': bool(is_significant),
            'interpretation': (
                f"Difference is {'statistically significant' if is_significant else 'not statistically significant'} "
                f"(p={p_value:.4f})"
            )
        },
        'verdict': 'PASS ✓' if passes_fairness else 'FAIL ✗'
    }

    return results


def generate_report(metrics, fairness_results, predictions_df, output_dir='outputs', k=20):
    """
    Generate comprehensive fairness report

    Args:
        metrics: Dictionary with metrics by gender
        fairness_results: Dictionary with fairness evaluation
        predictions_df: DataFrame with all prediction data
        output_dir: Directory to save reports
        k: K value for recall metric
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Text Report
    report_path = f'{output_dir}/fairness_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FAIRNESS EVALUATION REPORT: Gender-Based Recall@20 Disparity\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Prediction Log: app/logs/predictions_20251201.jsonl\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Metric: Recall@{k}\n")
        f.write(f"Fairness Threshold: {fairness_results['threshold']*100:.1f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("RESULTS BY GENDER\n")
        f.write("=" * 80 + "\n\n")

        for gender, label in [('M', 'Men'), ('F', 'Women')]:
            if gender not in metrics:
                f.write(f"{label}: NO DATA\n\n")
                continue

            f.write(f"{label}:\n")
            f.write(f"  Sample Size:           {metrics[gender]['n_users']:,} users\n")
            f.write(f"  Mean Recall@{k}:        {metrics[gender]['mean_recall']:.4f} ({metrics[gender]['mean_recall']*100:.2f}%)\n")
            f.write(f"  Median Recall@{k}:      {metrics[gender]['median_recall']:.4f}\n")
            f.write(f"  Std Deviation:         {metrics[gender]['std_recall']:.4f}\n")
            f.write(f"  Min / Max:             {metrics[gender]['min_recall']:.4f} / {metrics[gender]['max_recall']:.4f}\n")
            f.write(f"  25th / 75th percentile: {metrics[gender]['p25_recall']:.4f} / {metrics[gender]['p75_recall']:.4f}\n")
            f.write(f"  Users w/ Interactions: {metrics[gender]['users_with_interactions']:,} ({metrics[gender]['users_with_interactions']/metrics[gender]['n_users']*100:.1f}%)\n")
            f.write(f"  Avg Relevant Items:    {metrics[gender]['mean_relevant_items']:.2f}\n")
            f.write(f"  Avg Hits:              {metrics[gender]['mean_hits']:.2f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("FAIRNESS EVALUATION\n")
        f.write("=" * 80 + "\n\n")

        if 'error' in fairness_results:
            f.write(f"ERROR: {fairness_results['error']}\n\n")
        else:
            f.write(f"Men Recall@{k}:          {fairness_results['recall_men']:.4f} ({fairness_results['recall_men']*100:.2f}%)\n")
            f.write(f"Women Recall@{k}:        {fairness_results['recall_women']:.4f} ({fairness_results['recall_women']*100:.2f}%)\n\n")
            f.write(f"Absolute Disparity:     {fairness_results['absolute_disparity']:.4f} ({fairness_results['absolute_disparity']*100:.2f}%)\n")
            f.write(f"Relative Disparity:     {fairness_results['relative_disparity']:.4f} ({fairness_results['relative_disparity']*100:.2f}%)\n")
            f.write(f"Threshold:              {fairness_results['threshold']:.4f} ({fairness_results['threshold']*100:.2f}%)\n\n")

            f.write(f"Statistical Test (Welch's t-test):\n")
            f.write(f"  t-statistic: {fairness_results['statistical_test']['t_statistic']:.4f}\n")
            f.write(f"  p-value: {fairness_results['statistical_test']['p_value']:.4f}\n")
            f.write(f"  {fairness_results['statistical_test']['interpretation']}\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"VERDICT: {fairness_results['verdict']}\n")
        f.write("=" * 80 + "\n")

        if fairness_results['passes_fairness']:
            f.write("\n✓ The model meets the fairness requirement.\n")
            f.write(f"  The gender-based disparity in Recall@{k} is {fairness_results['absolute_disparity']*100:.2f}%,\n")
            f.write(f"  which is below the {fairness_results['threshold']*100:.1f}% threshold.\n")
        else:
            f.write("\n✗ The model FAILS the fairness requirement.\n")
            f.write(f"  The gender-based disparity in Recall@{k} is {fairness_results['absolute_disparity']*100:.2f}%,\n")
            f.write(f"  which exceeds the {fairness_results['threshold']*100:.1f}% threshold.\n")

    # 2. JSON Report
    json_path = f'{output_dir}/fairness_results.json'
    with open(json_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'fairness_results': fairness_results,
            'configuration': {
                'k': k,
                'threshold': fairness_results['threshold'],
                'prediction_log': 'app/logs/predictions_20251201.jsonl'
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    # 3. CSV with detailed data
    csv_path = f'{output_dir}/fairness_data.csv'
    predictions_df.to_csv(csv_path, index=False)

    print(f"\n  ✓ Reports saved to {output_dir}/")
    print(f"    - {report_path}")
    print(f"    - {json_path}")
    print(f"    - {csv_path}")


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description='Evaluate fairness: gender-based Recall@20 disparity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 fairness_evaluator.py
  python3 fairness_evaluator.py --predictions app/logs/predictions_20251201.jsonl
        """
    )
    parser.add_argument('--predictions', default='app/logs/predictions_20251201.jsonl',
                       help='Path to predictions JSONL file')
    parser.add_argument('--users', default='data-pull/data/meta/users_new.parquet',
                       help='Path to users metadata parquet')
    parser.add_argument('--ratings', default='data-pull/data/ratings/ratings.parquet',
                       help='Path to ratings parquet')
    parser.add_argument('--watches', default='data-pull/data/watches/watches.parquet',
                       help='Path to watches parquet')
    parser.add_argument('--window', type=int, default=7,
                       help='Time window in days for interactions (default: 7)')
    parser.add_argument('--k', type=int, default=20,
                       help='K for Recall@K metric (default: 20)')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Disparity threshold for fairness (default: 0.05 = 5%%)')
    parser.add_argument('--output', default='outputs',
                       help='Output directory for results (default: outputs)')

    args = parser.parse_args()

    print("=" * 80)
    print("FAIRNESS EVALUATION: Gender-Based Recall@20 Disparity")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Predictions: {args.predictions}")
    print(f"  Users:       {args.users}")
    print(f"  Ratings:     {args.ratings}")
    print(f"  Watches:     {args.watches}")
    print(f"  Time window: {args.window} days")
    print(f"  Metric:      Recall@{args.k}")
    print(f"  Threshold:   {args.threshold*100:.1f}%")
    print(f"  Output:      {args.output}/")

    # STEP 1: Load data
    print("\n" + "=" * 80)
    print("[1/7] Loading data...")
    print("=" * 80)
    predictions_df, users_df, interactions_df = load_data(
        args.predictions, args.users, args.ratings, args.watches
    )

    # STEP 2: Add gender to predictions
    print("\n" + "=" * 80)
    print("[2/7] Joining predictions with user gender...")
    print("=" * 80)
    predictions_df = add_gender_to_predictions(predictions_df, users_df)

    # STEP 3: Build relevant items per user
    print("\n" + "=" * 80)
    print(f"[3/7] Identifying relevant items per user (within {args.window}-day window)...")
    print("=" * 80)
    user_relevant_items = build_user_relevant_items(
        predictions_df, interactions_df, args.window
    )

    # STEP 4: Calculate Recall@K
    print("\n" + "=" * 80)
    print(f"[4/7] Calculating Recall@{args.k}...")
    print("=" * 80)
    predictions_df = calculate_recall_at_k(
        predictions_df, user_relevant_items, args.k
    )

    # STEP 5: Aggregate by gender
    print("\n" + "=" * 80)
    print("[5/7] Aggregating metrics by gender...")
    print("=" * 80)
    metrics = aggregate_by_gender(predictions_df, args.k)

    # STEP 6: Evaluate fairness
    print("\n" + "=" * 80)
    print("[6/7] Evaluating fairness...")
    print("=" * 80)
    fairness_results = evaluate_fairness(metrics, predictions_df, args.k, args.threshold)

    # STEP 7: Generate report
    print("\n" + "=" * 80)
    print("[7/7] Generating report...")
    print("=" * 80)
    generate_report(metrics, fairness_results, predictions_df, args.output, args.k)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if 'error' not in fairness_results:
        print(f"Men Recall@{args.k}:      {fairness_results['recall_men']:.4f} ({fairness_results['recall_men']*100:.2f}%)")
        print(f"Women Recall@{args.k}:    {fairness_results['recall_women']:.4f} ({fairness_results['recall_women']*100:.2f}%)")
        print(f"Absolute Disparity:  {fairness_results['absolute_disparity']:.4f} ({fairness_results['absolute_disparity']*100:.2f}%)")
        print(f"Threshold:           {fairness_results['threshold']:.4f} ({fairness_results['threshold']*100:.2f}%)")
    print(f"\nVERDICT: {fairness_results['verdict']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
