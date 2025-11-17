#!/usr/bin/env python3
"""
Quick script to extract results from hits_data.csv since the full pipeline failed during visualization
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the hits data
print("Loading hits data...")
hits_df = pd.read_csv('outputs/hits_data.csv')

# Parse the recommended_movies column (it's stored as string representation of list)
import ast
hits_df['recommended_movies'] = hits_df['recommended_movies'].apply(ast.literal_eval)
hits_df['hit_movies'] = hits_df['hit_movies'].apply(ast.literal_eval)

print(f"Loaded {len(hits_df):,} predictions")

# Get the two models
models = hits_df['trained_at'].unique()
print(f"\nModels found: {models}")

model_a = models[0]
model_b = models[1]

# Split data by model
df_a = hits_df[hits_df['trained_at'] == model_a]
df_b = hits_df[hits_df['trained_at'] == model_b]

# Aggregate to user level (average precision per user)
user_metrics_a = df_a.groupby('user_id')['precision@20'].mean().values
user_metrics_b = df_b.groupby('user_id')['precision@20'].mean().values

print(f"\nModel A: {len(user_metrics_a):,} users")
print(f"Model B: {len(user_metrics_b):,} users")

# Calculate basic statistics
stats_a = {
    'mean': float(np.mean(user_metrics_a)),
    'median': float(np.median(user_metrics_a)),
    'std': float(np.std(user_metrics_a)),
    'min': float(np.min(user_metrics_a)),
    'max': float(np.max(user_metrics_a))
}

stats_b = {
    'mean': float(np.mean(user_metrics_b)),
    'median': float(np.median(user_metrics_b)),
    'std': float(np.std(user_metrics_b)),
    'min': float(np.min(user_metrics_b)),
    'max': float(np.max(user_metrics_b))
}

# Run statistical tests
from scipy import stats as scipy_stats

# Mann-Whitney U test
u_stat, p_value_mw = scipy_stats.mannwhitneyu(user_metrics_a, user_metrics_b, alternative='two-sided')

# Welch's t-test
t_stat, p_value_t = scipy_stats.ttest_ind(user_metrics_a, user_metrics_b, equal_var=False)

# Calculate effect size (Cohen's d)
pooled_std = np.sqrt((np.var(user_metrics_a) + np.var(user_metrics_b)) / 2)
cohens_d = (stats_b['mean'] - stats_a['mean']) / pooled_std if pooled_std > 0 else 0

# Create results dictionary
results = {
    'model_a': {
        'timestamp': model_a,
        'num_predictions': int(len(df_a)),
        'num_users': int(len(user_metrics_a)),
        'statistics': stats_a
    },
    'model_b': {
        'timestamp': model_b,
        'num_predictions': int(len(df_b)),
        'num_users': int(len(user_metrics_b)),
        'statistics': stats_b
    },
    'comparison': {
        'mean_difference': float(stats_b['mean'] - stats_a['mean']),
        'percent_change': float((stats_b['mean'] - stats_a['mean']) / stats_a['mean'] * 100) if stats_a['mean'] > 0 else 0
    },
    'statistical_tests': {
        'mann_whitney_u': {
            'statistic': float(u_stat),
            'p_value': float(p_value_mw),
            'is_significant': bool(p_value_mw < 0.05)
        },
        'welch_t_test': {
            'statistic': float(t_stat),
            'p_value': float(p_value_t),
            'is_significant': bool(p_value_t < 0.05)
        },
        'effect_size_cohens_d': float(cohens_d)
    },
    'recommendation': None
}

# Determine recommendation
if p_value_mw < 0.05:
    if stats_b['mean'] > stats_a['mean']:
        results['recommendation'] = {
            'decision': 'DEPLOY MODEL B',
            'confidence': 'HIGH',
            'reasoning': f"Model B shows statistically significant improvement of {results['comparison']['percent_change']:.2f}%"
        }
    else:
        results['recommendation'] = {
            'decision': 'KEEP MODEL A',
            'confidence': 'HIGH',
            'reasoning': f"Model A performs significantly better ({-results['comparison']['percent_change']:.2f}% better)"
        }
else:
    results['recommendation'] = {
        'decision': 'INCONCLUSIVE - KEEP MODEL A',
        'confidence': 'LOW',
        'reasoning': f"No statistically significant difference (p={p_value_mw:.4f}). Observed difference: {results['comparison']['percent_change']:.2f}%"
    }

# Save to JSON
output_file = 'outputs/ab_test_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_file}")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nModel A: Mean Precision@20 = {stats_a['mean']:.6f}")
print(f"Model B: Mean Precision@20 = {stats_b['mean']:.6f}")
print(f"\nDifference: {results['comparison']['mean_difference']:+.6f} ({results['comparison']['percent_change']:+.2f}%)")
print(f"\nMann-Whitney U test: p = {p_value_mw:.4f} {'✓ Significant' if p_value_mw < 0.05 else '✗ Not significant'}")
print(f"Welch's t-test: p = {p_value_t:.4f} {'✓ Significant' if p_value_t < 0.05 else '✗ Not significant'}")
print(f"Effect size (Cohen's d): {cohens_d:.4f}")
print(f"\n{results['recommendation']['decision']}")
print(f"Reasoning: {results['recommendation']['reasoning']}")
print("="*80)
