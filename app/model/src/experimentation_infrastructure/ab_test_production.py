#!/usr/bin/env python3
"""
Production A/B Testing Framework
=================================
End-to-end pipeline for comparing two models using production logs.

This script:
1. Loads prediction logs and user interaction data
2. Detects "hits" (user interactions with recommended movies)
3. Calculates Precision@20 for both models
4. Performs statistical tests to determine significance
5. Generates text reports and saves JSON results

Usage:
    python ab_test_production.py [--predictions PRED_LOG] [--ratings RATINGS] [--watches WATCHES]
                                 [--model-a TIMESTAMP] [--model-b TIMESTAMP]
                                 [--window DAYS] [--output OUTPUT_DIR]

Example:
    python ab_test_production.py \\
        --predictions app/logs/predictions_20251115.jsonl \\
        --ratings data-pull/data/ratings/ratings.parquet \\
        --watches data-pull/data/watches/watches.parquet \\
        --model-a "2025-11-13T22:02:06.510993" \\
        --model-b "2025-11-14T23:45:05.172431" \\
        --window 7 \\
        --output outputs
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np

# Import our custom modules
from data_loader import load_all_data
from hit_detector import HitDetector
from metrics_calculator import MetricsCalculator
from statistical_tests import ABTester


def run_ab_test(predictions_log: str,
                ratings_path: str,
                watches_path: str,
                model_a_timestamp: str,
                model_b_timestamp: str,
                time_window_days: int = 7,
                k: int = 20,
                output_dir: str = "outputs",
                alpha: float = 0.05) -> Dict:
    """
    Run complete A/B test pipeline

    Args:
        predictions_log: Path to predictions JSONL file
        ratings_path: Path to ratings parquet file
        watches_path: Path to watches parquet file
        model_a_timestamp: trained_at timestamp for Model A (baseline)
        model_b_timestamp: trained_at timestamp for Model B (challenger)
        time_window_days: Days after recommendation to count interactions as hits
        k: Number of recommendations to consider for Precision@K
        output_dir: Directory to save results
        alpha: Significance level for statistical tests

    Returns:
        Dictionary with complete test results
    """
    print("=" * 80)
    print("PRODUCTION A/B TESTING FRAMEWORK")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Predictions log: {predictions_log}")
    print(f"  Ratings data: {ratings_path}")
    print(f"  Watches data: {watches_path}")
    print(f"  Model A (baseline): {model_a_timestamp}")
    print(f"  Model B (challenger): {model_b_timestamp}")
    print(f"  Time window: {time_window_days} days")
    print(f"  Metric: Precision@{k}")
    print(f"  Significance level: α = {alpha}")
    print(f"  Output directory: {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========================================
    # STEP 1: Load Data
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    model_timestamps = [model_a_timestamp, model_b_timestamp]
    predictions_df, interactions_df = load_all_data(
        predictions_log=predictions_log,
        ratings_path=ratings_path,
        watches_path=watches_path,
        model_timestamps=model_timestamps
    )

    # ========================================
    # STEP 2: Detect Hits
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 2: DETECTING HITS")
    print("=" * 80)

    hit_detector = HitDetector(time_window_days=time_window_days)
    hits_df = hit_detector.detect_all_hits(predictions_df, interactions_df)

    # Calculate Precision@K
    hits_df = hit_detector.calculate_precision_at_k(hits_df, k=k)

    # Save hits data
    hits_output = output_path / "hits_data.csv"
    hits_df.to_csv(hits_output, index=False)
    print(f"\nSaved hits data to {hits_output}")

    # ========================================
    # STEP 3: Calculate Metrics
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 3: CALCULATING METRICS")
    print("=" * 80)

    metrics_calc = MetricsCalculator(k=k)
    comparison = metrics_calc.compare_two_models(
        hits_df=hits_df,
        model_a_id=model_a_timestamp,
        model_b_id=model_b_timestamp
    )

    # Generate summary table
    summary_table = metrics_calc.get_summary_table(comparison)
    print("\n" + "=" * 80)
    print("METRICS SUMMARY TABLE")
    print("=" * 80)
    print(summary_table.to_string(index=False))

    # Save summary table
    summary_output = output_path / "metrics_summary.csv"
    summary_table.to_csv(summary_output, index=False)
    print(f"\nSaved metrics summary to {summary_output}")

    # ========================================
    # STEP 4: Statistical Testing
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 4: STATISTICAL TESTING")
    print("=" * 80)

    tester = ABTester(alpha=alpha)

    # Check normality
    normality_results = tester.check_normality(
        comparison['model_a'].user_level_precisions,
        comparison['model_b'].user_level_precisions
    )

    # Run all statistical tests
    test_results = tester.run_all_tests(
        comparison['model_a'].user_level_precisions,
        comparison['model_b'].user_level_precisions
    )

    # Print test results
    print("\n" + "=" * 80)
    print("STATISTICAL TEST RESULTS")
    print("=" * 80)

    for test_name, result in test_results.items():
        print(f"\n{result.test_name}:")
        print(f"  Statistic: {result.statistic:.6f}")
        print(f"  P-value: {result.p_value:.6f}")
        print(f"  Significant: {'YES ✓' if result.is_significant else 'NO'}")
        print(f"  Effect size: {result.effect_size:.4f}")
        print(f"  95% CI for difference: [{result.confidence_interval[0]:.6f}, {result.confidence_interval[1]:.6f}]")
        print(f"  Interpretation: {result.interpretation}")

    # ========================================
    # STEP 5: Generate Text Report
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 5: GENERATING TEXT REPORT")
    print("=" * 80)

    report = generate_text_report(
        comparison=comparison,
        test_results=test_results,
        normality_results=normality_results,
        model_a_timestamp=model_a_timestamp,
        model_b_timestamp=model_b_timestamp,
        time_window_days=time_window_days,
        k=k,
        alpha=alpha
    )

    # Save text report
    report_output = output_path / "ab_test_report.txt"
    with open(report_output, 'w') as f:
        f.write(report)
    print(f"Saved text report to {report_output}")

    # Print report to console
    print("\n" + report)

    # ========================================
    # STEP 6: Save JSON Results
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 6: SAVING RESULTS")
    print("=" * 80)

    # Create JSON-serializable results
    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'predictions_log': predictions_log,
            'model_a_timestamp': model_a_timestamp,
            'model_b_timestamp': model_b_timestamp,
            'time_window_days': time_window_days,
            'k': k,
            'alpha': alpha
        },
        'model_a': {
            'timestamp': model_a_timestamp,
            'num_predictions': comparison['model_a'].num_predictions,
            'num_users': comparison['model_a'].num_users,
            'mean_precision': float(comparison['model_a'].mean_precision),
            'median_precision': float(comparison['model_a'].median_precision),
            'std_precision': float(comparison['model_a'].std_precision),
            'overall_hit_rate': float(comparison['model_a'].overall_hit_rate),
            'user_hit_rate': float(comparison['model_a'].user_hit_rate),
            'total_hits': comparison['model_a'].total_hits
        },
        'model_b': {
            'timestamp': model_b_timestamp,
            'num_predictions': comparison['model_b'].num_predictions,
            'num_users': comparison['model_b'].num_users,
            'mean_precision': float(comparison['model_b'].mean_precision),
            'median_precision': float(comparison['model_b'].median_precision),
            'std_precision': float(comparison['model_b'].std_precision),
            'overall_hit_rate': float(comparison['model_b'].overall_hit_rate),
            'user_hit_rate': float(comparison['model_b'].user_hit_rate),
            'total_hits': comparison['model_b'].total_hits
        },
        'comparison': {
            'precision_difference': float(comparison['precision_difference']),
            'precision_percent_change': float(comparison['precision_percent_change']),
            'hit_rate_difference': float(comparison['hit_rate_difference']),
            'hit_rate_percent_change': float(comparison['hit_rate_percent_change'])
        },
        'statistical_tests': {
            test_name: {
                'p_value': float(result.p_value),
                'is_significant': result.is_significant,
                'effect_size': float(result.effect_size),
                'confidence_interval': [float(result.confidence_interval[0]), float(result.confidence_interval[1])],
                'interpretation': result.interpretation
            }
            for test_name, result in test_results.items()
        },
        'normality_tests': normality_results,
        'recommendation': determine_recommendation(comparison, test_results)
    }

    # Save JSON
    json_output = output_path / "ab_test_results.json"
    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON results to {json_output}")

    print("\n" + "=" * 80)
    print(f"COMPLETE! All results saved to: {output_dir}/")
    print("=" * 80)

    return results


def generate_text_report(comparison: Dict,
                         test_results: Dict,
                         normality_results: Dict,
                         model_a_timestamp: str,
                         model_b_timestamp: str,
                         time_window_days: int,
                         k: int,
                         alpha: float) -> str:
    """Generate comprehensive text report"""

    lines = []
    lines.append("=" * 80)
    lines.append("A/B TEST REPORT - PRODUCTION MODEL COMPARISON")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("CONFIGURATION")
    lines.append("-" * 80)
    lines.append(f"Model A (Baseline):  {model_a_timestamp}")
    lines.append(f"Model B (Challenger): {model_b_timestamp}")
    lines.append(f"Time Window: {time_window_days} days after recommendation")
    lines.append(f"Metric: Precision@{k}")
    lines.append(f"Significance Level: α = {alpha}")
    lines.append("")

    lines.append("SAMPLE SIZES")
    lines.append("-" * 80)
    lines.append(f"Model A:")
    lines.append(f"  Predictions: {comparison['model_a'].num_predictions:,}")
    lines.append(f"  Unique Users: {comparison['model_a'].num_users:,}")
    lines.append(f"  Total Hits: {comparison['model_a'].total_hits:,}")
    lines.append(f"")
    lines.append(f"Model B:")
    lines.append(f"  Predictions: {comparison['model_b'].num_predictions:,}")
    lines.append(f"  Unique Users: {comparison['model_b'].num_users:,}")
    lines.append(f"  Total Hits: {comparison['model_b'].total_hits:,}")
    lines.append("")

    lines.append("PRECISION@{} COMPARISON".format(k))
    lines.append("-" * 80)
    lines.append(f"Model A (Baseline):")
    lines.append(f"  Mean:   {comparison['model_a'].mean_precision:.8f}")
    lines.append(f"  Median: {comparison['model_a'].median_precision:.8f}")
    lines.append(f"  Std:    {comparison['model_a'].std_precision:.8f}")
    lines.append(f"")
    lines.append(f"Model B (Challenger):")
    lines.append(f"  Mean:   {comparison['model_b'].mean_precision:.8f}")
    lines.append(f"  Median: {comparison['model_b'].median_precision:.8f}")
    lines.append(f"  Std:    {comparison['model_b'].std_precision:.8f}")
    lines.append(f"")
    lines.append(f"Difference (B - A):")
    lines.append(f"  Absolute: {comparison['precision_difference']:+.8f}")
    lines.append(f"  Relative: {comparison['precision_percent_change']:+.4f}%")
    lines.append("")

    lines.append("STATISTICAL TEST RESULTS")
    lines.append("-" * 80)
    for test_name, result in test_results.items():
        lines.append(f"{result.test_name}:")
        lines.append(f"  P-value: {result.p_value:.6f}")
        lines.append(f"  Significant: {'YES ✓' if result.is_significant else 'NO'}")
        lines.append(f"  Effect Size: {result.effect_size:.4f}")
        lines.append(f"  95% CI: [{result.confidence_interval[0]:.8f}, {result.confidence_interval[1]:.8f}]")
        lines.append(f"  {result.interpretation}")
        lines.append("")

    # Recommendation
    recommendation = determine_recommendation(comparison, test_results)
    lines.append("=" * 80)
    lines.append("RECOMMENDATION")
    lines.append("=" * 80)
    lines.append(f"Decision: {recommendation['decision']}")
    lines.append(f"Confidence: {recommendation['confidence']}")
    lines.append(f"Reasoning: {recommendation['reasoning']}")
    lines.append("")

    return "\n".join(lines)


def determine_recommendation(comparison: Dict, test_results: Dict) -> Dict:
    """Determine final recommendation based on test results"""

    # Count significant tests
    num_significant = sum(1 for r in test_results.values() if r.is_significant)
    total_tests = len(test_results)

    mean_diff = comparison['model_b'].mean_precision - comparison['model_a'].mean_precision
    pct_change = comparison['precision_percent_change']

    if num_significant >= 3:  # Strong evidence (3+ out of 4 tests)
        if mean_diff > 0:
            decision = "DEPLOY MODEL B"
            confidence = "HIGH"
            reasoning = f"Model B shows statistically significant improvement of {pct_change:.2f}% in Precision@20. {num_significant}/{total_tests} statistical tests confirm this difference is unlikely due to chance."
        else:
            decision = "KEEP MODEL A"
            confidence = "HIGH"
            reasoning = f"Model A performs significantly better ({abs(pct_change):.2f}% higher Precision@20). {num_significant}/{total_tests} statistical tests confirm Model B is inferior."
    elif num_significant >= 2:  # Moderate evidence
        if mean_diff > 0:
            decision = "CONSIDER DEPLOYING MODEL B"
            confidence = "MODERATE"
            reasoning = f"Model B shows {pct_change:.2f}% improvement, with {num_significant}/{total_tests} tests showing statistical significance. Recommend additional data collection or extended A/B test."
        else:
            decision = "KEEP MODEL A"
            confidence = "MODERATE"
            reasoning = f"Model A performs {abs(pct_change):.2f}% better, with {num_significant}/{total_tests} tests showing statistical significance. Model B does not show clear improvement."
    else:  # Weak or no evidence
        decision = "INCONCLUSIVE - KEEP MODEL A"
        confidence = "LOW"
        reasoning = f"Only {num_significant}/{total_tests} tests show statistical significance. Difference of {pct_change:.2f}% may be due to random variation. Recommend collecting more data or investigating further before making changes."

    return {
        'decision': decision,
        'confidence': confidence,
        'reasoning': reasoning,
        'num_significant_tests': num_significant,
        'total_tests': total_tests
    }


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description='Run A/B test comparing two models using production logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--predictions',
        type=str,
        default='app/logs/predictions_20251115.jsonl',
        help='Path to predictions JSONL log file'
    )
    parser.add_argument(
        '--ratings',
        type=str,
        default='data-pull/data/ratings/ratings.parquet',
        help='Path to ratings parquet file'
    )
    parser.add_argument(
        '--watches',
        type=str,
        default='data-pull/data/watches/watches.parquet',
        help='Path to watches parquet file'
    )
    parser.add_argument(
        '--model-a',
        type=str,
        default='2025-11-13T22:02:06.510993',
        help='trained_at timestamp for Model A (baseline)'
    )
    parser.add_argument(
        '--model-b',
        type=str,
        default='2025-11-14T23:45:05.172431',
        help='trained_at timestamp for Model B (challenger)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=7,
        help='Time window in days to count hits (default: 7)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=20,
        help='K value for Precision@K metric (default: 20)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level for statistical tests (default: 0.05)'
    )

    args = parser.parse_args()

    # Run the test
    try:
        results = run_ab_test(
            predictions_log=args.predictions,
            ratings_path=args.ratings,
            watches_path=args.watches,
            model_a_timestamp=args.model_a,
            model_b_timestamp=args.model_b,
            time_window_days=args.window,
            k=args.k,
            output_dir=args.output,
            alpha=args.alpha
        )

        print("\n✓ A/B test completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Error running A/B test: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
