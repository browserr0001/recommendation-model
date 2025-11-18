# Production A/B Testing Framework

A comprehensive framework for comparing two recommendation models using production prediction logs and user interaction data.

## Overview

This framework analyzes production logs to determine if there is a statistically significant difference in performance between two models. It computes Precision@20 by matching recommendations with actual user interactions (ratings and watches) within a 7-day time window.

## Features

- **Hit Detection**: Matches recommendations with user interactions (ratings/watches)
- **Precision@K Metrics**: Calculates precision at various K values per user
- **Statistical Testing**: Multiple tests (Mann-Whitney U, Welch's t-test, Permutation, Bootstrap)
- **Detailed Reporting**: Text and JSON outputs with full statistical analysis
- **Data Export**: CSV files with hit data and metrics for further analysis

## Architecture

### Module Structure

```
experimentation_infrastructure/
├── data_loader.py              # Load prediction logs and interaction data
├── hit_detector.py             # Detect hits (matches between recommendations and interactions)
├── metrics_calculator.py       # Calculate Precision@K and aggregate metrics
├── statistical_tests.py        # Statistical hypothesis testing
├── ab_test_production.py       # Main runner script (entry point)
└── README.md                   # This file
```

### Data Flow

```
1. LOAD DATA
   - Parse predictions JSONL (filter by model timestamp)
   - Load ratings.parquet and watches.parquet
   - Filter interactions after first prediction

2. DETECT HITS
   - For each prediction, check if user interacted with recommended movies
   - within 7 days after recommendation
   - Count hits per prediction

3. CALCULATE METRICS
   - Compute Precision@20 for each prediction
   - Aggregate to user level
   - Calculate summary statistics per model

4. STATISTICAL TESTING
   - Run 4 statistical tests
   - Calculate effect sizes and confidence intervals
   - Apply multiple testing correction

5. GENERATE OUTPUTS
   - Generate text report
   - Save JSON results
   - Export data files (CSV)
```

## Usage

### Basic Usage

Run with default parameters (uses paths specified in script):

```bash
cd /path/to/repo
python3 app/model/src/experimentation_infrastructure/ab_test_production.py
```

### Custom Parameters

```bash
python3 app/model/src/experimentation_infrastructure/ab_test_production.py \
    --predictions app/logs/predictions_20251115.jsonl \
    --ratings data-pull/data/ratings/ratings.parquet \
    --watches data-pull/data/watches/watches.parquet \
    --model-a "2025-11-13T22:02:06.510993" \
    --model-b "2025-11-14T23:45:05.172431" \
    --window 7 \
    --k 20 \
    --output outputs \
    --alpha 0.05
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--predictions` | Path to predictions JSONL log file | `app/logs/predictions_20251115.jsonl` |
| `--ratings` | Path to ratings parquet file | `data-pull/data/ratings/ratings.parquet` |
| `--watches` | Path to watches parquet file | `data-pull/data/watches/watches.parquet` |
| `--model-a` | `trained_at` timestamp for Model A (baseline) | `2025-11-13T22:02:06.510993` |
| `--model-b` | `trained_at` timestamp for Model B (challenger) | `2025-11-14T23:45:05.172431` |
| `--window` | Time window in days to count hits | `7` |
| `--k` | K value for Precision@K metric | `20` |
| `--output` | Output directory for results | `outputs` |
| `--alpha` | Significance level for statistical tests | `0.05` |

## Outputs

The framework generates the following files in the output directory:

### 1. Data Files

**hits_data.csv**
- Complete dataset with hit detection results
- Columns: `user_id`, `recommendation_timestamp`, `trained_at`, `model_tag`, `recommended_movies`, `total_recommendations`, `hits_count`, `hit_movies`, `hit_rate`, `precision@20`

**metrics_summary.csv**
- Comparison table with key metrics for both models
- Includes predictions, users, precision stats, hit rates

### 2. Report Files

**ab_test_report.txt**
- Comprehensive text report with:
  - Configuration summary
  - Sample sizes
  - Precision@20 comparison
  - Statistical test results
  - Final recommendation

**ab_test_results.json**
- Machine-readable results including:
  - Model metrics
  - Statistical test results
  - Recommendation and confidence level

## Understanding the Results

### Key Metrics

**Precision@20**
- Proportion of recommended movies that user interacted with
- Calculated as: `hits_in_recommendations / 20`
- Higher is better

**Hit Rate**
- Overall proportion of hits across all recommendations
- `total_hits / (predictions × 20)`

**User Hit Rate**
- Percentage of users who had at least one hit
- Indicates engagement level

### Statistical Tests

The framework runs 4 different statistical tests:

1. **Mann-Whitney U Test** (recommended for non-normal data)
   - Non-parametric
   - Compares medians
   - Robust to outliers

2. **Welch's t-test**
   - Parametric
   - Compares means
   - Doesn't assume equal variances

3. **Permutation Test**
   - Exact test
   - No distributional assumptions
   - Computationally intensive

4. **Bootstrap Test**
   - Resampling-based
   - Robust confidence intervals

**Important Note on Multiple Testing**: Running 4 tests on the same data increases the risk of false positives (Type I error). If using α=0.05 for each test, consider applying Bonferroni correction (adjusted α=0.0125 for 4 tests) or requiring multiple tests to agree before making decisions.

### Interpretation

**P-value < 0.05**: Statistically significant difference
- Difference is unlikely due to random chance
- Strong evidence of performance difference

**Effect Size (Cohen's d)**:
- Small: 0.2
- Medium: 0.5
- Large: 0.8

**Recommendation Confidence**:
- **HIGH**: 3+ tests significant
- **MODERATE**: 2 tests significant
- **LOW**: 0-1 tests significant

## Example Output

```
================================================================================
RECOMMENDATION
================================================================================
Decision: DEPLOY MODEL B
Confidence: HIGH
Reasoning: Model B shows statistically significant improvement of 22.65% in
Precision@20. 4/4 statistical tests confirm this difference is unlikely due
to chance.
```

## Technical Details

### Hit Detection Logic

A "hit" occurs when:
1. Movie X was recommended to user Y at time T
2. User Y either:
   - Rated movie X after time T (within 7-day window), OR
   - Watched movie X after time T (within 7-day window)

### Time Window

- Default: 7 days
- Interactions before recommendation time are ignored
- Interactions more than 7 days after are ignored
- This ensures causal relationship between recommendation and interaction

### User-Level Aggregation

- If a user receives multiple predictions, their precision values are averaged
- Statistical tests compare user-level metrics (not prediction-level)
- This accounts for users with different activity levels

### Sample Size Considerations

- Minimum 30 users per model recommended for reliable statistics
- Larger samples = more statistical power
- Framework reports actual sample sizes in all outputs

## Dependencies

```
pandas
numpy
scipy
matplotlib
seaborn
pytz
tqdm
```

## Troubleshooting

### Issue: Timezone errors

**Solution**: The framework automatically handles timezone conversions. Ensure your Python environment has `pytz` installed.

### Issue: Memory errors with large logs

**Solution**: The framework processes data efficiently, but for very large logs (>1M predictions), consider:
- Filtering log file beforehand
- Increasing system memory
- Processing in batches

### Issue: No hits detected

**Possible causes**:
1. Time window too short (try increasing `--window`)
2. Interactions data doesn't overlap with prediction timeframe
3. Movie IDs don't match between predictions and interactions

**Debug**: Check the `hits_data.csv` output to inspect individual predictions

### Issue: All tests show no significance

**Possible causes**:
1. Models truly have similar performance
2. Sample size too small (increase data collection period)
3. High variance in user behavior (need more data)

**Solution**: Collect more data or examine effect sizes to determine practical significance

## Advanced Usage

### Using Individual Modules

You can import and use modules separately:

```python
from data_loader import load_all_data
from hit_detector import HitDetector
from metrics_calculator import MetricsCalculator
from statistical_tests import ABTester

# Load data
predictions_df, interactions_df = load_all_data(
    predictions_log="path/to/log.jsonl",
    ratings_path="path/to/ratings.parquet",
    watches_path="path/to/watches.parquet",
    model_timestamps=["timestamp1", "timestamp2"]
)

# Detect hits
detector = HitDetector(time_window_days=7)
hits_df = detector.detect_all_hits(predictions_df, interactions_df)
hits_df = detector.calculate_precision_at_k(hits_df, k=20)

# Calculate metrics
calc = MetricsCalculator(k=20)
comparison = calc.compare_two_models(
    hits_df, "model_a_timestamp", "model_b_timestamp"
)

# Run statistical tests
tester = ABTester(alpha=0.05)
test_results = tester.run_all_tests(
    comparison['model_a'].user_level_precisions,
    comparison['model_b'].user_level_precisions
)

# Generate report
# Results are printed to console and can be saved to files
```

## Best Practices

1. **Always specify time window**: Consider your use case - shorter windows for real-time systems, longer for research

2. **Examine distributions**: Check the distribution plots to validate statistical test assumptions

3. **Consider practical significance**: A statistically significant difference may not be practically important. Look at effect sizes and percentage improvements.

4. **Multiple metrics**: While this framework focuses on Precision@20, consider running multiple tests with different K values (5, 10, 20)

5. **Temporal effects**: Be aware that newer models may have different user populations. Check if user IDs overlap between models.

6. **Document assumptions**: Always document why you chose specific parameters (window, K value, etc.)
