"""
Statistical Tests for A/B Testing
==================================
This module performs statistical hypothesis testing to determine if differences
between two models are statistically significant.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class StatisticalTestResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str


class ABTester:
    """Perform statistical tests for A/B testing"""

    def __init__(self, alpha: float = 0.05):
        """
        Initialize A/B tester

        Args:
            alpha: Significance level (default: 0.05 for 95% confidence)
        """
        self.alpha = alpha

    def mann_whitney_test(self,
                         model_a_metrics: np.ndarray,
                         model_b_metrics: np.ndarray,
                         alternative: str = 'two-sided') -> StatisticalTestResult:
        """
        Perform Mann-Whitney U test (non-parametric test for independent samples)

        This test is appropriate when:
        - Samples are independent
        - Data may not be normally distributed
        - Comparing medians/distributions

        Args:
            model_a_metrics: Array of user-level metrics for Model A
            model_b_metrics: Array of user-level metrics for Model B
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            StatisticalTestResult object
        """
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            model_a_metrics,
            model_b_metrics,
            alternative=alternative
        )

        # Determine significance
        is_significant = p_value < self.alpha

        # Calculate effect size (rank-biserial correlation)
        n_a = len(model_a_metrics)
        n_b = len(model_b_metrics)
        effect_size = 1 - (2 * statistic) / (n_a * n_b)

        # Calculate confidence interval using bootstrap
        ci_lower, ci_upper = self._bootstrap_ci_for_difference(
            model_a_metrics, model_b_metrics
        )

        # Interpretation
        median_a = np.median(model_a_metrics)
        median_b = np.median(model_b_metrics)
        diff = median_b - median_a

        if is_significant:
            if diff > 0:
                interpretation = f"Model B has significantly higher median ({median_b:.6f} vs {median_a:.6f}, p={p_value:.4f})"
            else:
                interpretation = f"Model A has significantly higher median ({median_a:.6f} vs {median_b:.6f}, p={p_value:.4f})"
        else:
            interpretation = f"No significant difference in medians (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Mann-Whitney U Test",
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )

    def welch_t_test(self,
                    model_a_metrics: np.ndarray,
                    model_b_metrics: np.ndarray) -> StatisticalTestResult:
        """
        Perform Welch's t-test (parametric test that doesn't assume equal variances)

        This test is appropriate when:
        - Samples are independent
        - Data is approximately normally distributed
        - Variances may be unequal

        Args:
            model_a_metrics: Array of user-level metrics for Model A
            model_b_metrics: Array of user-level metrics for Model B

        Returns:
            StatisticalTestResult object
        """
        # Perform Welch's t-test
        statistic, p_value = stats.ttest_ind(
            model_a_metrics,
            model_b_metrics,
            equal_var=False  # Welch's t-test
        )

        # Determine significance
        is_significant = p_value < self.alpha

        # Calculate Cohen's d effect size
        effect_size = self._cohens_d(model_a_metrics, model_b_metrics)

        # Calculate confidence interval
        ci_lower, ci_upper = self._bootstrap_ci_for_difference(
            model_a_metrics, model_b_metrics
        )

        # Interpretation
        mean_a = np.mean(model_a_metrics)
        mean_b = np.mean(model_b_metrics)
        diff = mean_b - mean_a

        if is_significant:
            if diff > 0:
                interpretation = f"Model B has significantly higher mean ({mean_b:.6f} vs {mean_a:.6f}, p={p_value:.4f})"
            else:
                interpretation = f"Model A has significantly higher mean ({mean_a:.6f} vs {mean_b:.6f}, p={p_value:.4f})"
        else:
            interpretation = f"No significant difference in means (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Welch's t-test",
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )

    def permutation_test(self,
                        model_a_metrics: np.ndarray,
                        model_b_metrics: np.ndarray,
                        n_permutations: int = 10000) -> StatisticalTestResult:
        """
        Perform permutation test (exact test, no distributional assumptions)

        This test is appropriate when:
        - No assumptions about the distribution
        - Small sample sizes
        - Want an exact p-value

        Args:
            model_a_metrics: Array of user-level metrics for Model A
            model_b_metrics: Array of user-level metrics for Model B
            n_permutations: Number of permutations (default: 10,000)

        Returns:
            StatisticalTestResult object
        """
        # Observed difference in means
        observed_diff = np.mean(model_b_metrics) - np.mean(model_a_metrics)

        # Combine all data
        combined = np.concatenate([model_a_metrics, model_b_metrics])
        n_a = len(model_a_metrics)
        n_total = len(combined)

        # Perform permutations
        permuted_diffs = []
        for _ in range(n_permutations):
            # Randomly shuffle and split
            np.random.shuffle(combined)
            perm_a = combined[:n_a]
            perm_b = combined[n_a:]

            # Calculate difference
            perm_diff = np.mean(perm_b) - np.mean(perm_a)
            permuted_diffs.append(perm_diff)

        permuted_diffs = np.array(permuted_diffs)

        # Calculate p-value (two-sided)
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

        # Determine significance
        is_significant = p_value < self.alpha

        # Effect size
        effect_size = self._cohens_d(model_a_metrics, model_b_metrics)

        # Confidence interval from permutation distribution
        ci_lower = np.percentile(permuted_diffs, (self.alpha / 2) * 100)
        ci_upper = np.percentile(permuted_diffs, (1 - self.alpha / 2) * 100)

        # Interpretation
        mean_a = np.mean(model_a_metrics)
        mean_b = np.mean(model_b_metrics)

        if is_significant:
            if observed_diff > 0:
                interpretation = f"Model B has significantly higher mean ({mean_b:.6f} vs {mean_a:.6f}, p={p_value:.4f})"
            else:
                interpretation = f"Model A has significantly higher mean ({mean_a:.6f} vs {mean_b:.6f}, p={p_value:.4f})"
        else:
            interpretation = f"No significant difference in means (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Permutation Test",
            statistic=observed_diff,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )

    def bootstrap_test(self,
                      model_a_metrics: np.ndarray,
                      model_b_metrics: np.ndarray,
                      n_bootstrap: int = 10000) -> StatisticalTestResult:
        """
        Perform bootstrap hypothesis test

        Args:
            model_a_metrics: Array of user-level metrics for Model A
            model_b_metrics: Array of user-level metrics for Model B
            n_bootstrap: Number of bootstrap samples

        Returns:
            StatisticalTestResult object
        """
        # Observed difference
        observed_diff = np.mean(model_b_metrics) - np.mean(model_a_metrics)

        # Bootstrap
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            boot_a = np.random.choice(model_a_metrics, size=len(model_a_metrics), replace=True)
            boot_b = np.random.choice(model_b_metrics, size=len(model_b_metrics), replace=True)

            # Calculate difference
            boot_diff = np.mean(boot_b) - np.mean(boot_a)
            bootstrap_diffs.append(boot_diff)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # P-value: proportion of bootstrap samples where difference crosses zero
        # (two-sided test)
        p_value = 2 * min(np.mean(bootstrap_diffs >= 0), np.mean(bootstrap_diffs <= 0))

        # Determine significance
        is_significant = p_value < self.alpha

        # Effect size
        effect_size = self._cohens_d(model_a_metrics, model_b_metrics)

        # Confidence interval
        ci_lower = np.percentile(bootstrap_diffs, (self.alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - self.alpha / 2) * 100)

        # Interpretation
        mean_a = np.mean(model_a_metrics)
        mean_b = np.mean(model_b_metrics)

        if is_significant:
            if observed_diff > 0:
                interpretation = f"Model B has significantly higher mean ({mean_b:.6f} vs {mean_a:.6f}, p={p_value:.4f})"
            else:
                interpretation = f"Model A has significantly higher mean ({mean_a:.6f} vs {mean_b:.6f}, p={p_value:.4f})"
        else:
            interpretation = f"No significant difference in means (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="Bootstrap Test",
            statistic=observed_diff,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )

    def run_all_tests(self,
                     model_a_metrics: np.ndarray,
                     model_b_metrics: np.ndarray) -> Dict[str, StatisticalTestResult]:
        """
        Run multiple statistical tests for comprehensive analysis

        Args:
            model_a_metrics: Array of user-level metrics for Model A
            model_b_metrics: Array of user-level metrics for Model B

        Returns:
            Dictionary with results from all tests
        """
        print("\nRunning statistical tests...")

        results = {}

        # Mann-Whitney U test (recommended for non-normal data)
        print("  Running Mann-Whitney U test...")
        results['mann_whitney'] = self.mann_whitney_test(model_a_metrics, model_b_metrics)

        # Welch's t-test
        print("  Running Welch's t-test...")
        results['welch_t'] = self.welch_t_test(model_a_metrics, model_b_metrics)

        # Bootstrap test
        print("  Running Bootstrap test...")
        results['bootstrap'] = self.bootstrap_test(model_a_metrics, model_b_metrics)

        # Permutation test (computationally intensive)
        print("  Running Permutation test...")
        results['permutation'] = self.permutation_test(model_a_metrics, model_b_metrics)

        return results

    def check_normality(self,
                       model_a_metrics: np.ndarray,
                       model_b_metrics: np.ndarray) -> Dict[str, Dict]:
        """
        Check if data is normally distributed using Shapiro-Wilk test

        Args:
            model_a_metrics: Array of user-level metrics for Model A
            model_b_metrics: Array of user-level metrics for Model B

        Returns:
            Dictionary with normality test results for both models
        """
        print("\nChecking normality assumptions...")

        results = {}

        # Test Model A
        if len(model_a_metrics) <= 5000:  # Shapiro-Wilk limit
            stat_a, p_a = stats.shapiro(model_a_metrics)
            results['model_a'] = {
                'test': 'Shapiro-Wilk',
                'statistic': stat_a,
                'p_value': p_a,
                'is_normal': p_a > 0.05
            }
        else:
            # Use Kolmogorov-Smirnov for large samples
            stat_a, p_a = stats.kstest(model_a_metrics, 'norm')
            results['model_a'] = {
                'test': 'Kolmogorov-Smirnov',
                'statistic': stat_a,
                'p_value': p_a,
                'is_normal': p_a > 0.05
            }

        # Test Model B
        if len(model_b_metrics) <= 5000:
            stat_b, p_b = stats.shapiro(model_b_metrics)
            results['model_b'] = {
                'test': 'Shapiro-Wilk',
                'statistic': stat_b,
                'p_value': p_b,
                'is_normal': p_b > 0.05
            }
        else:
            stat_b, p_b = stats.kstest(model_b_metrics, 'norm')
            results['model_b'] = {
                'test': 'Kolmogorov-Smirnov',
                'statistic': stat_b,
                'p_value': p_b,
                'is_normal': p_b > 0.05
            }

        print(f"  Model A: {'Normal' if results['model_a']['is_normal'] else 'Not Normal'} (p={results['model_a']['p_value']:.4f})")
        print(f"  Model B: {'Normal' if results['model_b']['is_normal'] else 'Not Normal'} (p={results['model_b']['p_value']:.4f})")

        return results

    def _cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        var_a = np.var(a, ddof=1)
        var_b = np.var(b, ddof=1)
        n_a = len(a)
        n_b = len(b)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

        if pooled_std == 0:
            return 0

        return (mean_b - mean_a) / pooled_std

    def _bootstrap_ci_for_difference(self,
                                     a: np.ndarray,
                                     b: np.ndarray,
                                     n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for the difference in means"""
        bootstrap_diffs = []

        for _ in range(n_bootstrap):
            boot_a = np.random.choice(a, size=len(a), replace=True)
            boot_b = np.random.choice(b, size=len(b), replace=True)
            bootstrap_diffs.append(np.mean(boot_b) - np.mean(boot_a))

        ci_lower = np.percentile(bootstrap_diffs, (self.alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - self.alpha / 2) * 100)

        return ci_lower, ci_upper


if __name__ == "__main__":
    print("Statistical Tests Module for A/B Testing")
    print("=" * 80)
    print("\nThis module performs statistical hypothesis testing.")
    print("\nAvailable tests:")
    print("  - Mann-Whitney U Test (non-parametric, recommended)")
    print("  - Welch's t-test (parametric, unequal variances)")
    print("  - Permutation Test (exact, no assumptions)")
    print("  - Bootstrap Test (resampling-based)")
    print("\nEffect size measures:")
    print("  - Cohen's d (standardized difference)")
    print("  - Rank-biserial correlation (for Mann-Whitney)")
