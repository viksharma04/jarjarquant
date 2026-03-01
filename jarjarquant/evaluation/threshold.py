"""Threshold search and optimization for indicator evaluation.

Extracted from FeatureEvaluator. Provides threshold search (profit factor
analysis at different thresholds) and optimize_threshold (Cython-accelerated
optimal threshold finding).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import spearmanr

from jarjarquant._cython.opt_threshold import optimize_threshold_cython

logger = logging.getLogger(__name__)


def optimize_threshold(
    indicator_values: np.ndarray,
    return_values: np.ndarray,
    min_kept: float = 0.1,
    flip_sign: bool = False,
    return_pval: bool = True,
) -> dict:
    """Optimize threshold for an indicator to maximize performance factor.

    Args:
        indicator_values: Array of indicator values.
        return_values: Array of return values.
        min_kept: Minimum fraction of data points to keep.
        flip_sign: Whether to flip the sign of indicator values.
        return_pval: Whether to compute p-value via permutation test.

    Returns:
        Dictionary with spearman_corr, optimal thresholds, PFs, and p-value.
    """
    indicator_values = np.asarray(indicator_values, dtype=np.float64)
    return_values = np.asarray(return_values, dtype=np.float64)

    n = len(indicator_values)
    if n == 0:
        raise ValueError("Input arrays must have at least one element.")

    min_kept_count = max(int(n * min_kept), 1)

    # Spearman correlation
    spearman_result = spearmanr(indicator_values, return_values, nan_policy="omit")
    if hasattr(spearman_result, "statistic"):
        spearman_corr = float(getattr(spearman_result, "statistic"))
    else:
        spearman_corr = float(spearman_result[0])

    indicator_sign = -1.0 if spearman_corr < 0.0 else 1.0

    if flip_sign:
        work_signal = -indicator_sign * indicator_values.copy()
    else:
        work_signal = indicator_sign * indicator_values.copy()
    work_return = return_values.copy()

    # Drop NaN values
    nan_indices = np.isnan(work_signal) | np.isnan(work_return)
    work_signal = work_signal[~nan_indices]
    work_return = work_return[~nan_indices]

    n = len(work_signal)

    # Sort by signal
    sort_index = np.argsort(work_signal)
    work_signal = work_signal[sort_index]
    work_return = work_return[sort_index]

    (
        best_high_index,
        best_low_index,
        best_high_pf,
        best_low_pf,
        best_high_acc,
        best_low_acc,
    ) = optimize_threshold_cython(work_signal, work_return, int(min_kept_count))

    high_thresh = work_signal[best_high_index]
    low_thresh = work_signal[best_low_index]
    pf_high = best_high_pf
    pf_low = best_low_pf
    best_overall_pf = max(pf_high, pf_low)

    # P-value via permutation test
    if return_pval:
        i = 0
        for _ in range(1000):
            permuted_returns = np.random.choice(
                work_return, size=len(work_return), replace=True
            )
            _, _, high_pf, low_pf, _, _ = optimize_threshold_cython(
                work_signal, permuted_returns, int(min_kept_count)
            )
            permuted_pf = max(high_pf, low_pf)
            if permuted_pf >= best_overall_pf:
                i += 1
        best_pf_pval = i / 1000
    else:
        best_pf_pval = None

    return {
        "spearman_corr": spearman_corr,
        "optimal_long_thresh": float(high_thresh),
        "optimal_long_pf": float(pf_high),
        "optimal_long_acc": float(best_high_acc),
        "optimal_short_thresh": float(low_thresh),
        "optimal_short_pf": float(pf_low),
        "optimal_short_acc": float(best_low_acc),
        "best_pf": float(best_overall_pf),
        "best_pf_pval": best_pf_pval,
    }


def threshold_search(
    indicator_values: np.ndarray,
    associated_returns: np.ndarray,
    thresholds: list | None = None,
    n_thresholds: int | None = 10,
    threshold_option: str = "percentile",
) -> list[dict]:
    """Evaluate profit factors for different thresholds of an indicator.

    Args:
        indicator_values: Array of indicator values.
        associated_returns: Array of associated returns.
        thresholds: Predefined threshold values. If None, computed automatically.
        n_thresholds: Number of thresholds to evaluate.
        threshold_option: 'linear' or 'percentile'.

    Returns:
        List of dicts, each containing threshold and profit factor metrics.
    """
    indicator_values = np.asarray(indicator_values, dtype=np.float64)
    associated_returns = np.asarray(associated_returns, dtype=np.float64)

    if len(indicator_values) != len(associated_returns):
        raise ValueError(
            "indicator_values and associated_returns must have the same length."
        )

    if thresholds is None:
        if n_thresholds is None:
            raise ValueError("Either n_thresholds or thresholds must be provided.")
        if threshold_option == "linear":
            min_val, max_val = indicator_values.min(), indicator_values.max()
            thresholds = np.linspace(min_val, max_val, n_thresholds + 2)[1:-1]
        elif threshold_option == "percentile":
            percentiles = np.linspace(0, 100, n_thresholds + 2)[1:-1]
            thresholds = np.percentile(indicator_values, percentiles)
        else:
            raise ValueError("threshold_option must be 'linear' or 'percentile'.")

    results = []

    for threshold in thresholds:
        above = indicator_values > threshold
        below = indicator_values < threshold

        # Above threshold metrics
        pos_above = associated_returns[above & (associated_returns > 0)].sum()
        neg_above = -associated_returns[above & (associated_returns < 0)].sum()
        pf_long_above = pos_above / neg_above if neg_above > 0 else 0.0
        pf_short_above = neg_above / pos_above if pos_above > 0 else 0.0

        above_returns = associated_returns[above]
        mean_return_above = float(np.mean(above_returns)) if len(above_returns) > 0 else 0.0
        std_return_above = float(np.std(above_returns, ddof=1)) if len(above_returns) > 1 else 0.0
        median_return_above = float(np.median(above_returns)) if len(above_returns) > 0 else 0.0
        q25_above = float(np.percentile(above_returns, 25)) if len(above_returns) > 0 else 0.0
        q75_above = float(np.percentile(above_returns, 75)) if len(above_returns) > 0 else 0.0

        if len(above_returns) > 2:
            spearman_above, _ = spearmanr(
                indicator_values[above], above_returns
            )
        else:
            spearman_above = 0.0

        # Below threshold metrics
        pos_below = associated_returns[below & (associated_returns > 0)].sum()
        neg_below = -associated_returns[below & (associated_returns < 0)].sum()
        pf_long_below = pos_below / neg_below if neg_below > 0 else 0.0
        pf_short_below = neg_below / pos_below if pos_below > 0 else 0.0

        below_returns = associated_returns[below]
        mean_return_below = float(np.mean(below_returns)) if len(below_returns) > 0 else 0.0
        std_return_below = float(np.std(below_returns, ddof=1)) if len(below_returns) > 1 else 0.0
        median_return_below = float(np.median(below_returns)) if len(below_returns) > 0 else 0.0
        q25_below = float(np.percentile(below_returns, 25)) if len(below_returns) > 0 else 0.0
        q75_below = float(np.percentile(below_returns, 75)) if len(below_returns) > 0 else 0.0

        if len(below_returns) > 2:
            spearman_below, _ = spearmanr(
                indicator_values[below], below_returns
            )
        else:
            spearman_below = 0.0

        results.append(
            {
                "Threshold": float(threshold),
                "% values > threshold": float(above.mean() * 100),
                "Spearman correlation above threshold": float(np.nan_to_num(spearman_above)),
                "Mean return above threshold": mean_return_above,
                "Std dev return above threshold": std_return_above,
                "Median return above threshold": median_return_above,
                "Q25 return above threshold": q25_above,
                "Q75 return above threshold": q75_above,
                "PF Long above threshold": float(np.nan_to_num(pf_long_above)),
                "PF Short above threshold": float(np.nan_to_num(pf_short_above)),
                "% values < threshold": float(below.mean() * 100),
                "Spearman correlation below threshold": float(np.nan_to_num(spearman_below)),
                "Mean return below threshold": mean_return_below,
                "Std dev return below threshold": std_return_below,
                "Median return below threshold": median_return_below,
                "Q25 return below threshold": q25_below,
                "Q75 return below threshold": q75_below,
                "PF Long below threshold": float(np.nan_to_num(pf_long_below)),
                "PF Short below threshold": float(np.nan_to_num(pf_short_below)),
            }
        )

    return results
