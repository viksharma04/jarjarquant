"""Tests for evaluation/threshold module."""

import numpy as np
import pytest
from jarjarquant.evaluation.threshold import optimize_threshold, threshold_search


def test_optimize_threshold_returns_dict():
    np.random.seed(42)
    signal = np.sort(np.random.randn(200))
    returns = np.random.randn(200) * 0.01
    result = optimize_threshold(signal, returns, min_kept=0.1, return_pval=False)
    assert isinstance(result, dict)
    assert "spearman_corr" in result
    assert "optimal_long_thresh" in result
    assert "optimal_short_thresh" in result
    assert "best_pf" in result


def test_threshold_search_returns_list():
    np.random.seed(42)
    indicator_values = np.random.randn(200)
    returns = np.random.randn(200) * 0.01
    result = threshold_search(
        indicator_values=indicator_values,
        associated_returns=returns,
        n_thresholds=5,
    )
    assert isinstance(result, list)
    assert len(result) == 5
    assert "Threshold" in result[0]
    assert "PF Long above threshold" in result[0]
