import pytest
import numpy as np
import pandas as pd
from jarjarquant.data_analyst import DataAnalyst


@pytest.fixture
def sample_data():
    return np.random.randn(100)


def test_visual_stationary_test(sample_data):
    # This function plots a graph, so we won't test its output directly
    DataAnalyst.visual_stationary_test(sample_data)


def test_adf_test_stationary(sample_data):
    result = DataAnalyst.adf_test(sample_data, verbose=True)
    assert result in ["passed", "failed"]


def test_jb_normality_test(sample_data):
    result = DataAnalyst.jb_normality_test(sample_data, verbose=True)
    assert result in ["passed", "failed"]


def test_relative_entropy(sample_data):
    entropy = DataAnalyst.relative_entropy(sample_data, verbose=True)
    assert 0 <= entropy <= 1


def test_range_iqr_ratio(sample_data):
    ratio = DataAnalyst.range_iqr_ratio(sample_data, verbose=True)
    assert ratio > 0


def test_mutual_information(sample_data):
    lag = 5
    nmi_scores = DataAnalyst.mutual_information(sample_data, lag)
    assert len(nmi_scores) == lag
    assert all(0 <= score <= 1 for score in nmi_scores if not np.isnan(score))


def test_discretize_array(sample_data):
    n_bins = 5
    discretized = DataAnalyst.discretize_array(sample_data, n_bins)
    assert len(discretized) == len(sample_data)
    assert all(0 <= val < n_bins for val in discretized)


def test_atr():
    atr_length = 14
    high_series = pd.Series(np.random.randn(100))
    low_series = pd.Series(np.random.randn(100))
    close_series = pd.Series(np.random.randn(100))
    atr_values = DataAnalyst.atr(
        atr_length, high_series, low_series, close_series)
    assert len(atr_values) == len(high_series)


def test_compute_legendre_coefficients():
    lookback = 10
    degree = 2
    coeffs = DataAnalyst.compute_legendre_coefficients(lookback, degree)
    assert len(coeffs) == lookback


def test_calculate_regression_coefficient():
    prices = np.random.randn(10)
    legendre_coeffs = DataAnalyst.compute_legendre_coefficients(10, 2)
    slope = DataAnalyst.calculate_regression_coefficient(
        prices, legendre_coeffs)
    assert isinstance(slope, float)


def test_get_daily_vol():
    close_series = pd.Series(np.random.randn(100))
    daily_vol = DataAnalyst.get_daily_vol(close_series)
    assert len(daily_vol) == len(close_series)


def test_get_spearman_correlation():
    series1 = pd.Series(np.random.randn(100))
    series2 = pd.Series(np.random.randn(100))
    result = DataAnalyst.get_spearman_correlation(series1, series2)
    assert 'spearman_corr' in result
    assert 'spearman_corr_quartile' in result
    assert len(result['spearman_corr_quartile']) == 4
