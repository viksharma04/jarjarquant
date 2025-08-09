"""
Unit tests for the data_analyst module.

Tests all functions and their return types, including the enhanced directional change algorithm.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from jarjarquant.data_analyst import (
    ADFTestResult,
    EntropyResult,
    NormalityTestResult,
    RangeIQRResult,
    adf_test,
    atr,
    calculate_regression_coefficient,
    compute_legendre_coefficients,
    compute_normalized_legendre_coefficients,
    directional_change_pivots,
    discretize_array,
    get_daily_vol,
    get_spearman_correlation,
    jb_normality_test,
    mutual_information,
    plot_loess,
    range_iqr_ratio,
    relative_entropy,
    visual_stationary_test,
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.randn(100)


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    n_points = 50
    base_prices = 100 + np.cumsum(np.random.normal(0, 1, n_points))
    
    data = []
    for base in base_prices:
        volatility = 0.02
        daily_range = base * volatility * np.random.uniform(0.5, 2.0)
        low = base - daily_range * np.random.uniform(0.2, 0.8)
        high = base + daily_range * np.random.uniform(0.2, 0.8)
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
        close = low + (high - low) * np.random.uniform(0.2, 0.8)
        data.append([open_price, high, low, close])
    
    return np.array(data)


@pytest.fixture
def sample_price_series():
    """Generate sample price series for directional change testing."""
    return np.array([100, 102, 105, 108, 104, 101, 98, 102, 106, 103, 99, 102])


class TestVisualizationFunctions:
    """Test visualization functions."""
    
    def test_visual_stationary_test(self, sample_data):
        """Test visual stationary test function."""
        # This function creates a plot, so we test it doesn't raise exceptions
        ax = visual_stationary_test(sample_data)
        assert ax is not None
        
    def test_visual_stationary_test_with_percentiles(self, sample_data):
        """Test visual stationary test with custom percentiles."""
        ax = visual_stationary_test(sample_data, upper_percentile=0.9, lower_percentile=0.1)
        assert ax is not None
        
    def test_plot_loess(self, sample_data):
        """Test LOESS plotting function."""
        x = np.linspace(0, 1, len(sample_data))
        # This function creates a plot, test it doesn't raise exceptions
        plot_loess(x, sample_data)


class TestStatisticalTests:
    """Test statistical analysis functions."""
    
    def test_adf_test_basic(self, sample_data):
        """Test basic ADF test functionality."""
        result = adf_test(sample_data)
        
        assert isinstance(result, ADFTestResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert isinstance(result.lags, int)
        assert isinstance(result.nobs, int)
        assert isinstance(result.critical_values, dict)
        assert result.decision in ["stationary", "non_stationary", "strong_evidence_stationary"]
        assert isinstance(result.is_stationary, bool)
        
    def test_adf_test_verbose(self, sample_data):
        """Test ADF test with verbose output."""
        result = adf_test(sample_data, verbose=True)
        assert isinstance(result, ADFTestResult)
        
    def test_adf_test_short_series(self):
        """Test ADF test with too short series."""
        short_data = np.array([1, 2])
        with pytest.raises(ValueError, match="Series too short"):
            adf_test(short_data)
            
    def test_adf_test_with_nans(self):
        """Test ADF test with NaN values."""
        data_with_nans = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        result = adf_test(data_with_nans)
        assert isinstance(result, ADFTestResult)
        
    def test_jb_normality_test_basic(self, sample_data):
        """Test basic Jarque-Bera normality test."""
        result = jb_normality_test(sample_data)
        
        assert isinstance(result, NormalityTestResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert result.method in ["jb", "dagostino"]
        assert result.decision in ["normal", "not_normal"]
        assert isinstance(result.is_normal, bool)
        
    def test_jb_normality_test_methods(self, sample_data):
        """Test both normality test methods."""
        result_jb = jb_normality_test(sample_data, method="jb")
        result_dagostino = jb_normality_test(sample_data, method="dagostino")
        
        assert result_jb.method == "jb"
        assert result_dagostino.method == "dagostino"
        
    def test_jb_normality_test_invalid_method(self, sample_data):
        """Test normality test with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            jb_normality_test(sample_data, method="invalid")
            
    def test_jb_normality_test_constant_series(self):
        """Test normality test with constant series."""
        constant_data = np.array([5, 5, 5, 5, 5])
        with pytest.raises(ValueError, match="Cannot perform normality test on constant series"):
            jb_normality_test(constant_data)
            
    def test_jb_normality_test_with_plot(self, sample_data):
        """Test normality test with plotting enabled."""
        result = jb_normality_test(sample_data, plot=True)
        assert isinstance(result, NormalityTestResult)


class TestEntropyAndOutliers:
    """Test entropy and outlier detection functions."""
    
    def test_relative_entropy_basic(self, sample_data):
        """Test basic entropy calculation."""
        result = relative_entropy(sample_data)
        
        assert isinstance(result, EntropyResult)
        assert isinstance(result.entropy, float)
        assert isinstance(result.normalized_entropy, float)
        assert isinstance(result.n_bins, int)
        assert isinstance(result.n_observations, int)
        assert result.quality_assessment in ["VERY CONCERNING", "CONCERNING", "FINE", "EXCELLENT"]
        assert isinstance(result.is_concerning, bool)
        assert 0 <= result.normalized_entropy <= 1
        
    def test_relative_entropy_verbose(self, sample_data):
        """Test entropy calculation with verbose output."""
        result = relative_entropy(sample_data, verbose=True, explain=True)
        assert isinstance(result, EntropyResult)
        
    def test_relative_entropy_empty_series(self):
        """Test entropy calculation with empty series."""
        with pytest.raises(ValueError, match="Cannot calculate entropy"):
            relative_entropy(np.array([]))
            
    def test_relative_entropy_different_sizes(self):
        """Test entropy calculation with different data sizes."""
        small_data = np.random.randn(10)
        medium_data = np.random.randn(500)
        large_data = np.random.randn(5000)
        
        result_small = relative_entropy(small_data)
        result_medium = relative_entropy(medium_data)
        result_large = relative_entropy(large_data)
        
        assert result_small.n_bins == 3
        assert result_medium.n_bins == 5
        assert result_large.n_bins == 10
        
    def test_range_iqr_ratio_basic(self, sample_data):
        """Test basic range/IQR ratio calculation."""
        result = range_iqr_ratio(sample_data)
        
        assert isinstance(result, RangeIQRResult)
        assert isinstance(result.ratio, float)
        assert isinstance(result.range_value, float)
        assert isinstance(result.iqr_value, float)
        assert isinstance(result.q25, float)
        assert isinstance(result.q75, float)
        assert isinstance(result.min_value, float)
        assert isinstance(result.max_value, float)
        assert isinstance(result.n_observations, int)
        assert result.quality_assessment in [
            "GREAT DISTRIBUTION - MINIMAL OUTLIERS",
            "PASSABLE DISTRIBUTION - SOME OUTLIERS - INSPECT VISUALLY",
            "CONCERNING AMOUNT OF OUTLIERS - CONSIDER TRANSFORMATIONS"
        ]
        assert isinstance(result.has_excessive_outliers, bool)
        assert result.ratio > 0
        
    def test_range_iqr_ratio_verbose(self, sample_data):
        """Test range/IQR ratio with verbose output."""
        result = range_iqr_ratio(sample_data, verbose=True, explain=True)
        assert isinstance(result, RangeIQRResult)
        
    def test_range_iqr_ratio_constant_series(self):
        """Test range/IQR ratio with constant series."""
        constant_data = np.array([5, 5, 5, 5, 5])
        with pytest.raises(ValueError, match="IQR is zero"):
            range_iqr_ratio(constant_data)
            
    def test_range_iqr_ratio_empty_series(self):
        """Test range/IQR ratio with empty series."""
        with pytest.raises(ValueError, match="Cannot calculate range/IQR ratio"):
            range_iqr_ratio(np.array([]))


class TestMutualInformation:
    """Test mutual information functions."""
    
    def test_mutual_information_basic(self, sample_data):
        """Test basic mutual information calculation."""
        lag = 5
        nmi_scores = mutual_information(sample_data, lag)
        
        assert isinstance(nmi_scores, np.ndarray)
        assert len(nmi_scores) == lag
        # Check that scores are between 0 and 1 (excluding NaN)
        valid_scores = nmi_scores[~np.isnan(nmi_scores)]
        assert all(0 <= score <= 1 for score in valid_scores)
        
    def test_mutual_information_discrete(self, sample_data):
        """Test mutual information with discrete data."""
        discrete_data = discretize_array(sample_data, 5)
        lag = 3
        nmi_scores = mutual_information(discrete_data, lag, is_discrete=True)
        
        assert isinstance(nmi_scores, np.ndarray)
        assert len(nmi_scores) == lag
        
    def test_mutual_information_custom_bins(self, sample_data):
        """Test mutual information with custom number of bins."""
        lag = 3
        n_bins = 8
        nmi_scores = mutual_information(sample_data, lag, n_bins=n_bins)
        
        assert isinstance(nmi_scores, np.ndarray)
        assert len(nmi_scores) == lag
        
    def test_mutual_information_invalid_lag(self, sample_data):
        """Test mutual information with invalid lag."""
        with pytest.raises(ValueError, match="Lag is greater than or equal"):
            mutual_information(sample_data, len(sample_data))
            
    def test_discretize_array(self, sample_data):
        """Test array discretization."""
        n_bins = 5
        discretized = discretize_array(sample_data, n_bins)
        
        assert len(discretized) == len(sample_data)
        assert all(0 <= val < n_bins for val in discretized)
        assert isinstance(discretized, np.ndarray)


class TestATR:
    """Test Average True Range calculation."""
    
    def test_atr_basic(self, sample_ohlc_data):
        """Test basic ATR calculation."""
        open_prices, high_prices, low_prices, close_prices = sample_ohlc_data.T
        
        high_series = pd.Series(high_prices)
        low_series = pd.Series(low_prices)
        close_series = pd.Series(close_prices)
        
        atr_length = 14
        atr_values = atr(atr_length, high_series, low_series, close_series)
        
        assert isinstance(atr_values, pd.Series)
        assert len(atr_values) == len(high_series)
        assert atr_values.index.equals(high_series.index)
        
    def test_atr_with_ema(self, sample_ohlc_data):
        """Test ATR calculation with EMA."""
        open_prices, high_prices, low_prices, close_prices = sample_ohlc_data.T
        
        high_series = pd.Series(high_prices)
        low_series = pd.Series(low_prices)
        close_series = pd.Series(close_prices)
        
        atr_values = atr(14, high_series, low_series, close_series, ema=True)
        assert isinstance(atr_values, pd.Series)
        
    def test_atr_with_expanding(self, sample_ohlc_data):
        """Test ATR calculation with expanding window."""
        open_prices, high_prices, low_prices, close_prices = sample_ohlc_data.T
        
        high_series = pd.Series(high_prices)
        low_series = pd.Series(low_prices)
        close_series = pd.Series(close_prices)
        
        atr_values = atr(14, high_series, low_series, close_series, expanding=True)
        assert isinstance(atr_values, pd.Series)
        
    def test_atr_invalid_inputs(self):
        """Test ATR with invalid inputs."""
        # Test non-Series input
        with pytest.raises(ValueError, match="All price inputs must be pandas Series"):
            atr(14, [1, 2, 3], [1, 2, 3], [1, 2, 3])
            
        # Test mismatched indices
        high_series = pd.Series([1, 2, 3], index=[0, 1, 2])
        low_series = pd.Series([1, 2, 3], index=[1, 2, 3])
        close_series = pd.Series([1, 2, 3], index=[0, 1, 2])
        
        with pytest.raises(ValueError, match="All series must have the same index"):
            atr(14, high_series, low_series, close_series)
            
    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        high_series = pd.Series([1, 2])
        low_series = pd.Series([1, 2])
        close_series = pd.Series([1, 2])
        
        with pytest.raises(ValueError, match="Insufficient data"):
            atr(14, high_series, low_series, close_series)


class TestLegendreCoefficients:
    """Test Legendre polynomial functions."""
    
    def test_compute_legendre_coefficients(self):
        """Test Legendre coefficient computation."""
        lookback = 10
        degree = 2
        coeffs = compute_legendre_coefficients(lookback, degree)
        
        assert isinstance(coeffs, np.ndarray)
        assert len(coeffs) == lookback
        
    def test_compute_legendre_coefficients_invalid_degree(self):
        """Test Legendre coefficients with invalid degree."""
        with pytest.raises(ValueError, match="Only degrees 1, 2, or 3"):
            compute_legendre_coefficients(10, 4)
            
    def test_compute_normalized_legendre_coefficients(self):
        """Test normalized Legendre coefficient computation."""
        n = 15
        for degree in [1, 2, 3]:
            coeffs = compute_normalized_legendre_coefficients(n, degree)
            assert isinstance(coeffs, np.ndarray)
            assert len(coeffs) == n
            # Check normalization (should have unit length)
            assert np.isclose(np.linalg.norm(coeffs), 1.0, atol=1e-10)
            
    def test_calculate_regression_coefficient(self):
        """Test regression coefficient calculation."""
        prices = np.random.randn(10)
        legendre_coeffs = compute_legendre_coefficients(10, 2)
        slope = calculate_regression_coefficient(prices, legendre_coeffs)
        
        assert isinstance(slope, float)
        assert not np.isnan(slope)


class TestFinancialFunctions:
    """Test financial utility functions."""
    
    def test_get_daily_vol(self):
        """Test daily volatility calculation."""
        date_range = pd.date_range(start="2022-01-01", periods=100, freq="D")
        close_series = pd.Series(100 + np.cumsum(np.random.randn(100)), index=date_range)
        
        daily_vol = get_daily_vol(close_series)
        
        assert isinstance(daily_vol, pd.Series)
        assert len(daily_vol) == len(close_series)
        assert daily_vol.index.equals(close_series.index)
        assert all(vol >= 0 for vol in daily_vol.dropna())
        
    def test_get_daily_vol_custom_span(self):
        """Test daily volatility with custom span."""
        date_range = pd.date_range(start="2022-01-01", periods=100, freq="D")
        close_series = pd.Series(100 + np.cumsum(np.random.randn(100)), index=date_range)
        
        daily_vol = get_daily_vol(close_series, span=50)
        assert isinstance(daily_vol, pd.Series)
        
    def test_get_spearman_correlation(self, sample_data):
        """Test Spearman correlation calculation."""
        series1 = pd.Series(sample_data)
        series2 = pd.Series(np.random.randn(len(sample_data)))
        
        result = get_spearman_correlation(series1, series2)
        
        assert isinstance(result, dict)
        assert "spearman_corr" in result
        assert "spearman_corr_quartile" in result
        assert len(result["spearman_corr_quartile"]) == 4
        assert isinstance(result["spearman_corr"], (float, np.float64))
        
    def test_get_spearman_correlation_numpy_arrays(self, sample_data):
        """Test Spearman correlation with numpy arrays."""
        series1 = sample_data
        series2 = np.random.randn(len(sample_data))
        
        result = get_spearman_correlation(series1, series2)
        assert isinstance(result, dict)
        assert "spearman_corr" in result


class TestDirectionalChangePivots:
    """Test the enhanced directional change pivots algorithm."""
    
    def test_directional_change_pivots_basic(self, sample_price_series):
        """Test basic directional change functionality."""
        recognition_indices = directional_change_pivots(
            sample_price_series, threshold_type="static", threshold_value=0.03
        )
        
        assert isinstance(recognition_indices, np.ndarray)
        assert recognition_indices.dtype == np.int64
        assert len(recognition_indices) > 0
        assert all(0 <= idx < len(sample_price_series) for idx in recognition_indices)
        
    def test_directional_change_pivots_with_extremes(self, sample_price_series):
        """Test directional change with extreme indices returned."""
        recognition_indices, extreme_indices = directional_change_pivots(
            sample_price_series, threshold_type="static", threshold_value=0.03, return_extremes=True
        )
        
        assert isinstance(recognition_indices, np.ndarray)
        assert isinstance(extreme_indices, np.ndarray)
        assert len(recognition_indices) == len(extreme_indices)
        assert all(0 <= idx < len(sample_price_series) for idx in recognition_indices)
        assert all(0 <= idx < len(sample_price_series) for idx in extreme_indices)
        
    def test_directional_change_pivots_static_threshold(self, sample_price_series):
        """Test static threshold functionality."""
        recognition_indices = directional_change_pivots(
            sample_price_series, threshold_type="static", threshold_value=0.02
        )
        assert isinstance(recognition_indices, np.ndarray)
        
    def test_directional_change_pivots_volatility_threshold(self, sample_ohlc_data):
        """Test volatility-based threshold functionality."""
        open_prices, high_prices, low_prices, close_prices = sample_ohlc_data.T
        
        recognition_indices = directional_change_pivots(
            close_prices,
            threshold_type="volatility",
            atr_window=10,
            high_series=high_prices,
            low_series=low_prices,
            close_series=close_prices
        )
        
        assert isinstance(recognition_indices, np.ndarray)
        assert len(recognition_indices) > 0
        
    def test_directional_change_pivots_volatility_with_extremes(self, sample_ohlc_data):
        """Test volatility threshold with extreme indices."""
        open_prices, high_prices, low_prices, close_prices = sample_ohlc_data.T
        
        recognition_indices, extreme_indices = directional_change_pivots(
            close_prices,
            threshold_type="volatility",
            atr_window=10,
            high_series=high_prices,
            low_series=low_prices,
            close_series=close_prices,
            return_extremes=True
        )
        
        assert isinstance(recognition_indices, np.ndarray)
        assert isinstance(extreme_indices, np.ndarray)
        assert len(recognition_indices) == len(extreme_indices)
        
    def test_directional_change_pivots_empty_series(self):
        """Test directional change with empty series."""
        empty_series = np.array([])
        
        # Test without extremes
        recognition_indices = directional_change_pivots(empty_series)
        assert isinstance(recognition_indices, np.ndarray)
        assert len(recognition_indices) == 0
        
        # Test with extremes
        recognition_indices, extreme_indices = directional_change_pivots(
            empty_series, return_extremes=True
        )
        assert isinstance(recognition_indices, np.ndarray)
        assert isinstance(extreme_indices, np.ndarray)
        assert len(recognition_indices) == 0
        assert len(extreme_indices) == 0
        
    def test_directional_change_pivots_single_value(self):
        """Test directional change with single value."""
        single_value = np.array([100])
        
        recognition_indices = directional_change_pivots(single_value)
        assert len(recognition_indices) == 1
        assert recognition_indices[0] == 0
        
    def test_directional_change_pivots_invalid_threshold_type(self, sample_price_series):
        """Test directional change with invalid threshold type."""
        with pytest.raises(ValueError, match="Invalid threshold_type"):
            directional_change_pivots(sample_price_series, threshold_type="invalid")
            
    def test_directional_change_pivots_missing_threshold_value(self, sample_price_series):
        """Test directional change with missing threshold value."""
        with pytest.raises(ValueError, match="threshold_value must be provided"):
            directional_change_pivots(sample_price_series, threshold_type="static", threshold_value=None)
            
    def test_directional_change_pivots_missing_ohlc_data(self, sample_price_series):
        """Test directional change with missing OHLC data for volatility threshold."""
        with pytest.raises(ValueError, match="high_series and low_series must be provided"):
            directional_change_pivots(sample_price_series, threshold_type="volatility")
            
    def test_directional_change_pivots_mismatched_lengths(self):
        """Test directional change with mismatched series lengths."""
        close_prices = np.array([100, 101, 102])
        high_prices = np.array([101, 102])  # Different length
        low_prices = np.array([99, 100, 101])
        
        with pytest.raises(ValueError, match="All price series must have the same length"):
            directional_change_pivots(
                close_prices,
                threshold_type="volatility",
                high_series=high_prices,
                low_series=low_prices
            )
            
    def test_directional_change_pivots_with_nans(self):
        """Test directional change with NaN values."""
        data_with_nans = np.array([100, np.nan, 105, 108, np.nan, 101, 98])
        recognition_indices = directional_change_pivots(data_with_nans, threshold_value=0.03)
        
        assert isinstance(recognition_indices, np.ndarray)
        # Should handle NaNs gracefully
        
    def test_directional_change_pivots_with_zeros(self):
        """Test directional change with zero values."""
        data_with_zeros = np.array([100, 0, 105, 108, 0, 101, 98])
        recognition_indices = directional_change_pivots(data_with_zeros, threshold_value=0.03)
        
        assert isinstance(recognition_indices, np.ndarray)
        # Should handle zeros gracefully
        
    def test_directional_change_pivots_different_thresholds(self, sample_price_series):
        """Test directional change with different threshold values."""
        thresholds = [0.01, 0.02, 0.05, 0.10]
        
        prev_count = float('inf')
        for threshold in thresholds:
            recognition_indices = directional_change_pivots(
                sample_price_series, threshold_value=threshold
            )
            current_count = len(recognition_indices)
            # Higher thresholds should generally produce fewer pivots
            assert current_count <= prev_count
            prev_count = current_count
            
    def test_directional_change_pivots_backward_compatibility(self, sample_price_series):
        """Test backward compatibility of return values."""
        # Old-style call (default return_extremes=False)
        old_result = directional_change_pivots(sample_price_series, threshold_value=0.03)
        
        # New-style call with return_extremes=True
        new_result = directional_change_pivots(
            sample_price_series, threshold_value=0.03, return_extremes=True
        )
        
        recognition_indices, extreme_indices = new_result
        
        # Old result should match recognition indices from new result
        assert np.array_equal(old_result, recognition_indices)
        assert isinstance(old_result, np.ndarray)
        assert isinstance(new_result, tuple)
        assert len(new_result) == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_functions_with_constant_data(self):
        """Test various functions with constant data where applicable."""
        constant_data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        
        # Functions that should handle constant data
        visual_stationary_test(constant_data)  # Should work
        
        # directional_change_pivots should work with constant data
        recognition_indices = directional_change_pivots(constant_data, threshold_value=0.01)
        assert len(recognition_indices) == 1  # Should return just the initial point
        
    def test_functions_with_very_small_data(self):
        """Test functions with very small datasets."""
        tiny_data = np.array([1.0, 2.0])
        
        # Most functions should handle tiny data gracefully
        visual_stationary_test(tiny_data)
        
        # Some functions may have minimum size requirements
        with pytest.raises(ValueError):
            adf_test(tiny_data)


if __name__ == "__main__":
    pytest.main([__file__])