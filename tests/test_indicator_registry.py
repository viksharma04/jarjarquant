"""
Tests for the indicator registry system.

This module tests the new indicator registry functionality, including
IndicatorType enum, registration, and the updated add_indicator method.
"""

import pytest
import pandas as pd
import numpy as np
from jarjarquant.indicators.registry import (
    IndicatorType,
    get_indicator_class,
    list_available_indicators,
    is_indicator_registered,
    INDICATOR_REGISTRY
)
from jarjarquant.indicators import RSI, MACD, ADX, Stochastic
from jarjarquant.jarjarquant import Jarjarquant


class TestIndicatorRegistry:
    """Test the indicator registry functionality."""
    
    def test_indicator_registry_populated(self):
        """Test that the registry is populated with indicators."""
        assert len(INDICATOR_REGISTRY) > 0, "Registry should not be empty"
        
        # Check for some key indicators
        expected_indicators = [
            IndicatorType.RSI,
            IndicatorType.MACD, 
            IndicatorType.ADX,
            IndicatorType.STOCHASTIC
        ]
        
        for indicator in expected_indicators:
            assert indicator in INDICATOR_REGISTRY, f"{indicator} should be registered"
    
    def test_get_indicator_class(self):
        """Test getting indicator classes from the registry."""
        # Test valid indicator types
        rsi_class = get_indicator_class(IndicatorType.RSI)
        assert rsi_class == RSI, "Should return RSI class"
        
        macd_class = get_indicator_class(IndicatorType.MACD)
        assert macd_class == MACD, "Should return MACD class"
    
    def test_get_indicator_class_invalid(self):
        """Test error handling for invalid indicator types."""
        # Create a mock enum that's not registered
        class MockIndicatorType:
            INVALID = "invalid"
        
        with pytest.raises(KeyError) as exc_info:
            get_indicator_class(MockIndicatorType.INVALID)
        
        assert "not found in registry" in str(exc_info.value)
    
    def test_list_available_indicators(self):
        """Test listing all available indicators."""
        available = list_available_indicators()
        assert isinstance(available, list), "Should return a list"
        assert len(available) > 0, "Should have indicators available"
        assert IndicatorType.RSI in available, "RSI should be available"
    
    def test_is_indicator_registered(self):
        """Test checking if an indicator is registered."""
        assert is_indicator_registered(IndicatorType.RSI) == True
        assert is_indicator_registered(IndicatorType.MACD) == True
        
        # Test with a mock unregistered type
        class MockIndicatorType:
            UNREGISTERED = "unregistered"
        
        assert is_indicator_registered(MockIndicatorType.UNREGISTERED) == False


class TestJarjarquantAddIndicator:
    """Test the updated add_indicator method in Jarjarquant."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = {
            'Open': prices * (1 + np.random.normal(0, 0.001, 100)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'Close': prices,
            'Volume': np.random.randint(10000, 100000, 100)
        }
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def jjq_instance(self, sample_data):
        """Create a Jarjarquant instance with sample data."""
        jjq = Jarjarquant()
        jjq._df = sample_data
        return jjq
    
    def test_add_indicator_rsi(self, jjq_instance):
        """Test adding RSI indicator."""
        initial_columns = len(jjq_instance._df.columns)
        
        # Add RSI indicator
        jjq_instance.add_indicator(IndicatorType.RSI, "rsi_14", period=14)
        
        # Check that column was added
        assert len(jjq_instance._df.columns) == initial_columns + 1
        assert "rsi_14" in jjq_instance._df.columns
        
        # Check that values exist (this RSI is normalized: (RSI - 50) / 10)
        rsi_values = jjq_instance._df["rsi_14"].dropna()
        assert len(rsi_values) > 0, "Should have RSI values"
        # The normalized RSI should be centered around 0
        assert not rsi_values.isna().all(), "RSI values should not be all NaN"
    
    def test_add_indicator_macd(self, jjq_instance):
        """Test adding MACD indicator."""
        initial_columns = len(jjq_instance._df.columns)
        
        # Add MACD indicator
        jjq_instance.add_indicator(IndicatorType.MACD, "macd", short_period=12, long_period=26)
        
        # Check that column was added
        assert len(jjq_instance._df.columns) == initial_columns + 1
        assert "macd" in jjq_instance._df.columns
        
        # Check that we have MACD values
        macd_values = jjq_instance._df["macd"].dropna()
        assert len(macd_values) > 0, "Should have MACD values"
    
    def test_add_indicator_stochastic(self, jjq_instance):
        """Test adding Stochastic indicator."""
        initial_columns = len(jjq_instance._df.columns)
        
        # Add Stochastic indicator (using correct parameter name 'lookback')
        jjq_instance.add_indicator(IndicatorType.STOCHASTIC, "stoch", lookback=14, n_smooth=3)
        
        # Check that column was added
        assert len(jjq_instance._df.columns) == initial_columns + 1
        assert "stoch" in jjq_instance._df.columns
        
        # Check that we have Stochastic values
        stoch_values = jjq_instance._df["stoch"].dropna()
        assert len(stoch_values) > 0, "Should have Stochastic values"
        assert not stoch_values.isna().all(), "Stochastic values should not be all NaN"
    
    def test_add_multiple_indicators(self, jjq_instance):
        """Test adding multiple indicators."""
        initial_columns = len(jjq_instance._df.columns)
        
        # Add multiple indicators
        jjq_instance.add_indicator(IndicatorType.RSI, "rsi_14", period=14)
        jjq_instance.add_indicator(IndicatorType.RSI, "rsi_21", period=21)
        jjq_instance.add_indicator(IndicatorType.MACD, "macd", short_period=12, long_period=26)
        
        # Check that all columns were added
        assert len(jjq_instance._df.columns) == initial_columns + 3
        assert "rsi_14" in jjq_instance._df.columns
        assert "rsi_21" in jjq_instance._df.columns
        assert "macd" in jjq_instance._df.columns
    
    def test_add_indicator_empty_dataframe(self):
        """Test error handling for empty DataFrame."""
        jjq = Jarjarquant()
        # _df is empty by default
        
        with pytest.raises(ValueError) as exc_info:
            jjq.add_indicator(IndicatorType.RSI, "rsi", period=14)
        
        assert "DataFrame is empty" in str(exc_info.value)
    
    def test_add_indicator_invalid_type(self, jjq_instance):
        """Test error handling for invalid indicator type."""
        class MockIndicatorType:
            INVALID = "invalid"
        
        with pytest.raises(KeyError) as exc_info:
            jjq_instance.add_indicator(MockIndicatorType.INVALID, "invalid_indicator")
        
        assert "not found in registry" in str(exc_info.value)
    
    def test_add_indicator_custom_parameters(self, jjq_instance):
        """Test adding indicators with custom parameters."""
        # Test RSI with different period
        jjq_instance.add_indicator(IndicatorType.RSI, "rsi_30", period=30)
        assert "rsi_30" in jjq_instance._df.columns
        
        # Test MACD with custom parameters
        jjq_instance.add_indicator(
            IndicatorType.MACD, "macd_fast", 
            short_period=5, long_period=20, smoothing_factor=3
        )
        assert "macd_fast" in jjq_instance._df.columns


class TestIndicatorTypeEnum:
    """Test the IndicatorType enum."""
    
    def test_indicator_type_values(self):
        """Test that IndicatorType has expected values."""
        expected_types = [
            "rsi", "macd", "adx", "aroon", "stochastic", 
            "stochastic_rsi", "chaikin_money_flow", "detrended_rsi",
            "cmma", "moving_average_difference", "price_change_oscillator",
            "price_intensity", "regression_trend", "regression_trend_deviation"
        ]
        
        actual_values = [indicator.value for indicator in IndicatorType]
        
        for expected in expected_types:
            assert expected in actual_values, f"{expected} should be in IndicatorType"
    
    def test_indicator_type_string_representation(self):
        """Test string representation of IndicatorType."""
        assert IndicatorType.RSI.value == "rsi"
        assert IndicatorType.MACD.value == "macd"
        assert IndicatorType.STOCHASTIC.value == "stochastic"


if __name__ == "__main__":
    pytest.main([__file__])