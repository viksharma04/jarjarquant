import numpy as np
import pytest
from jarjarquant.volatility import calculate_volatility
from jarjarquant.schemas import VolatilityMeasure


def test_atr_volatility_output_shape():
    n = 100
    close = np.cumsum(np.random.randn(n)) + 100
    open_ = close + np.random.randn(n) * 0.1
    high = close + abs(np.random.randn(n)) + 0.5
    low = close - abs(np.random.randn(n)) - 0.5
    result = calculate_volatility(open_, high, low, close, VolatilityMeasure.ATR, n=14)
    assert len(result) == n


def test_ewm_std_volatility():
    close = np.cumsum(np.random.randn(100)) + 100
    open_ = close + np.random.randn(100) * 0.1
    high = close + 1
    low = close - 1
    result = calculate_volatility(open_, high, low, close, VolatilityMeasure.EWM_STD, n=20)
    assert len(result) == len(close)
    assert not np.all(np.isnan(result))


def test_invalid_measure_raises():
    with pytest.raises((ValueError, KeyError)):
        calculate_volatility(
            np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]),
            "invalid", n=10,
        )
