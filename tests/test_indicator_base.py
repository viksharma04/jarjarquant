import numpy as np
import polars as pl
import pytest
from jarjarquant.indicators.base import Indicator, IndicatorSpec
from jarjarquant.indicators.registry import IndicatorType


class MockIndicator(Indicator):
    """Test indicator that returns a fixed array."""

    def _compute(self) -> np.ndarray:
        return np.array([1.0, 4.0, 9.0, 16.0])


def test_indicator_calculate_without_transform():
    df = pl.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100.0]})
    ind = MockIndicator(df)
    result = ind.calculate()
    np.testing.assert_array_equal(result, np.array([1.0, 4.0, 9.0, 16.0]))


def test_indicator_calculate_with_root_transform():
    df = pl.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100.0]})
    ind = MockIndicator(df)
    ind._transform = "root"
    result = ind.calculate()
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0, 4.0]))


def test_indicator_spec_creates_indicator():
    spec = IndicatorSpec(IndicatorType.RSI, {"period": 14})
    df = pl.DataFrame({
        "Open": np.random.randn(50).cumsum() + 100,
        "High": np.random.randn(50).cumsum() + 102,
        "Low": np.random.randn(50).cumsum() + 98,
        "Close": np.random.randn(50).cumsum() + 100,
        "Volume": np.random.rand(50) * 1000,
    })
    indicator = spec.create_indicator(df)
    result = indicator.calculate()
    assert isinstance(result, np.ndarray)


def test_indicator_spec_with_transform():
    spec = IndicatorSpec(IndicatorType.RSI, {"period": 14, "transform": "log"})
    assert spec.parameters.get("transform") == "log"
