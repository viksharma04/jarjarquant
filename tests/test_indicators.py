import numpy as np
import polars as pl
import pytest
from jarjarquant.indicators.registry import IndicatorType, list_available_indicators, get_indicator_class


def _make_ohlcv(n: int = 200) -> pl.DataFrame:
    close = np.cumsum(np.random.randn(n)) + 100
    return pl.DataFrame({
        "Open": close + np.random.randn(n) * 0.5,
        "High": close + abs(np.random.randn(n)),
        "Low": close - abs(np.random.randn(n)),
        "Close": close,
        "Volume": (np.random.rand(n) * 1e6).astype(np.float64),
    })


@pytest.mark.parametrize("indicator_type", list(IndicatorType))
def test_every_indicator_calculates(indicator_type):
    """Every registered indicator must produce an ndarray without crashing."""
    cls = get_indicator_class(indicator_type)
    df = _make_ohlcv(200)
    indicator = cls(df)
    result = indicator.calculate()
    assert isinstance(result, np.ndarray)
    assert len(result) == 200


@pytest.mark.parametrize("indicator_type", list(IndicatorType))
def test_every_indicator_with_transform(indicator_type):
    """Every indicator must work with a transform applied."""
    cls = get_indicator_class(indicator_type)
    df = _make_ohlcv(200)
    indicator = cls(df)
    indicator._transform = "root"
    result = indicator.calculate()
    assert isinstance(result, np.ndarray)
    assert len(result) == 200
