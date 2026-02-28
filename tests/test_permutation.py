import numpy as np
import polars as pl
import pytest
from jarjarquant.permutation import BarPermute, PricePermute


def test_bar_permute_init():
    n = 50
    df = pl.DataFrame({
        "Open": np.random.randn(n).cumsum() + 100,
        "High": np.random.randn(n).cumsum() + 102,
        "Low": np.random.randn(n).cumsum() + 98,
        "Close": np.random.randn(n).cumsum() + 100,
    })
    bp = BarPermute(df)
    assert bp.basis_prices is not None
    assert bp.relative_prices is not None


def test_bar_permute_output_shape():
    n = 50
    close = np.cumsum(np.random.randn(n)) + 100
    df = pl.DataFrame({
        "Open": close + np.random.randn(n) * 0.1,
        "High": close + abs(np.random.randn(n)),
        "Low": close - abs(np.random.randn(n)),
        "Close": close,
    })
    bp = BarPermute(df)
    result = bp.permute()
    assert isinstance(result, pl.DataFrame)
    assert "Open" in result.columns
    assert "Close" in result.columns
    assert len(result) == n


def test_price_permute_init():
    prices = pl.Series("price", np.cumsum(np.random.randn(50)) + 100)
    pp = PricePermute(prices)
    assert pp.basis_prices is not None


def test_price_permute_output_length():
    prices = pl.Series("price", np.cumsum(np.random.randn(50)) + 100)
    pp = PricePermute(prices)
    result = pp.permute()
    assert isinstance(result, pl.Series)
    assert len(result) == 50
