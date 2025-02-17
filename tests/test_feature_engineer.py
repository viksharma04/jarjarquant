import pytest
import pandas as pd
import numpy as np
from jarjarquant.feature_engineer import BarPermute, PricePermute


@pytest.fixture
def price_series_list():
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data1 = np.cumsum(np.random.randn(100))
    data2 = np.cumsum(np.random.randn(100))
    series1 = pd.Series(data1, index=dates)
    series2 = pd.Series(data2, index=dates)
    return [series1, series2]


def test_price_permute_initialization(price_series_list):
    pp = PricePermute(price_series_list)
    assert pp.n_markets == 2
    assert len(pp.price_series_list) == 2
    assert pp.price_series_list[0].iloc[0] == price_series_list[0].iloc[0]
    assert pp.price_series_list[1].iloc[0] == price_series_list[1].iloc[0]


def test_price_permute_initialization_with_empty_list():
    with pytest.raises(ValueError):
        PricePermute([])


def test_price_permute_initialization_with_different_lengths(price_series_list):
    series3 = price_series_list[0][:-1]
    with pytest.raises(ValueError):
        PricePermute([price_series_list[0], series3])


def test_price_permute_permute(price_series_list):
    pp = PricePermute(price_series_list)
    shuffled_series_list = pp.permute()
    assert len(shuffled_series_list) == 2
    assert len(shuffled_series_list[0]) == 100
    assert len(shuffled_series_list[1]) == 100
    assert (shuffled_series_list[0].index == price_series_list[0].index).all()
    assert (shuffled_series_list[1].index == price_series_list[1].index).all()


@pytest.fixture
def ohlc_df_list():
    dates = pd.date_range('2023-01-01', periods=5)
    data1 = {
        'Open': [1, 2, 3, 4, 5],
        'High': [2, 3, 4, 5, 6],
        'Low': [0.5, 1.5, 2.5, 3.5, 4.5],
        'Close': [1.5, 2.5, 3.5, 4.5, 5.5]
    }
    data2 = {
        'Open': [10, 20, 30, 40, 50],
        'High': [20, 30, 40, 50, 60],
        'Low': [5, 15, 25, 35, 45],
        'Close': [15, 25, 35, 45, 55]
    }
    df1 = pd.DataFrame(data1, index=dates)
    df2 = pd.DataFrame(data2, index=dates)
    return [df1, df2]


def test_bar_permute_initialization(ohlc_df_list):
    bar_permute = BarPermute(ohlc_df_list)
    assert bar_permute.n_markets == 2
    assert bar_permute.original_index.tolist(
    ) == ohlc_df_list[0].index.tolist()
    assert len(bar_permute.rel_prices) == 2


def test_bar_permute_permute(ohlc_df_list):
    bar_permute = BarPermute(ohlc_df_list)
    shuffled_df_list = bar_permute.permute()
    assert len(shuffled_df_list) == 2
    for df in shuffled_df_list:
        assert len(df) == len(ohlc_df_list[0])
        assert all(df.columns == ['Open', 'High', 'Low', 'Close'])
        assert all(df.index == ohlc_df_list[0].index)


def test_bar_permute_invalid_initialization():
    with pytest.raises(ValueError):
        BarPermute([])


def test_bar_permute_permute_length_mismatch(ohlc_df_list):
    bar_permute = BarPermute(ohlc_df_list)
    bar_permute.original_index = bar_permute.original_index[:-1]
    with pytest.raises(ValueError):
        bar_permute.permute()
