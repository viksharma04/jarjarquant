import pytest
import pandas as pd
from jarjarquant.data_gatherer import DataGatherer


@pytest.fixture
def data_gatherer():
    return DataGatherer()


def test_generate_random_normal(data_gatherer):
    series = data_gatherer.generate_random_normal()
    assert isinstance(series, pd.Series)
    assert len(series) == 252
    assert series.name == 'price'


def test_get_yf_ticker():
    series = DataGatherer.get_yf_ticker("AAPL")
    assert isinstance(series, pd.DataFrame)
    assert not series.empty
    assert all(col in series.columns for col in [
               'Open', 'High', 'Low', 'Close', 'Volume'])


@pytest.mark.asyncio
async def test_get_tws_ticker():
    df = await DataGatherer.get_tws_ticker(ticker='AAPL', end_date='20031126 15:59:00 US/Eastern')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_get_random_price_samples_yf(data_gatherer):
    dataframes = data_gatherer.get_random_price_samples_yf()
    assert isinstance(dataframes, list)
    assert len(dataframes) == 30
    for df in dataframes:
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


@pytest.mark.asyncio
async def test_get_random_price_samples_tws(data_gatherer):
    dataframes = await data_gatherer.get_random_price_samples_tws(num_tickers_to_sample=1, years_in_sample=1)
    assert isinstance(dataframes, list)
    assert len(dataframes) == 1
    for df in dataframes:
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
