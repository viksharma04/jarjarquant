# data_gatherer/yahoo.py
import polars as pl
import yfinance as yf

from .base import DataSource, register_data_source


@register_data_source("yahoo")
class YFinanceDataSource(DataSource):
    async def fetch(self, ticker: str, **kwargs) -> pl.DataFrame:
        df = yf.Ticker(ticker).history(auto_adjust=True, **kwargs)
        return pl.from_pandas(df[["Open", "High", "Low", "Close", "Volume"]])
