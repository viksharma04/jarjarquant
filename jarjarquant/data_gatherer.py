from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


class DataGatherer:

    def __init__(self):

        self.series = []

    def generate_random_normal(self, loc: float = 0.005, volatility: float = 0.05, periods: int = 252, freq: str = 'B', start: Optional[str] = '2100-01-01', name: Optional[str] = 'price', persist: Optional[bool] = False, **kwags):

        date_range = pd.date_range(start=start, periods=periods, freq=freq)
        returns = np.random.normal(
            loc=loc, scale=volatility, size=periods, **kwags)
        prices = 100 * (1 + returns).cumprod()

        if persist:
            self.series.append(pd.Series(prices, index=date_range, name=name))

        return pd.Series(prices, index=date_range, name=name)

    def get_yf_ticker(self, ticker: str = "SPY", **kwags):

        ticker = yf.Ticker(ticker)
        series = ticker.history(**kwags)['Close']

        return series
