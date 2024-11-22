from typing import Optional
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
import yfinance as yf
import concurrent.futures


class DataGatherer:

    def __init__(self):

        self.series = []

    def generate_random_normal(self, loc: float = 0.005, volatility: float = 0.05, periods: int = 252, freq: str = 'B', start: Optional[str] = '2100-01-01', name: Optional[str] = 'price', persist: Optional[bool] = False, **kwags):

        date_range = pd.date_range(start=start, periods=periods, freq=freq)
        returns = np.random.normal(
            loc=loc, scale=volatility, size=periods, **kwags)
        prices = 100 * (1 + returns).cumprod()

        if persist:
            self.series.append(pd.DataFrame(prices, index=date_range))

        return pd.Series(prices, index=date_range, name=name)

    @staticmethod
    def get_yf_ticker(ticker: str = "SPY", **kwags):

        ticker = yf.Ticker(ticker)
        series = ticker.history(
            auto_adjust=True, **kwags)[['Open', 'High', 'Low', 'Close', 'Volume']]

        return series

    def get_random_price_samples(self, years_back: int = 30, years_in_sample: int = 10, tickers: Optional[list] = None, num_tickers_to_sample: Optional[int] = 30, persist: Optional[bool] = False):

        # Set today's date (October 23, 2024) and the time range for the last 30 years
        today = datetime.today()
        start_limit = today - timedelta(days=years_back*365)
        num_years = timedelta(days=years_in_sample*365)

        # List of 30 stock tickers with replacements for problematic ones
        if tickers is None:
            tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'KO', 'MCD', 'NVDA', 'JPM', 'DIS',
                'BAC', 'CVX', 'INTC', 'CSCO', 'PEP', 'WMT', 'PG', 'ADBE', 'PFE',
                'XOM', 'T', 'NKE', 'MRK', 'ORCL', 'IBM', 'HON', 'BA', 'MMM', 'UNH',
                'GS', 'LMT', 'ABT', 'MO', 'AXP', 'CL', 'COP', 'DOW', 'GE', 'HD',
                'JNJ', 'TRV', 'VZ', 'WFC'
            ]

        tickers = random.sample(tickers, num_tickers_to_sample)

        # Function to get random start and end dates
        # TODO: Implement accepting a ticker and select start and end dates based on available price history
        def get_random_date():
            random_start = start_limit + \
                timedelta(days=np.random.randint(
                    0, years_back*365 - years_in_sample*365))
            random_end = random_start + num_years
            return random_start.strftime('%Y-%m-%d'), random_end.strftime('%Y-%m-%d')

        # List to store the DataFrames
        dataframes = []

        # Loop over each ticker and fetch 2 years of data with random start dates
        for ticker in tickers:
            start_date, end_date = get_random_date()
            df = self.get_yf_ticker(ticker, start=start_date, end=end_date)
            dataframes.append(df)

        if persist:
            self.series.append(dataframes)

        return dataframes
