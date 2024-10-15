# Imports
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class Labelling:
    "Class to label data - implements commonn methods used during labelling for financial ml"

    def __init__(self, timeseries: Optional[pd.Series] = None):
        """Initialize Labelling

        Args:
            timeseries (Optional[pd.Series], optional): any timeseires with a datetime index. Defaults to None.
        """
        if timeseries is None:
            print("No time series provided - gnerating random price series")
            timeseries = self.generate_random_price_series()
        
        self.series = timeseries

    @classmethod
    def from_random_series(cls, periods: int, volatility:float, freq:str = 'B'):

        random_price_series = cls.generate_random_price_series(periods=periods, volatility=volatility, freq=freq)

        return cls(random_price_series)
    
    @classmethod
    def from_yf_ticker(cls, ticker: str = "SPY", **kwags):

        ticker = yf.Ticker(ticker)
        series = ticker.history(**kwags)['Close']
        
        return cls(series)

    
    @staticmethod
    def generate_random_price_series(periods: int = 252, loc:float = 0.001, volatility:float = 0.01, freq: str = 'B'):

        date_range = pd.date_range(start='2100-01-01', periods=periods, freq=freq)
        # TODO: Implement random series with distributions other than normal
        returns = np.random.normal(loc=loc, scale=volatility, size=periods)
        prices = 100 * (1 + returns).cumprod()

        return pd.Series(prices, index=date_range, name='price')
    
    @staticmethod
    def inverse_cumsum_filter(series: pd.Series, h: float, n: int) -> pd.Series:
        """
        Apply a cumulative sum filter to a time series based on a rolling period.
        
        Parameters:
        - series: pd.Series, time series of prices with time stamp index
        - h: float, threshold value for filtering
        - n: int, lookback period for the rolling window
        
        Returns:
        - pd.Series, boolean series where True indicates dates flagged by the filter
        """
        returns = series.pct_change()       
        # Ensure the series is sorted by index (time)
        returns = returns.add(1)
        
        # Calculate the rolling cumulative sum over the lookback period n
        rolling_cumsum = returns.rolling(window=n).apply(np.prod) -1
        
        # Flag dates where the cumulative return is less than the absolute value of h
        flagged = (rolling_cumsum.abs() < h)
        
        return flagged