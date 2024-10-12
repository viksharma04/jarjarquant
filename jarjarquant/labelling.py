# Imports
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
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
    
    @staticmethod
    def generate_random_price_series(periods: int = 252, volatility:float = 0.01, freq: str = 'B'):

        date_range = pd.date_range(start='2100-01-01', periods=periods, freq=freq)
        # TODO: Implement random series with distributions other than normal
        returns = np.random.normal(loc=0, scale=volatility, size=periods)
        prices = 100 * (1 + returns).cumprod()

        return pd.Series(prices, index=date_range, name='price')
    
    # def inverse_cumsum_filter(self, )