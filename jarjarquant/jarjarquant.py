"""Main module for jarjarquant package."""

import importlib
import inspect
import json
import pkgutil
from typing import Optional

import pandas as pd

import jarjarquant.indicators as indicators_pkg
from jarjarquant.indicators import (
    ADX,
    CMMA,
    MACD,
    RSI,
    Aroon,
    ChaikinMoneyFlow,
    DetrendedRSI,
    MovingAverageDifference,
    PriceChangeOscillator,
    PriceIntensity,
    RegressionTrend,
    RegressionTrendDeviation,
    Stochastic,
    StochasticRSI,
)

from .data_analyst import DataAnalyst
from .data_gatherer import DataGatherer
from .feature_engineer import FeatureEngineer
from .feature_evaluator import FeatureEvaluator
from .labeller import Labeller


class Jarjarquant(Labeller):
    """
    Jarjarquant integrates data gathering, labeling, and feature engineering for financial time series.
    """

    def __init__(
        self,
        data_frame: Optional[pd.DataFrame] = None,
        data_source: Optional[str] = None,
    ):
        """_summary_

        Args:
            ohlcv_df (pd.DataFrame): _description_
            data_source (Optional[str]): _description_

        Raises:
            ValueError: _description_
        """
        if data_frame is None and data_source is None:
            raise TypeError("Provide a data frame or a data source ('tws' or 'yf')")

        if data_frame is not None:
            self._df = data_frame
        else:
            self.data_gatherer = DataGatherer()
            try:
                samples = (
                    self.data_gatherer.get_random_price_samples_tws(
                        num_tickers_to_sample=1
                    )
                    if data_source == "tws"
                    else self.data_gatherer.get_random_price_samples_yf(
                        num_tickers_to_sample=1
                    )
                )
                self._df = samples[0]
            except Exception as e:
                raise ValueError(
                    f"Unable to fetch price sample from {data_source}: {e}"
                )

        # Super init
        super().__init__(self._df)
        self.feature_engineer = FeatureEngineer()
        self.data_analyst = DataAnalyst()
        self.feature_evaluator = FeatureEvaluator()

    @classmethod
    def from_random_normal(
        cls, loc: float = 0.005, volatility: float = 0.05, periods: int = 252, **kwargs
    ):
        """
        Create a random price series using returns from a normal distribution.

        Args:
            loc (float): Mean return. Defaults to 0.005.
            volatility (float): Period volatility. Defaults to 0.05.
            periods (int): Number of data points. Defaults to 252.

        Returns:
            Jarjarquant: Instance of Jarjarquant with generated series.
        """
        data_gatherer = DataGatherer()
        series = data_gatherer.generate_random_normal(
            loc=loc, volatility=volatility, periods=periods, **kwargs
        )
        return cls(series, data_source=None)

    @classmethod
    def from_random_sample(
        cls, num_tickers_to_sample: int = 1, years_in_sample: int = 10, **kwargs
    ):
        """
        Create a random price series using a random sample of tickers.

        Args:
            num_tickers_to_sample (int): Number of tickers to sample. Defaults to 1.

        Returns:
            Jarjarquant: Instance of Jarjarquant with generated series.
        """
        data_gatherer = DataGatherer()
        samples = data_gatherer.get_random_price_samples_tws(
            num_tickers_to_sample=num_tickers_to_sample,
            years_in_sample=years_in_sample,
            **kwargs,
        )
        if not samples:
            raise ValueError(
                "No price samples were returned. Please check the data source."
            )
        return cls(samples[0], data_source=None)

    @classmethod
    def from_yf_ticker(cls, ticker: str = "SPY", **kwargs):
        """
        Initialize from a Yahoo Finance ticker.

        Args:
            ticker (str): Ticker symbol. Defaults to "SPY".

        Returns:
            Jarjarquant: Instance of Jarjarquant with data from the ticker.
        """
        data_gatherer = DataGatherer()  # or access via composition
        try:
            series = data_gatherer.get_yf_ticker(ticker, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to fetch data for ticker '{ticker}'. Error: {e}"
            ) from e
        return cls(series, data_source=None)

    @property
    def df(self):
        """
        Returns the DataFrame stored in the instance.

        Returns:
            pandas.DataFrame: The DataFrame stored in the instance.
        """
        return self._df

    @df.setter
    def df(self, df):
        """
        Sets the DataFrame for the instance.

        Parameters:
        df (pd.DataFrame): The DataFrame to be set. Must be an instance of pandas DataFrame.

        Raises:
        ValueError: If the provided df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Must be a pandas DataFrame!")
        self._df = df

    @df.deleter
    def df(self):
        """
        Deletes the attribute `_df` from the instance.

        This method removes the `_df` attribute from the instance, effectively
        deleting any data stored in it. Use this method with caution as it will
        permanently remove the attribute and its contents.

        Raises:
            AttributeError: If the `_df` attribute does not exist.
        """
        del self._df

    def list_indicators(self) -> str:
        """
        Discover all classes in jarjarquant.indicators and
        return their __init__ signature parameters (excluding self & ohlcv_df)
        as a JSON string.
        """
        result = {}
        for _, module_name, _ in pkgutil.iter_modules(indicators_pkg.__path__):
            module = importlib.import_module(f"{indicators_pkg.__name__}.{module_name}")
            for cls_name, cls in inspect.getmembers(module, inspect.isclass):
                # only own classes (skip imported ones)
                if cls.__module__ != module.__name__:
                    continue
                if cls_name in ("Indicator", "IndicatorEvalResult"):
                    continue
                sig = inspect.signature(cls.__init__)
                # drop 'self' and the first 'ohlcv_df' argument
                params = []
                for p in list(sig.parameters.values())[2:]:
                    default = p.default if p.default is not inspect._empty else None
                    ann = (
                        p.annotation.__name__
                        if hasattr(p.annotation, "__name__")
                        else str(p.annotation)
                        if p.annotation is not inspect._empty
                        else None
                    )
                    params.append({"name": p.name, "type": ann, "default": default})
                result[cls_name] = params
        # pretty-print JSON
        return json.dumps(result, indent=2)

    # Indicator methods
    # Add any additional methods or functionality here
    @staticmethod
    def rsi(ohlcv_df, period: int = 14, transform=None):
        """
        Calculate the Relative Strength Index (RSI) for a given OHLCV DataFrame.

        Parameters:
        ohlcv_df (pd.DataFrame): DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
                                 Must include a 'Close' column.
        period (int): The period over which to calculate the RSI. Default is 14.

        Returns:
        pd.Series: A pandas Series containing the RSI values.

        Raises:
        ValueError: If the input DataFrame does not contain a 'Close' column.
        """
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input DataFrame must contain a 'Close' column for RSI calculation."
            )
        rsi_indicator = RSI(_df, period, transform)
        return rsi_indicator

    @staticmethod
    def detrended_rsi(
        ohlcv_df,
        short_period: int = 2,
        long_period: int = 21,
        regression_length: int = 120,
        transform=None,
    ):
        """
        Calculate the Detrended Relative Strength Index (RSI) for a given OHLCV DataFrame.

        Parameters:
        ohlcv_df (pd.DataFrame): DataFrame containing OHLCV data. Must include a 'Close' column.
        short_period (int, optional): The short period for RSI calculation. Default is 2.
        long_period (int, optional): The long period for RSI calculation. Default is 21.
        regression_length (int, optional): The length of the regression window. Default is 120.

        Returns:
        pd.Series: A pandas Series containing the Detrended RSI values.

        Raises:
        ValueError: If the input DataFrame does not contain a 'Close' column.
        """
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input DataFrame must contain a 'Close' column for Detrended RSI calculation."
            )
        detrended_rsi_indicator = DetrendedRSI(
            _df, short_period, long_period, regression_length, transform
        )
        return detrended_rsi_indicator

    @staticmethod
    def stochastic(
        ohlcv_df: pd.DataFrame, period: int = 14, n_smooth: int = 2, transform=None
    ):
        """
        Calculate the Stochastic indicator for the given OHLCV DataFrame.
        Parameters:
        ohlcv_df (pd.DataFrame): DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
        period (int, optional): The look-back period for the Stochastic calculation. Default is 14.
        n_smooth (int, optional): The smoothing factor for the Stochastic calculation. Default is 2.
        Returns:
        pd.DataFrame: DataFrame containing the Stochastic indicator values.
        Raises:
        ValueError: If the input DataFrame does not contain a 'Close' column.
        """
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Stochastic calculation"
            )
        stochastic_indicator = Stochastic(_df, period, n_smooth, transform)

        return stochastic_indicator

    @staticmethod
    def stochastic_rsi(
        ohlcv_df: pd.DataFrame,
        rsi_period: int = 14,
        stochastic_period: int = 14,
        n_smooth: int = 2,
        transform=None,
    ):
        """
        Calculate the Stochastic RSI (Relative Strength Index) for a given OHLCV (Open, High, Low, Close, Volume) DataFrame.

        Parameters:
            ohlcv_df (pd.DataFrame): DataFrame containing OHLCV data. Must include a 'Close' column.
            rsi_period (int): The period for calculating the RSI. Default is 14.
            stochastic_period (int): The period for calculating the Stochastic RSI. Default is 14.
            n_smooth (int): The smoothing factor for the Stochastic RSI. Default is 2.

        Returns:
            pd.Series: A pandas Series containing the Stochastic RSI values.

        Raises:
            ValueError: If the input DataFrame does not contain a 'Close' column.
        """
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Stochastic RSI calculation"
            )
        stochastic_rsi_indicator = StochasticRSI(
            ohlcv_df, rsi_period, stochastic_period, n_smooth, transform
        )

        return stochastic_rsi_indicator

    @staticmethod
    def moving_average_difference(
        ohlcv_df: pd.DataFrame,
        short_period: int = 5,
        long_period: int = 20,
        transform=None,
    ):
        """
        Calculate the Moving Average Difference for the given OHLCV DataFrame.

        Parameters:
            ohlcv_df (pd.DataFrame): A pandas DataFrame containing OHLCV data.
                                 Must include a 'Close' column.
            short_period (int): The short moving average window. Default is 5.
            long_period (int): The long moving average window. Default is 20.

        Returns:
            pd.Series: A pandas Series containing the Moving Average Difference values.

        Raises:
            ValueError: If the input DataFrame does not contain a 'Close' column.
        """
        if short_period >= long_period:
            raise ValueError("short_window must be less than long_window")

        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Moving Average Difference calculation"
            )
        mad_indicator = MovingAverageDifference(
            _df, short_period, long_period, transform
        )

        return mad_indicator

    @staticmethod
    def cmma(
        ohlcv_df: pd.DataFrame, lookback: int = 21, atr_length: int = 21, transform=None
    ):
        """
        Calculate the Custom Moving Average (CMMA) for the given OHLCV DataFrame.

        Parameters:
            ohlcv_df (pd.DataFrame): A pandas DataFrame containing OHLCV data.
                                 Must include a 'Close' column.
            lookback (int): The lookback period for the CMMA calculation. Default is 21.
            atr_length (int): The length of the Average True Range (ATR) period. Default is 21.

        Returns:
            pd.Series: A pandas Series containing the CMMA values.

        Raises:
            ValueError: If the input DataFrame does not contain a 'Close' column.
        """
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for CMMA calculation"
            )
        cmma_indicator = CMMA(_df, lookback, atr_length, transform)

        return cmma_indicator

    @staticmethod
    def macd(
        ohlcv_df: pd.DataFrame,
        short_period: int = 5,
        long_period: int = 20,
        smoothing_factor: int = 2,
        transform: Optional[bool] = None,
    ):
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for MACD calculation"
            )
        if transform is None:
            transform = False

        macd_indicator = MACD(
            _df, short_period, long_period, smoothing_factor, transform
        )

        return macd_indicator

    @staticmethod
    def regression_trend(
        ohlcv_df: pd.DataFrame,
        lookback: int = 21,
        atr_length_mult: int = 3,
        degree: int = 1,
        transform=None,
    ):
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Regression Trend calculation"
            )
        regression_trend_indicator = RegressionTrend(
            _df, lookback, atr_length_mult, degree, transform
        )

        return regression_trend_indicator

    @staticmethod
    def price_intensity(
        ohlcv_df: pd.DataFrame, smoothing_factor: int = 2, transform=None
    ):
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Price Intensity calculation"
            )
        price_intensity_indicator = PriceIntensity(_df, smoothing_factor, transform)

        return price_intensity_indicator

    @staticmethod
    def adx(ohlcv_df: pd.DataFrame, lookback: int = 14, transform=None):
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Price Intensity calculation"
            )
        adx_indicator = ADX(_df, lookback, transform)

        return adx_indicator

    @staticmethod
    def aroon(ohlcv_df: pd.DataFrame, lookback: int = 14, transform=None):
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Aroon calculation"
            )
        aroon_indicator = Aroon(_df, lookback, transform)

        return aroon_indicator

    @staticmethod
    def regression_trend_deviation(
        ohlcv_df: pd.DataFrame, lookback: int = 14, fit_degree: int = 1, transform=None
    ):
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Regression Trend Deviation calculation"
            )
        regression_trend_deviation_indicator = RegressionTrendDeviation(
            _df, lookback, fit_degree, transform
        )

        return regression_trend_deviation_indicator

    @staticmethod
    def pco(
        ohlcv_df: pd.DataFrame,
        short_lookback: int = 5,
        long_lookback_multiplier: int = 3,
        transform=None,
    ):
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Price Channel Oscillator calculation"
            )
        pco_indicator = PriceChangeOscillator(
            _df, short_lookback, long_lookback_multiplier, transform
        )

        return pco_indicator

    @staticmethod
    def chaikin_money_flow(
        ohlcv_df: pd.DataFrame,
        smoothing_lookback: int = 21,
        volume_lookback: int = 21,
        return_cmf: bool = False,
        transform=None,
    ):
        _df = ohlcv_df.copy()
        if "Close" not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Chaikin Money Flow calculation"
            )
        cmf_indicator = ChaikinMoneyFlow(
            _df, smoothing_lookback, volume_lookback, return_cmf, transform
        )

        return cmf_indicator

    def add_indicator(self, indicator_func, column_name: str, *args, **kwargs):
        """
        Generic method to add an indicator column to the DataFrame.

        Parameters:
            indicator_func (callable): The indicator function to compute the indicator.
            column_name (str): The name of the column to add.
            *args: Positional arguments to pass to the indicator function.
            **kwargs: Keyword arguments to pass to the indicator function.

        Returns:
            None: The method modifies the DataFrame in place by adding the new column.

        Example usage:
            # Add RSI indicator
            jq.add_indicator(Jarjarquant.rsi, "rsi", period=14)
            # Add Stochastic indicator
            jq.add_indicator(Jarjarquant.stochastic, "stochastic", period=14, n_smooth=2)
        """
        self._df = self._df.assign(
            **{column_name: indicator_func(self._df, *args, **kwargs).calculate()}
        )
