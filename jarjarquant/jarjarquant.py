"""Main module for jarjarquant package."""
import pandas as pd
from .data_gatherer import DataGatherer
from .feature_engineer import FeatureEngineer
from .labeller import Labeller
from .data_analyst import DataAnalyst

from .indicator import RSI, DetrendedRSI, Stochastic, StochasticRSI, CMMA, MovingAverageDifference, MACD, RegressionTrend, PriceIntensity, ADX, Aroon


class Jarjarquant:
    """
    Jarjarquant integrates data gathering, labeling, and feature engineering for financial time series.
    """

    def __init__(self, ohlcv_df=None):
        self.data_gatherer = DataGatherer()
        if ohlcv_df is None:
            samples = self.data_gatherer.get_random_price_samples(
                num_tickers_to_sample=1)
            if not samples:
                raise ValueError(
                    "No price samples were returned. Please check the data source.")
            self._df = samples[0]
        else:
            self._df = ohlcv_df
        self.labeller = Labeller(self._df)
        self.feature_engineer = FeatureEngineer()
        self.data_analyst = DataAnalyst()

    @classmethod
    def from_random_normal(cls, loc: float = 0.005, volatility: float = 0.05, periods: int = 252, **kwargs):
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
            loc=loc, volatility=volatility, periods=periods, **kwargs)
        return cls(series)

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
            raise ValueError(f"Failed to fetch data for ticker '{
                             ticker}'. Error: {e}") from e
        return cls(series)

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

    # Add any additional methods or functionality here
    @staticmethod
    def rsi(ohlcv_df, period: int = 14):
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
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input DataFrame must contain a 'Close' column for RSI calculation.")
        rsi_indicator = RSI(_df, period)
        return rsi_indicator

    def add_rsi(self, period: int = 14):
        """
        Adds the Relative Strength Index (RSI) to the DataFrame.

        Parameters:
        period (int): The period to calculate the RSI. Default is 14.

        Returns:
        None: The method modifies the DataFrame in place by adding a new column 'RSI'.
        """
        self._df = self._df.assign(RSI=self.rsi(
            ohlcv_df=self._df, period=period))

    @staticmethod
    def detrended_rsi(ohlcv_df, short_period: int = 2, long_period: int = 21, regression_length: int = 120):
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
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input DataFrame must contain a 'Close' column for Detrended RSI calculation.")
        detrended_rsi_indicator = DetrendedRSI(
            _df, short_period, long_period, regression_length)
        return detrended_rsi_indicator

    def add_detrended_rsi(self, short_period: int = 2, long_period: int = 21, regression_length: int = 120):
        """
        Adds the Detrended RSI (Relative Strength Index) to the DataFrame.

        The Detrended RSI is calculated using the specified short period, long period, 
        and regression length. The result is assigned to a new column 'Detrended_RSI' 
        in the DataFrame.

        Parameters:
        short_period (int): The short period for calculating the RSI. Default is 2.
        long_period (int): The long period for calculating the RSI. Default is 21.
        regression_length (int): The length of the regression for detrending. Default is 120.

        Returns:
        None
        """
        self._df = self._df.assign(Detrended_RSI=self.detrended_rsi(
            ohlcv_df=self._df, short_period=short_period, long_period=long_period, regression_length=regression_length))

    @staticmethod
    def stochastic(ohlcv_df: pd.DataFrame, period: int = 14, n_smooth: int = 2):
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
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Stochastic calculation")
        stochastic_indicator = Stochastic(_df, period, n_smooth)

        return stochastic_indicator

    def add_stochastic(self, period: int = 14, n_smooth: int = 2):
        """
        Adds the Stochastic indicator to the DataFrame.

        The Stochastic indicator is a momentum indicator comparing a particular closing price 
        of a security to a range of its prices over a certain period of time.

        Args:
            period (int): The number of periods to use for the Stochastic calculation. Default is 14.
            n_smooth (int): The number of periods to use for smoothing the Stochastic values. Default is 2.

        Returns:
            None: The method modifies the DataFrame in place by adding a 'Stochastic' column.
        """
        self._df = self._df.assign(Stochastic=self.stochastic(
            ohlcv_df=self._df, period=period, n_smooth=n_smooth))

    @staticmethod
    def stochastic_rsi(ohlcv_df: pd.DataFrame, rsi_period: int = 14, stochastic_period: int = 14, n_smooth: int = 2):
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
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Stochastic RSI calculation")
        stochastic_rsi_indicator = StochasticRSI(
            ohlcv_df, rsi_period, stochastic_period, n_smooth)

        return stochastic_rsi_indicator

    def add_stochastic_rsi(self, rsi_period: int = 14, stochastic_period: int = 14, n_smooth: int = 2):
        """
        Adds the Stochastic RSI indicator to the DataFrame.

        Parameters:
        rsi_period (int): The period for calculating the RSI. Default is 14.
        stochastic_period (int): The period for calculating the Stochastic RSI. Default is 14.
        n_smooth (int): The smoothing factor for the Stochastic RSI. Default is 2.

        Returns:
        None: The method modifies the DataFrame in place by adding a new column 'Stochastic_RSI'.
        """
        self._df = self._df.assign(Stochastic_RSI=self.stochastic_rsi(
            ohlcv_df=self._df, rsi_period=rsi_period, stochastic_period=stochastic_period, n_smooth=n_smooth))

    @staticmethod
    def moving_average_difference(ohlcv_df: pd.DataFrame, short_period: int = 5, long_period: int = 20):
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
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Moving Average Difference calculation")
        mad_indicator = MovingAverageDifference(_df, short_period, long_period)

        return mad_indicator

    def add_moving_average_difference(self, short_period: int = 5, long_period: int = 20):
        """
        Adds the Moving Average Difference column to the DataFrame.

        Parameters:
            short_period (int): The short moving average window. Default is 5.
            long_period (int): The long moving average window. Default is 20.

        Returns:
            None: The method modifies the DataFrame in place by adding a new column 'Moving_Average_Difference'.
        """
        self._df = self._df.assign(Moving_Average_Difference=self.moving_average_difference(
            ohlcv_df=self._df, short_period=short_period, long_period=long_period))

    @staticmethod
    def cmma(ohlcv_df: pd.DataFrame, lookback: int = 21, atr_length: int = 21):
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
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for CMMA calculation")
        cmma_indicator = CMMA(_df, lookback, atr_length)

        return cmma_indicator

    def add_cmma(self, lookback: int = 21, atr_length: int = 21):
        """
        Adds the CMMA (Custom Moving Average) column to the DataFrame.

        Parameters:
        lookback (int): The lookback period for the CMMA calculation. Default is 21.
        atr_length (int): The length of the ATR (Average True Range) period used in the CMMA calculation. Default is 21.

        Returns:
        None: The method modifies the DataFrame in place by adding a new column 'CMMA'.
        """
        self._df = self._df.assign(CMMA=self.cmma(
            ohlcv_df=self._df, lookback=lookback, atr_length=atr_length))

    @staticmethod
    def macd(ohlcv_df: pd.DataFrame, short_period: int = 5, long_period: int = 20, smoothing_factor: int = 2):
        _df = ohlcv_df.copy()
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for MACD calculation")
        macd_indicator = MACD(_df, short_period, long_period, smoothing_factor)

        return macd_indicator

    def add_macd(self, short_period: int = 5, long_period: int = 20, smoothing_factor: int = 2):
        self._df = self._df.assign(MACD=self.macd(
            ohlcv_df=self._df, short_period=short_period, long_period=long_period, smoothing_factor=smoothing_factor))

    @staticmethod
    def regression_trend(ohlcv_df: pd.DataFrame, lookback: int = 21, atr_length_mult: int = 3, degree: int = 1):
        _df = ohlcv_df.copy()
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Regression Trend calculation")
        regression_trend_indicator = RegressionTrend(
            _df, lookback, atr_length_mult, degree)

        return regression_trend_indicator

    def add_regression_trend(self, lookback: int = 21, atr_length_mult: int = 3, degree: int = 1):
        self._df = self._df.assign(Regression_Trend=self.regression_trend(
            ohlcv_df=self._df, lookback=lookback, atr_length_mult=atr_length_mult, degree=degree))

    @staticmethod
    def price_intensity(ohlcv_df: pd.DataFrame, smoothing_factor: int = 2):
        _df = ohlcv_df.copy()
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Price Intensity calculation")
        price_intensity_indicator = PriceIntensity(_df, smoothing_factor)

        return price_intensity_indicator

    def add_price_intensity(self, smoothing_factor: int = 2):
        self._df = self._df.assign(Price_Intensity=self.price_intensity(
            ohlcv_df=self._df, smoothing_factor=smoothing_factor))

    @staticmethod
    def adx(ohlcv_df: pd.DataFrame, lookback: int = 14):
        _df = ohlcv_df.copy()
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Price Intensity calculation")
        adx_indicator = ADX(_df, lookback)

        return adx_indicator

    def add_adx(self, lookback: int = 14):
        self._df = self._df.assign(ADX=self.adx(
            ohlcv_df=self._df, lookback=lookback))

    @staticmethod
    def aroon(ohlcv_df: pd.DataFrame, lookback: int = 14):
        _df = ohlcv_df.copy()
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for Aroon calculation")
        aroon_indicator = Aroon(_df, lookback)

        return aroon_indicator

    def add_aroon(self, lookback: int = 14):
        self._df = self._df.assign(Aroon=self.aroon(
            ohlcv_df=self._df, lookback=lookback))
