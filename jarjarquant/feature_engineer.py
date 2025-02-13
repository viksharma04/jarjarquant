"""Feature engineer specializes in engineering features, it transforms and augments raw data/price series to extract more information.
Unlike an indicator, these transformations are repeatable and non-specific to a market hypothesis"""
from typing import Optional
import scipy
import pandas as pd
import numpy as np
import concurrent.futures


class FeatureEngineer:
    """Class to implement common featrue engineering transformations"""

    def __init__(self, features_df: pd.DataFrame = None):
        """Initialize Featureengineering

        Args:
            features_df (pd.DataFrame): 
                a pandas dataframe containg the features timeseries with pd.DateTime index
        """
        self.features_df = features_df

    @staticmethod
    def get_weights(d, size):
        """
        Calculate the weights for fractional differentiation.

        Parameters:
        -----------
        d : float
            The order of differentiation. Determines the depth of the fractional differentiation.
        size : int
            The size of the output array (typically corresponds to the number of observations 
            or time periods).

        Returns:
        --------
        numpy.ndarray
            A 2D numpy array (size x 1) of weights, where each weight corresponds to the fractional 
            differentiation weight for that time period.
        """
        w = [1.]
        for k in range(1, size):
            w_ = -w[-1]/k*(d-k+1)
            w.append(w_)
        w = np.array(w[::-1]).reshape(-1, 1)
        return w

    def frac_diff(self, features_df: pd.DataFrame = None, d: float = 0.7, thres=.01):
        """
        Apply fractional differentiation to time series data, preserving memory of long-term dependencies.

        Parameters:
        -----------
        d : float
            The order of fractional differentiation.
        thres : float, optional (default = 0.01)
            The threshold for determining how many weights to skip based on cumulative sum of the weights.
            Helps to avoid applying very small weights.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame with fractionally differentiated series for each column in `features_df`.
            The output retains long-term memory by applying the computed weights to each time series.
        """
        if features_df is None:
            features_df = self.features_df

        w = self.get_weights(d, features_df.shape[0])
        w_ = np.cumsum(abs(w))
        w_ /= w_[-1]
        skip = w_[w_ > thres].shape[0]
        df = {}
        for name in features_df.columns:
            seriesF, df_ = features_df[[name]].ffill().dropna(
            ), pd.Series(index=features_df.index, dtype=float)
            for iloc in range(skip, seriesF.shape[0]):
                loc = seriesF.index[iloc]
                if not np.isfinite(features_df.loc[loc, name]):
                    continue
                df_[loc] = np.dot(w[-(iloc+1):, :].T, seriesF.loc[:loc])[0, 0]
            df[name] = df_.copy(deep=True)
        df = pd.concat(df, axis=1)

        features_df = df

        return features_df

    @staticmethod
    def getWeights_FFD(d, thres):
        """
        Compute the weights for Fractional Differencing using the Fixed-Width Window method.

        This function calculates the weights required for fractional differencing based on the 
        parameter `d`. The weights decrease in magnitude as the function iterates and stops 
        when the absolute value of the next weight falls below a specified threshold (`thres`).

        Args:
        - d (float): The fractional differencing parameter. Determines the strength of differencing.
        - thres (float): Threshold for the smallest weight to include. Iteration stops when the 
        next weight is smaller than this threshold.

        Returns:
        - np.ndarray: Array of fractional differencing weights, arranged in reverse order.
        """
        w, k = [1.], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def fracDiff_FFD(self, features_df: pd.DataFrame = None, d: float = 0.7, thres=1e-5):
        """
        Apply Fractional Differencing with Fixed-Width Window on a time series to make it stationary.

        Fractional differencing aims to remove long-term dependence from time series data while 
        preserving memory in the data. This method uses a fixed-width window and calculates the 
        fractional differences using the precomputed weights.

        Args:
        - series (pd.DataFrame): Time series data (columns represent individual time series) to be fractionally differenced.
        - d (float): The fractional differencing parameter. Determines the strength of differencing.
        - thres (float, optional): Threshold for the smallest weight to include. Default is 1e-5.

        Returns:
        - pd.DataFrame: A DataFrame with the same structure as `series`, containing the fractionally differenced series.
        """
        # Get fractional differencing weights using the provided threshold and d value
        w, df = self.getWeights_FFD(d, thres), {}
        width = len(w) - 1

        if features_df is None:
            features_df = self.features_df

        # Apply fractional differencing to each column in the DataFrame
        for name in features_df.columns:
            # Forward fill to handle missing data and create a new series for results
            seriesF = features_df[[name]].ffill().dropna()
            df_ = pd.Series(index=features_df.index, dtype=float)

            # Loop through the series and apply the fractional differencing weights
            for iloc in range(width, seriesF.shape[0]):
                loc0, loc1 = seriesF.index[iloc - width], seriesF.index[iloc]
                if not np.isfinite(features_df.loc[loc1, name]):
                    continue
                df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]

            # Store the fractionally differenced column in the result dictionary
            df[name] = df_.copy(deep=True)

        # Combine all columns into a DataFrame
        df = pd.concat(df, axis=1)

        features_df = df

        return features_df

    @staticmethod
    def calc_vwap_on_snippet(df):

        rolling_pv = ((df['Open']+df['High']+df['Low'] +
                      df['Close'])/4 * df['Volume']).cumsum()
        rolling_v = df['Volume'].cumsum()
        vwap = rolling_pv / rolling_v

        return vwap

    def anchored_vwap(self, close_volume_df: pd.DataFrame, anchor_method: Optional[str] = "rolling_window", anchor_lookback: Optional[int] = 10, anchor_price_thresh: Optional[int] = None):
        """Accepts a price series and returns a data frame with columns for price, last anchor, current_vwap, indicator value, and signal value.

        Args:
            df (pd.DataFrame): A dataframe with columns 'Close' and 'Volume'
            anchor_method (Optional[str], optional): Anchors can be found using a rolling window or a directional change algorithm. Defaults to "rolling_window". Other valid values: ["directional_change"].
            anchor_lookback (Optional[int], optional): _description_. Defaults to 10.
            anchor_price_thresh (Optional[int], optional): _description_. Defaults to None.
            indicator_value_method (Optional[str], optional): _description_. Defaults to "price_proximity".
            price_proximity_threshold (Optional[float], optional): _description_. Defaults to 0.01.
        """

        df = close_volume_df.copy(deep=True)

        if anchor_method == "rolling_window":
            bottoms = scipy.signal.argrelextrema(
                df['Close'].to_numpy(), np.less, order=anchor_lookback)
            df['bottom_anchor'] = 0
            df.loc[df.index[bottoms], 'bottom_anchor'] = 1

            tops = scipy.signal.argrelextrema(
                df['Close'].to_numpy(), np.greater, order=anchor_lookback)
            df['top_anchor'] = 0
            df.loc[df.index[tops], 'top_anchor'] = 1

        elif anchor_method == "directional_change":
            raise NotImplementedError("Method not implemented!")

        else:
            raise ValueError("Invalid anchor method!")

        # Extend the index array with the length of the DataFrame to cover the last slice
        indexes = list(bottoms[0]) + [len(df)]

        start_end = [[start+anchor_lookback+1, end+anchor_lookback+1]
                     for start, end in zip(indexes[:-1], indexes[1:])]
        start_end = [[start, end] if end <= len(
            df) else [start, len(df)] for start, end in start_end]

        # Use list comprehension to slice the DataFrame and apply the VWAP function to each slice
        vwap_slices = [df.iloc[start:end] for start, end in start_end]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            vwap_s = list(executor.map(self.calc_vwap_on_snippet, vwap_slices))

        df['bottom_avwap'] = pd.concat(vwap_s).ffill()

        indexes = list(tops[0]) + [len(df)]

        start_end = [[start+anchor_lookback+1, end+anchor_lookback+1]
                     for start, end in zip(indexes[:-1], indexes[1:])]
        start_end = [[start, end] if end <= len(
            df) else [start, len(df)] for start, end in start_end]

        # Use list comprehension to slice the DataFrame and apply the VWAP function to each slice
        vwap_slices = [df.iloc[start:end] for start, end in start_end]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            vwap_s = list(executor.map(self.calc_vwap_on_snippet, vwap_slices))

        df['top_avwap'] = pd.concat(vwap_s).ffill()

        return df

    @staticmethod
    def simple_root(series, power=2):
        """
        Apply a simple root transformation to the series.
        For negative values, take the root of the absolute value and reapply the sign.

        Parameters:
        - series: pd.Series
        - power: float, the root to apply (e.g., 2 for square root)

        Returns:
        - pd.Series with root applied
        """
        signs = np.sign(series)
        roots = np.abs(series) ** (1 / power)
        return signs * roots

    @staticmethod
    def log_transform(series):
        """
        Apply a logarithmic transformation to the series.
        Handles non-positive values by shifting the data if necessary.

        Parameters:
        - series: pd.Series

        Returns:
        - pd.Series with log applied
        """
        min_val = series.min()
        if min_val <= 0:
            shifted = series + abs(min_val) + 1
            print(f"\nSeries shifted by {
                  abs(min_val) + 1} to make all values positive for log transformation.")
        else:
            shifted = series
        return np.log(shifted)

    @staticmethod
    def sigmoid_transform(series: np.ndarray, transform_type: str = 'tanh', center: bool = True) -> np.ndarray:
        """
        Apply a sigmoid transformation using the hyperbolic tangent function.

        Parameters:
        - series: np.ndarray

        Returns:
        - np.ndarray with tanh applied
        """
        transformed = series
        if transform_type == 'tanh':
            # Center the data around 0
            if center:
                series -= pd.Series(series).expanding().median()
            # Extract the 15th and 85th percentiles
            p15 = pd.Series(series).expanding().apply(
                lambda x: np.percentile(x, 15))
            p85 = pd.Series(series).expanding().apply(
                lambda x: np.percentile(x, 85))
            # Scale the values to be between -1.5 and 1.5 using the 15 and 85 percentiles
            series /= (p85 - p15) * 1.5
            series = series.values
            # Apply the tanh function
            transformed = np.tanh(series)

        transformed = np.where(np.isnan(transformed), 0, transformed)
        return transformed

    def transform(self, series: pd.Series, method: str, power: Optional[float] = 2, center: Optional[bool] = False) -> pd.Series:
        """
        Apply a specified transformation to the series.

        Parameters:
        - series: pd.Series
        - method: str, the transformation method to apply ('simple_root', 'log', 'sigmoid')

        Returns:
        - pd.Series with the specified transformation applied
        """
        if method == 'root':
            return self.simple_root(series, power)
        elif method == 'log':
            return self.log_transform(series)
        elif method == 'tanh':
            return self.sigmoid_transform(series, 'tanh', center)
        else:
            raise ValueError(f"Unknown transformation method: {method}")

    @staticmethod
    def exp_smoothing(series: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Apply exponential smoothing to the series.

        Parameters:
        - series: np.ndarray
        - degree: int, the degree of exponential smoothing to apply

        Returns:
        - np.ndarray with exponential smoothing applied
        """

        alpha = 2 / (degree + 1)
        smoothed = np.full(len(series), np.nan)
        smoothed[0] = series[0]

        for i in range(1, len(series)):
            smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i - 1]

        return smoothed


class BarPermute:
    def __init__(self, ohlc_df_list: list):

        self.n_markets = len(ohlc_df_list)
        if self.n_markets < 1:
            raise ValueError("Need at least one market")

        # Ensure that all dataframes have the same length
        if not all(len(df) == len(ohlc_df_list[0]) for df in ohlc_df_list):
            raise ValueError("All dataframes must have the same length")

        # Store the index to reassign later
        self.original_index = ohlc_df_list[0].index

        self.ohlc_df_list = []

        # Drop the index for simplicity of permutation
        for df in ohlc_df_list:
            self.ohlc_df_list.append(df.reset_index(drop=True))

        self.basis_prices = {i: {'Open': self.ohlc_df_list[i]['Open'].iloc[0],
                                 'High': self.ohlc_df_list[i]['High'].iloc[0],
                                 'Low': self.ohlc_df_list[i]['Low'].iloc[0],
                                 'Close': self.ohlc_df_list[i]['Close'].iloc[0]}
                             for i in range(self.n_markets)}

        self.rel_prices = []
        for df in self.ohlc_df_list:
            rel_df = pd.DataFrame({
                'rel_open': df['Open'] - df['Close'].shift(1),
                'rel_high': df['High'] - df['Open'],
                'rel_low': df['Low'] - df['Open'],
                'rel_close': df['Close'] - df['Open']
            })
            self.rel_prices.append(rel_df.dropna())

    def permute(self):

        index_array = np.array(self.rel_prices[0].index)

        # Shuffle each relative high/low/close series using the same permutation
        shuffled_indices_hlc = np.random.choice(
            index_array, len(index_array), replace=True)
        shuffled_indices_open = np.random.choice(
            index_array, len(index_array), replace=True)

        shuffled_rel_prices = [
            pd.DataFrame({
                # Need to subtract 1 as .to_numpy() resets the index
                'rel_open': rel_prices['rel_open'].to_numpy()[shuffled_indices_open-1],
                'rel_high': rel_prices['rel_high'].to_numpy()[shuffled_indices_hlc-1],
                'rel_low': rel_prices['rel_low'].to_numpy()[shuffled_indices_hlc-1],
                'rel_close': rel_prices['rel_close'].to_numpy()[shuffled_indices_hlc-1]
            })
            for rel_prices in self.rel_prices
        ]

        shuffled_df_list = []

        # Reconstruct the shuffled OHLC dataframes using the basis prices
        for i in range(self.n_markets):
            temp_open = []
            temp_high = []
            temp_low = []
            temp_close = []

            temp_open.append(self.basis_prices[i]['Open'])
            temp_high.append(self.basis_prices[i]['High'])
            temp_low.append(self.basis_prices[i]['Low'])
            temp_close.append(self.basis_prices[i]['Close'])

            rel_price_df = shuffled_rel_prices[i]

            for j in range(len(rel_price_df.index)):

                temp_open.append(
                    temp_close[j] + rel_price_df['rel_open'].iloc[j])
                temp_high.append(temp_open[j+1] +
                                 rel_price_df['rel_high'].iloc[j])
                temp_low.append(temp_open[j+1] +
                                rel_price_df['rel_low'].iloc[j])
                temp_close.append(
                    temp_open[j+1] + rel_price_df['rel_close'].iloc[j])

            ohlc_df = pd.DataFrame({
                'Open': temp_open,
                'High': temp_high,
                'Low': temp_low,
                'Close': temp_close
            })

            if len(self.original_index) == len(ohlc_df):
                ohlc_df.set_index(self.original_index, inplace=True)
            else:
                raise ValueError(
                    "Length of self.original_index does not match DataFrame length")

            shuffled_df_list.append(ohlc_df)

        return shuffled_df_list


class PricePermute:
    def __init__(self, price_series_list: list):
        self.n_markets = len(price_series_list)
        if self.n_markets < 1:
            raise ValueError("Need at least one market")

        # Ensure that all series have the same length
        if not all(len(series) == len(price_series_list[0]) for series in price_series_list):
            raise ValueError("All series must have the same length")

        # Store the index to reassign later
        self.original_index = price_series_list[0].index

        self.price_series_list = [series.reset_index(
            drop=True) for series in price_series_list]

        self.basis_prices = {
            i: self.price_series_list[i].iloc[0] for i in range(self.n_markets)}

        self.rel_prices = [series.diff().dropna()
                           for series in self.price_series_list]

    def permute(self):
        index_array = np.array(self.rel_prices[0].index)

        # Shuffle each relative price series using the same permutation
        shuffled_indices = np.random.choice(
            index_array, len(index_array), replace=True)

        shuffled_rel_prices = [rel_prices.to_numpy()[shuffled_indices-1]
                               for rel_prices in self.rel_prices]

        shuffled_series_list = []

        # Reconstruct the shuffled price series using the basis prices
        for i in range(self.n_markets):
            temp_prices = [self.basis_prices[i]]

            for j in range(len(shuffled_rel_prices[i])):
                temp_prices.append(temp_prices[j] + shuffled_rel_prices[i][j])

            price_series = pd.Series(temp_prices)

            if len(self.original_index) == len(price_series):
                price_series.index = self.original_index
            else:
                raise ValueError(
                    "Length of self.original_index does not match Series length")

            shuffled_series_list.append(price_series)

        return shuffled_series_list
