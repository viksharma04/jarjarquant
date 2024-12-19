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
