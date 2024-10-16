"""Feature engineering class to manipulate features df"""
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Class to implement common featrue engineering transformations"""

    def __init__(self, features_df):
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

    def frac_diff(self, d, thres=.01):
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
        w = self.get_weights(d, self.features_df.shape[0])
        w_ = np.cumsum(abs(w))
        w_ /= w_[-1]
        skip = w_[w_ > thres].shape[0]
        df = {}
        for name in self.features_df.columns:
            seriesF, df_ = self.features_df[[name]].ffill().dropna(
            ), pd.Series(index=self.features_df.index, dtype=float)
            for iloc in range(skip, seriesF.shape[0]):
                loc = seriesF.index[iloc]
                if not np.isfinite(self.features_df.loc[loc, name]):
                    continue
                df_[loc] = np.dot(w[-(iloc+1):, :].T, seriesF.loc[:loc])[0, 0]
            df[name] = df_.copy(deep=True)
        df = pd.concat(df, axis=1)

        self.features_df = df

        return self.features_df

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

    def fracDiff_FFD(self, d, thres=1e-5):
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

        # Apply fractional differencing to each column in the DataFrame
        for name in self.features_df.columns:
            # Forward fill to handle missing data and create a new series for results
            seriesF = self.features_df[[name]].ffill().dropna()
            df_ = pd.Series(index=self.features_df.index, dtype=float)

            # Loop through the series and apply the fractional differencing weights
            for iloc in range(width, seriesF.shape[0]):
                loc0, loc1 = seriesF.index[iloc - width], seriesF.index[iloc]
                if not np.isfinite(self.features_df.loc[loc1, name]):
                    continue
                df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]

            # Store the fractionally differenced column in the result dictionary
            df[name] = df_.copy(deep=True)

        # Combine all columns into a DataFrame
        df = pd.concat(df, axis=1)

        self.features_df = df

        return self.features_df
