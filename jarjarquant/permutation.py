"""Bar and price permutation for statistical testing."""

import numpy as np
import polars as pl

from jarjarquant._cython.bar_permute import permute_cython, permute_cython_single


class BarPermute:
    """Permute OHLC bars while preserving intra-bar relationships.

    Args:
        ohlc_df: Polars DataFrame with Open, High, Low, Close columns.
    """

    def __init__(self, ohlc_df: pl.DataFrame):
        if ohlc_df is None or ohlc_df.is_empty():
            raise ValueError("ohlc_df must be a non-empty DataFrame")

        self._n_rows = len(ohlc_df)

        # Extract numpy arrays
        open_arr = ohlc_df["Open"].to_numpy().astype(np.float64)
        high_arr = ohlc_df["High"].to_numpy().astype(np.float64)
        low_arr = ohlc_df["Low"].to_numpy().astype(np.float64)
        close_arr = ohlc_df["Close"].to_numpy().astype(np.float64)

        # Basis prices (first row)
        self.basis_prices = np.array(
            [[open_arr[0], high_arr[0], low_arr[0], close_arr[0]]]
        )

        # Relative prices
        rel_open = open_arr[1:] - close_arr[:-1]
        rel_high = high_arr[1:] - open_arr[1:]
        rel_low = low_arr[1:] - open_arr[1:]
        rel_close = close_arr[1:] - open_arr[1:]

        self.relative_prices = np.column_stack(
            [rel_open, rel_high, rel_low, rel_close]
        )

    def permute(self) -> pl.DataFrame:
        """Generate a permuted OHLC DataFrame.

        Returns:
            Polars DataFrame with permuted Open, High, Low, Close columns.
        """
        n = len(self.relative_prices)
        indices = np.arange(n)

        # Shuffle indices for open and hlc separately
        shuffled_hlc = np.random.choice(indices, n, replace=True)
        shuffled_open = np.random.choice(indices, n, replace=True)

        shuffled_rel = np.array(
            [[
                self.relative_prices[shuffled_open, 0],
                self.relative_prices[shuffled_hlc, 1],
                self.relative_prices[shuffled_hlc, 2],
                self.relative_prices[shuffled_hlc, 3],
            ]]
        ).transpose(0, 2, 1)

        permuted = permute_cython(self.basis_prices, shuffled_rel)

        return pl.DataFrame({
            "Open": permuted[0, :, 0],
            "High": permuted[0, :, 1],
            "Low": permuted[0, :, 2],
            "Close": permuted[0, :, 3],
        })


class PricePermute:
    """Permute a price series while preserving return distribution.

    Args:
        prices: Polars Series of prices.
    """

    def __init__(self, prices: pl.Series):
        if prices is None or len(prices) == 0:
            raise ValueError("prices must be a non-empty Series")

        self._name = prices.name
        self._n = len(prices)
        values = prices.to_numpy().astype(np.float64)

        self.basis_prices = np.array(
            [[values[0], values[0], values[0], values[0]]]
        )

        # Relative prices = first differences
        self._rel_prices = np.diff(values)

    def permute(self) -> pl.Series:
        """Generate a permuted price series.

        Returns:
            Polars Series with permuted prices.
        """
        n = len(self._rel_prices)
        indices = np.arange(n)
        shuffled = np.random.choice(indices, n, replace=True)

        shuffled_rel = np.array(
            [self._rel_prices[shuffled].reshape(-1, 1)]
        )

        permuted = permute_cython_single(self.basis_prices, shuffled_rel)

        return pl.Series(self._name, permuted[0].flatten())
