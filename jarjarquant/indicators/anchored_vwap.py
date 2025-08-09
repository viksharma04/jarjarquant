import numpy as np
import pandas as pd

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType
# DataAnalyst is no longer a class - functions are standalone


@register_indicator(IndicatorType.ANCHORED_VWAP)
class AnchoredVWAP(Indicator):
    """
    Anchored VWAP indicator that calculates VWAP from pivot points detected using directional change algorithm.

    The indicator works in three steps:
    1. Find pivot points using directional change algorithm
    2. Calculate VWAP from each anchor point forward (avoiding look-ahead bias)
    3. Normalize the distance between price and VWAP using ATR
    """

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        threshold_value: float = 0.02,
        atr_period: int = 14,
        price_formula: str = "ohlc4",
        transform=None,
    ):
        """
        Initialize Anchored VWAP indicator.

        Args:
            ohlcv_df: DataFrame with OHLCV data
            threshold_value: Threshold for directional change pivot detection (default 2%)
            atr_period: Period for ATR calculation for normalization
            price_formula: "ohlc4" for (O+H+L+C)/4 or "hlc3" for (H+L+C)/3
            transform: Optional transformation to apply to output
        """
        super().__init__(ohlcv_df)
        self.threshold_value = threshold_value
        self.atr_period = atr_period
        self.price_formula = price_formula
        self.indicator_type = "continuous"
        self.transform = transform

    def calculate(self) -> np.ndarray:
        """Calculate the anchored VWAP indicator values."""
        n = len(self.df)

        # Get price series for pivot detection (using close)
        close_prices = self.df["Close"].values

        # Step 1: Find pivot points using directional change algorithm
        pivots = DataAnalyst.directional_change_pivots(
            series=close_prices, threshold_value=self.threshold_value
        )

        # Step 2: Calculate typical price based on formula
        if self.price_formula == "ohlc4":
            typical_price = (
                self.df["Open"] + self.df["High"] + self.df["Low"] + self.df["Close"]
            ) / 4
        elif self.price_formula == "hlc3":
            typical_price = (self.df["High"] + self.df["Low"] + self.df["Close"]) / 3
        else:
            raise ValueError("price_formula must be 'ohlc4' or 'hlc3'")

        volume = self.df["Volume"].values
        typical_price_values = typical_price.values

        # Step 3: Calculate ATR for normalization
        atr_values = DataAnalyst.atr(
            atr_length=self.atr_period,
            high_series=self.df["High"],
            low_series=self.df["Low"],
            close_series=self.df["Close"],
        ).values

        # Initialize output array
        output = np.full(n, np.nan)

        # Step 4: Calculate anchored VWAP and indicator values
        for i, pivot_idx in enumerate(pivots):
            # For each pivot, calculate VWAP from that point forward
            # Start from the bar AFTER the pivot to avoid look-ahead bias
            start_idx = pivot_idx + 1

            if start_idx >= n:
                continue

            # Calculate running VWAP from this anchor point
            cumulative_pv = 0.0
            cumulative_volume = 0.0

            for j in range(start_idx, n):
                # Update cumulative values
                cumulative_pv += typical_price_values[j] * volume[j]
                cumulative_volume += volume[j]

                if cumulative_volume > 0:
                    vwap = cumulative_pv / cumulative_volume

                    # Calculate normalized distance
                    price_diff = abs(close_prices[j] - vwap)

                    # Normalize by ATR (avoid division by zero)
                    if atr_values[j] > 0:
                        normalized_distance = price_diff / atr_values[j]
                    else:
                        normalized_distance = 0.0

                    # Only update if this is the most recent pivot affecting this bar
                    # (later pivots override earlier ones)
                    output[j] = normalized_distance

        # Fill any remaining NaN values with 0
        output = np.nan_to_num(output, nan=0.0)

        # Apply transformation if specified
        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
