"""
Synthetic OHLCV data generation with various statistical distributions.

This module provides a comprehensive synthetic data source for generating realistic
OHLCV (Open, High, Low, Close, Volume) data using various statistical distributions
and market microstructure assumptions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from .base import DataSource, register_data_source


class DistributionType(Enum):
    """Statistical distributions available for price generation."""

    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    STUDENT_T = "student_t"
    LAPLACE = "laplace"
    GBM = "geometric_brownian_motion"  # Geometric Brownian Motion
    JUMP_DIFFUSION = "jump_diffusion"  # Merton Jump Diffusion
    GARCH = "garch"  # GARCH(1,1) volatility clustering
    REGIME_SWITCHING = "regime_switching"  # Two-regime switching


@dataclass
class IntradayPriceParams:
    """Parameters for intraday price (OHLC) generation from close prices."""

    # High/Low spreads as percentage of close price
    high_spread_mean: float = 0.015  # 1.5% average high above close
    high_spread_std: float = 0.005  # Standard deviation
    low_spread_mean: float = 0.015  # 1.5% average low below close
    low_spread_std: float = 0.005  # Standard deviation

    # Open price relative to previous close
    gap_mean: float = 0.0  # Average overnight gap
    gap_std: float = 0.008  # Gap volatility (0.8%)

    # Distribution for spreads (normal, gamma, etc.)
    spread_distribution: str = "gamma"  # More realistic than normal


@dataclass
class VolumeParams:
    """Parameters for volume generation."""

    base_volume: int = 1000000  # Base daily volume
    volume_std: float = 0.3  # Volume volatility (30%)
    volume_trend: float = 0.0  # Secular volume trend
    volume_weekday_effect: bool = True  # Lower volume on Mon/Fri
    volume_distribution: str = "log_normal"  # Volume distribution type


@dataclass
class MarketParams:
    """Overall market parameters."""

    trading_days_per_year: int = 252
    start_price: float = 100.0
    start_date: str = "2023-01-01"

    # Market hours (for intraday data)
    market_open_hour: int = 9
    market_close_hour: int = 16
    timezone: str = "US/Eastern"


@register_data_source("synthetic")
class SyntheticDataSource(DataSource):
    """
    Advanced synthetic OHLCV data generator with realistic market microstructure.

    This data source generates synthetic financial time series data using various
    statistical distributions and realistic assumptions about intraday price behavior,
    volume patterns, and market microstructure.

    Example:
        # Create normally distributed returns
        source = SyntheticDataSource()
        data = await source.fetch(
            distribution_type=DistributionType.NORMAL,
            periods=252,
            return_mean=0.001,
            return_std=0.02
        )

        # Create Jump Diffusion model
        data = await source.fetch(
            distribution_type=DistributionType.JUMP_DIFFUSION,
            periods=1000,
            drift=0.0002,
            volatility=0.015,
            jump_intensity=0.1,
            jump_size_mean=0.02,
            jump_size_std=0.01
        )
    """

    def __init__(self):
        """Initialize the synthetic data source."""
        self.rng = np.random.default_rng()  # Use modern random number generator

    async def fetch(
        self,
        ticker: str = "",  # Ignored for synthetic data, but needed for DataGatherer compatibility
        distribution_type: Union[DistributionType, str] = DistributionType.NORMAL,
        periods: int = 252,
        freq: str = "D",  # D=daily, H=hourly, T=minute
        # Distribution-specific parameters
        return_mean: float = 0.001,  # Daily return mean
        return_std: float = 0.02,  # Daily return std
        drift: Optional[float] = None,  # For GBM (defaults to return_mean)
        volatility: Optional[float] = None,  # For GBM (defaults to return_std)
        # Student-t parameters
        df: float = 3.0,  # Degrees of freedom for t-distribution
        # Jump diffusion parameters
        jump_intensity: float = 0.05,  # Jumps per period
        jump_size_mean: float = 0.0,  # Mean jump size
        jump_size_std: float = 0.03,  # Jump size volatility
        # GARCH parameters
        garch_omega: float = 0.0001,  # GARCH constant
        garch_alpha: float = 0.1,  # GARCH alpha (volatility reaction)
        garch_beta: float = 0.85,  # GARCH beta (volatility persistence)
        # Regime switching parameters
        regime_prob: float = 0.02,  # Probability of regime switch per period
        regime_1_mean: float = 0.001,  # Bull market mean return
        regime_1_std: float = 0.015,  # Bull market volatility
        regime_2_mean: float = -0.002,  # Bear market mean return
        regime_2_std: float = 0.035,  # Bear market volatility
        # Intraday and volume parameters
        intraday_params: Optional[IntradayPriceParams] = None,
        volume_params: Optional[VolumeParams] = None,
        market_params: Optional[MarketParams] = None,
        # Other parameters
        seed: Optional[int] = None,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Generate synthetic OHLCV data using specified distribution.

        Args:
            distribution_type: Statistical distribution for returns generation
            periods: Number of periods to generate
            freq: Frequency ('D' for daily, 'H' for hourly, 'T' for minute)
            return_mean: Expected return per period
            return_std: Return volatility per period
            drift: Drift parameter for GBM (defaults to return_mean)
            volatility: Volatility parameter for GBM (defaults to return_std)
            df: Degrees of freedom for Student-t distribution
            jump_intensity: Average number of jumps per period
            jump_size_mean: Mean size of jumps
            jump_size_std: Standard deviation of jump sizes
            garch_omega: GARCH omega parameter
            garch_alpha: GARCH alpha parameter
            garch_beta: GARCH beta parameter
            regime_prob: Probability of regime switch
            regime_1_mean: Mean return in regime 1
            regime_1_std: Volatility in regime 1
            regime_2_mean: Mean return in regime 2
            regime_2_std: Volatility in regime 2
            intraday_params: Parameters for OHLC generation
            volume_params: Parameters for volume generation
            market_params: General market parameters
            seed: Random seed for reproducibility

        Returns:
            DataFrame with OHLCV data
        """

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Convert string to enum if needed
        if isinstance(distribution_type, str):
            distribution_type = DistributionType(distribution_type)

        # Set defaults
        if intraday_params is None:
            intraday_params = IntradayPriceParams()
        if volume_params is None:
            volume_params = VolumeParams()
        if market_params is None:
            market_params = MarketParams()

        # Generate date index
        start_date = pd.to_datetime(market_params.start_date)
        if freq == "D":
            date_index = pd.bdate_range(start=start_date, periods=periods, freq="B")
        elif freq == "H":
            date_index = pd.date_range(start=start_date, periods=periods, freq="H")
        elif freq == "T":
            # Business hours only for minute data
            date_index = pd.date_range(start=start_date, periods=periods, freq="T")
            # Filter to market hours (simplified)
            date_index = date_index[
                (date_index.hour >= market_params.market_open_hour)
                & (date_index.hour < market_params.market_close_hour)
            ][:periods]
        else:
            raise ValueError(f"Unsupported frequency: {freq}")

        # Generate returns based on distribution type
        returns = self._generate_returns(
            distribution_type=distribution_type,
            periods=periods,
            return_mean=return_mean,
            return_std=return_std,
            drift=drift,
            volatility=volatility,
            df=df,
            jump_intensity=jump_intensity,
            jump_size_mean=jump_size_mean,
            jump_size_std=jump_size_std,
            garch_omega=garch_omega,
            garch_alpha=garch_alpha,
            garch_beta=garch_beta,
            regime_prob=regime_prob,
            regime_1_mean=regime_1_mean,
            regime_1_std=regime_1_std,
            regime_2_mean=regime_2_mean,
            regime_2_std=regime_2_std,
        )

        # Convert returns to prices
        prices = market_params.start_price * np.exp(np.cumsum(returns))

        # Generate OHLC data from close prices
        ohlc_data = self._generate_ohlc(prices, intraday_params)

        # Generate volume data
        volumes = self._generate_volume(periods, date_index, volume_params)

        # Create DataFrame
        df_data = {
            "date": date_index[: len(prices)],  # Ensure same length
            "Open": ohlc_data["open"],
            "High": ohlc_data["high"],
            "Low": ohlc_data["low"],
            "Close": prices,
            "Volume": volumes[: len(prices)],
        }

        df = pl.DataFrame(df_data)

        # Ensure proper data types
        df = df.with_columns(
            [
                pl.col("Open").cast(pl.Float64),
                pl.col("High").cast(pl.Float64),
                pl.col("Low").cast(pl.Float64),
                pl.col("Close").cast(pl.Float64),
                pl.col("Volume").cast(pl.Int64),
            ]
        )

        return df

    def _generate_returns(
        self, distribution_type: DistributionType, periods: int, **params
    ) -> np.ndarray:
        """Generate returns based on the specified distribution."""

        if distribution_type == DistributionType.NORMAL:
            return self.rng.normal(params["return_mean"], params["return_std"], periods)

        elif distribution_type == DistributionType.LOG_NORMAL:
            # For log-normal, we generate the underlying normal and exponentiate
            mu = np.log(1 + params["return_mean"]) - 0.5 * params["return_std"] ** 2
            sigma = params["return_std"]
            return np.exp(self.rng.normal(mu, sigma, periods)) - 1

        elif distribution_type == DistributionType.STUDENT_T:
            # Student-t distribution (fat tails)
            raw_t = self.rng.standard_t(params["df"], periods)
            # Scale and shift to match desired mean and std
            return params["return_mean"] + params["return_std"] * raw_t / np.sqrt(
                params["df"] / (params["df"] - 2)
            )

        elif distribution_type == DistributionType.LAPLACE:
            # Laplace distribution (symmetric, fat tails)
            scale = params["return_std"] / np.sqrt(2)
            return self.rng.laplace(params["return_mean"], scale, periods)

        elif distribution_type == DistributionType.GBM:
            # Geometric Brownian Motion
            drift = params.get("drift") or params.get("return_mean", 0.001)
            volatility = params.get("volatility") or params.get("return_std", 0.02)
            dt = 1.0  # Time step
            return self.rng.normal(
                drift - 0.5 * volatility**2, volatility * np.sqrt(dt), periods
            )

        elif distribution_type == DistributionType.JUMP_DIFFUSION:
            # Merton Jump Diffusion Model
            returns = self._generate_jump_diffusion(
                periods,
                params["return_mean"],
                params["return_std"],
                params["jump_intensity"],
                params["jump_size_mean"],
                params["jump_size_std"],
            )
            return returns

        elif distribution_type == DistributionType.GARCH:
            # GARCH(1,1) model
            returns = self._generate_garch(
                periods,
                params["return_mean"],
                params["garch_omega"],
                params["garch_alpha"],
                params["garch_beta"],
            )
            return returns

        elif distribution_type == DistributionType.REGIME_SWITCHING:
            # Two-regime switching model
            returns = self._generate_regime_switching(
                periods,
                params["regime_prob"],
                params["regime_1_mean"],
                params["regime_1_std"],
                params["regime_2_mean"],
                params["regime_2_std"],
            )
            return returns

        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")

    def _generate_jump_diffusion(
        self,
        periods: int,
        drift: float,
        volatility: float,
        jump_intensity: float,
        jump_size_mean: float,
        jump_size_std: float,
    ) -> np.ndarray:
        """Generate returns using Merton Jump Diffusion model."""

        # Continuous part (Brownian motion)
        continuous_returns = self.rng.normal(
            drift - 0.5 * volatility**2, volatility, periods
        )

        # Jump part
        # Number of jumps follows Poisson distribution
        n_jumps = self.rng.poisson(jump_intensity, periods)

        jump_returns = np.zeros(periods)
        for i in range(periods):
            if n_jumps[i] > 0:
                # Jump sizes are normally distributed
                jumps = self.rng.normal(jump_size_mean, jump_size_std, n_jumps[i])
                jump_returns[i] = np.sum(jumps)

        return continuous_returns + jump_returns

    def _generate_garch(
        self, periods: int, mean_return: float, omega: float, alpha: float, beta: float
    ) -> np.ndarray:
        """Generate returns using GARCH(1,1) model."""

        returns = np.zeros(periods)
        variances = np.zeros(periods)

        # Initialize variance
        variances[0] = omega / (1 - alpha - beta)  # Unconditional variance

        for t in range(periods):
            if t > 0:
                # GARCH variance equation
                variances[t] = (
                    omega + alpha * returns[t - 1] ** 2 + beta * variances[t - 1]
                )

            # Generate return with time-varying volatility
            returns[t] = (
                mean_return + np.sqrt(variances[t]) * self.rng.standard_normal()
            )

        return returns

    def _generate_regime_switching(
        self,
        periods: int,
        switch_prob: float,
        regime_1_mean: float,
        regime_1_std: float,
        regime_2_mean: float,
        regime_2_std: float,
    ) -> np.ndarray:
        """Generate returns using a two-regime switching model."""

        returns = np.zeros(periods)
        regime = 0  # Start in regime 0 (bull market)

        for t in range(periods):
            # Check for regime switch
            if self.rng.random() < switch_prob:
                regime = 1 - regime  # Switch regime

            # Generate return based on current regime
            if regime == 0:
                returns[t] = self.rng.normal(regime_1_mean, regime_1_std)
            else:
                returns[t] = self.rng.normal(regime_2_mean, regime_2_std)

        return returns

    def _generate_ohlc(
        self, close_prices: np.ndarray, params: IntradayPriceParams
    ) -> Dict[str, np.ndarray]:
        """Generate realistic OHLC data from close prices."""

        n_periods = len(close_prices)

        # Generate overnight gaps for open prices
        gaps = self.rng.normal(params.gap_mean, params.gap_std, n_periods)
        open_prices = np.zeros(n_periods)
        open_prices[0] = close_prices[0]  # First open = first close

        for i in range(1, n_periods):
            open_prices[i] = close_prices[i - 1] * (1 + gaps[i])

        # Generate high and low spreads
        if params.spread_distribution == "gamma":
            # Gamma distribution for more realistic spreads (always positive)
            high_spreads = self.rng.gamma(
                shape=(params.high_spread_mean / params.high_spread_std) ** 2,
                scale=params.high_spread_std**2 / params.high_spread_mean,
                size=n_periods,
            )
            low_spreads = self.rng.gamma(
                shape=(params.low_spread_mean / params.low_spread_std) ** 2,
                scale=params.low_spread_std**2 / params.low_spread_mean,
                size=n_periods,
            )
        else:
            # Normal distribution (can be negative, will be clipped)
            high_spreads = np.clip(
                self.rng.normal(
                    params.high_spread_mean, params.high_spread_std, n_periods
                ),
                0.001,
                None,  # Minimum 0.1% spread
            )
            low_spreads = np.clip(
                self.rng.normal(
                    params.low_spread_mean, params.low_spread_std, n_periods
                ),
                0.001,
                None,  # Minimum 0.1% spread
            )

        # Calculate high and low prices
        high_prices = np.maximum(
            close_prices * (1 + high_spreads),
            np.maximum(open_prices, close_prices),  # High must be >= max(open, close)
        )

        low_prices = np.minimum(
            close_prices * (1 - low_spreads),
            np.minimum(open_prices, close_prices),  # Low must be <= min(open, close)
        )

        return {"open": open_prices, "high": high_prices, "low": low_prices}

    def _generate_volume(
        self, periods: int, date_index: pd.DatetimeIndex, params: VolumeParams
    ) -> np.ndarray:
        """Generate realistic volume data."""

        # Base volume with trend
        time_trend = np.arange(periods) * params.volume_trend / periods
        base_volumes = params.base_volume * (1 + time_trend)

        # Volume noise
        if params.volume_distribution == "log_normal":
            # Log-normal ensures positive volumes
            sigma = np.sqrt(np.log(1 + params.volume_std**2))
            mu = np.log(base_volumes) - 0.5 * sigma**2
            volumes = self.rng.lognormal(mu, sigma)
        else:
            # Normal with clipping
            volumes = np.clip(
                self.rng.normal(base_volumes, base_volumes * params.volume_std),
                base_volumes * 0.1,  # Minimum 10% of base
                None,
            )

        # Weekday effects (if daily data)
        if params.volume_weekday_effect and len(date_index) > 0:
            weekdays = date_index.weekday
            # Reduce volume on Monday (0) and Friday (4)
            volume_multiplier = np.where(
                (weekdays == 0) | (weekdays == 4),
                0.8,  # 20% lower volume
                1.0,
            )
            volumes = volumes * volume_multiplier[: len(volumes)]

        return volumes.astype(int)
