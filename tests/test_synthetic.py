"""
Test suite for the synthetic data generation module.

This module tests all functionality in jarjarquant.data_gatherer.synthetic,
including various distribution types, parameter validation, and data quality.
"""

import asyncio
import unittest

import numpy as np
import pandas as pd
import polars as pl

from jarjarquant.data_gatherer.base import get_data_source, list_data_sources
from jarjarquant.data_gatherer.synthetic import (
    DistributionType,
    IntradayPriceParams,
    MarketParams,
    SyntheticDataSource,
    VolumeParams,
)


class TestSyntheticDataSource(unittest.TestCase):
    """Test suite for SyntheticDataSource class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.source = SyntheticDataSource()
        self.default_periods = 100
        self.seed = 42  # For reproducible tests

    def test_initialization(self):
        """Test that SyntheticDataSource initializes correctly."""
        self.assertIsInstance(self.source, SyntheticDataSource)
        self.assertTrue(hasattr(self.source, "rng"))

    def test_data_source_registration(self):
        """Test that synthetic data source is properly registered."""
        self.assertIn("synthetic", list_data_sources())
        registered_class = get_data_source("synthetic")
        self.assertEqual(registered_class, SyntheticDataSource)

    async def test_basic_fetch(self):
        """Test basic fetch functionality with default parameters."""
        data = await self.source.fetch(
            ticker="TEST", periods=self.default_periods, seed=self.seed
        )

        # Check data structure
        self.assertIsInstance(data, pl.DataFrame)
        self.assertEqual(len(data), self.default_periods)

        # Check columns
        expected_columns = ["date", "Open", "High", "Low", "Close", "Volume"]
        self.assertEqual(list(data.columns), expected_columns)

        # Check data types
        self.assertEqual(data["Open"].dtype, pl.Float64)
        self.assertEqual(data["High"].dtype, pl.Float64)
        self.assertEqual(data["Low"].dtype, pl.Float64)
        self.assertEqual(data["Close"].dtype, pl.Float64)
        self.assertEqual(data["Volume"].dtype, pl.Int64)

    async def test_ohlc_consistency(self):
        """Test that OHLC data maintains proper relationships."""
        data = await self.source.fetch(
            ticker="TEST", periods=self.default_periods, seed=self.seed
        )

        # Convert to pandas for easier testing
        df = data.to_pandas()

        # High should be >= max(Open, Close)
        self.assertTrue(all(df["High"] >= df[["Open", "Close"]].max(axis=1)))

        # Low should be <= min(Open, Close)
        self.assertTrue(all(df["Low"] <= df[["Open", "Close"]].min(axis=1)))

        # All prices should be positive
        self.assertTrue(all(df["Open"] > 0))
        self.assertTrue(all(df["High"] > 0))
        self.assertTrue(all(df["Low"] > 0))
        self.assertTrue(all(df["Close"] > 0))

        # Volume should be positive integers
        self.assertTrue(all(df["Volume"] > 0))
        self.assertTrue(all(df["Volume"] == df["Volume"].astype(int)))

    async def test_normal_distribution(self):
        """Test normal distribution returns generation."""
        data = await self.source.fetch(
            ticker="TEST",
            distribution_type=DistributionType.NORMAL,
            periods=1000,  # Larger sample for statistical tests
            return_mean=0.001,
            return_std=0.02,
            seed=self.seed,
        )

        df = data.to_pandas()
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        # Check statistical properties (with some tolerance)
        self.assertAlmostEqual(returns.mean(), 0.001, delta=0.005)
        self.assertAlmostEqual(returns.std(), 0.02, delta=0.005)

    async def test_student_t_distribution(self):
        """Test Student-t distribution with fat tails."""
        data = await self.source.fetch(
            ticker="TEST",
            distribution_type=DistributionType.STUDENT_T,
            periods=1000,
            df=3.0,
            return_mean=0.001,
            return_std=0.02,
            seed=self.seed,
        )

        df_data = data.to_pandas()
        returns = np.log(df_data["Close"] / df_data["Close"].shift(1)).dropna()

        # Student-t should have fatter tails than normal
        # Check that kurtosis is higher than normal (3)
        kurtosis = returns.kurtosis()
        self.assertGreater(kurtosis, 3)

    async def test_log_normal_distribution(self):
        """Test log-normal distribution returns generation."""
        data = await self.source.fetch(
            ticker="TEST",
            distribution_type=DistributionType.LOG_NORMAL,
            periods=500,
            return_mean=0.001,
            return_std=0.02,
            seed=self.seed,
        )

        self.assertEqual(len(data), 500)
        df = data.to_pandas()

        # All prices should be positive (guaranteed by log-normal)
        self.assertTrue(all(df["Close"] > 0))

    async def test_laplace_distribution(self):
        """Test Laplace distribution returns generation."""
        data = await self.source.fetch(
            ticker="TEST",
            distribution_type=DistributionType.LAPLACE,
            periods=500,
            return_mean=0.001,
            return_std=0.02,
            seed=self.seed,
        )

        self.assertEqual(len(data), 500)
        df = data.to_pandas()
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        # Laplace should have fatter tails than normal but not as fat as Student-t
        kurtosis = returns.kurtosis()
        self.assertGreater(kurtosis, 3)

    async def test_gbm_distribution(self):
        """Test Geometric Brownian Motion."""
        drift = 0.0002
        volatility = 0.015

        data = await self.source.fetch(
            ticker="TEST",
            distribution_type=DistributionType.GBM,
            periods=500,
            drift=drift,
            volatility=volatility,
            seed=self.seed,
        )

        self.assertEqual(len(data), 500)
        df = data.to_pandas()

        # All prices should be positive
        self.assertTrue(all(df["Close"] > 0))

    async def test_jump_diffusion(self):
        """Test Jump Diffusion model."""
        data = await self.source.fetch(
            ticker="TEST",
            distribution_type=DistributionType.JUMP_DIFFUSION,
            periods=500,
            return_mean=0.001,
            return_std=0.015,
            jump_intensity=0.1,
            jump_size_mean=0.02,
            jump_size_std=0.01,
            seed=self.seed,
        )

        self.assertEqual(len(data), 500)
        df = data.to_pandas()
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        # Jump diffusion should have higher kurtosis due to jumps
        kurtosis = returns.kurtosis()
        self.assertGreater(kurtosis, 3)

    async def test_garch_model(self):
        """Test GARCH(1,1) volatility clustering."""
        data = await self.source.fetch(
            ticker="TEST",
            distribution_type=DistributionType.GARCH,
            periods=500,
            return_mean=0.001,
            garch_omega=0.0001,
            garch_alpha=0.1,
            garch_beta=0.85,
            seed=self.seed,
        )

        self.assertEqual(len(data), 500)
        df = data.to_pandas()
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        # Check that returns exist and are not all the same
        self.assertGreater(returns.std(), 0)

    async def test_regime_switching(self):
        """Test regime switching model."""
        data = await self.source.fetch(
            ticker="TEST",
            distribution_type=DistributionType.REGIME_SWITCHING,
            periods=500,
            regime_prob=0.02,
            regime_1_mean=0.001,
            regime_1_std=0.015,
            regime_2_mean=-0.002,
            regime_2_std=0.035,
            seed=self.seed,
        )

        self.assertEqual(len(data), 500)
        df = data.to_pandas()
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        # Should have varying volatility due to regime switching
        self.assertGreater(returns.std(), 0)

    async def test_frequency_daily(self):
        """Test daily frequency data generation."""
        data = await self.source.fetch(
            ticker="TEST", periods=50, freq="D", seed=self.seed
        )

        self.assertEqual(len(data), 50)
        df = data.to_pandas()

        # Check that dates are business days
        date_diffs = df["date"].diff().dropna()
        # Most differences should be 1 day (business days)
        # Some might be 3 days (weekends)

    async def test_frequency_hourly(self):
        """Test hourly frequency data generation."""
        data = await self.source.fetch(
            ticker="TEST", periods=24, freq="H", seed=self.seed
        )

        self.assertEqual(len(data), 24)

    async def test_frequency_minute(self):
        """Test minute frequency data generation."""
        data = await self.source.fetch(
            ticker="TEST", periods=60, freq="T", seed=self.seed
        )

        # May be less than 60 due to market hours filtering
        self.assertLessEqual(len(data), 60)
        self.assertGreater(len(data), 0)

    async def test_custom_intraday_params(self):
        """Test custom intraday price parameters."""
        custom_params = IntradayPriceParams(
            high_spread_mean=0.03,  # 3% spread
            high_spread_std=0.01,
            low_spread_mean=0.03,
            low_spread_std=0.01,
            gap_mean=0.005,  # 0.5% overnight gap
            gap_std=0.01,
        )

        data = await self.source.fetch(
            ticker="TEST", periods=100, intraday_params=custom_params, seed=self.seed
        )

        self.assertEqual(len(data), 100)
        df = data.to_pandas()

        # Check that spreads are roughly in expected range
        high_spread = df["High"] / df["Close"] - 1
        low_spread = df["Close"] / df["Low"] - 1

        # Most spreads should be in reasonable range around the mean
        self.assertGreater(high_spread.mean(), 0.01)  # At least 1%
        self.assertGreater(low_spread.mean(), 0.01)  # At least 1%

    async def test_custom_volume_params(self):
        """Test custom volume parameters."""
        custom_params = VolumeParams(
            base_volume=2000000,
            volume_std=0.5,
            volume_trend=0.1,
            volume_weekday_effect=True,
        )

        data = await self.source.fetch(
            ticker="TEST", periods=100, volume_params=custom_params, seed=self.seed
        )

        self.assertEqual(len(data), 100)
        df = data.to_pandas()

        # Check that volume is roughly around expected base
        self.assertGreater(df["Volume"].mean(), 1000000)  # Should be > 1M
        self.assertLess(df["Volume"].mean(), 5000000)  # Should be < 5M

    async def test_custom_market_params(self):
        """Test custom market parameters."""
        custom_params = MarketParams(
            start_price=200.0,
            start_date="2022-01-01",
            market_open_hour=10,
            market_close_hour=15,
        )

        data = await self.source.fetch(
            ticker="TEST", periods=50, market_params=custom_params, seed=self.seed
        )

        self.assertEqual(len(data), 50)
        df = data.to_pandas()

        # Check that first price is around start price
        self.assertAlmostEqual(df["Close"].iloc[0], 200.0, delta=20.0)

        # Check start date
        start_date = pd.to_datetime("2022-01-01")
        self.assertGreaterEqual(df["date"].iloc[0], start_date)

    async def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        seed = 123

        data1 = await self.source.fetch(
            ticker="TEST",
            periods=100,
            seed=seed,
            distribution_type=DistributionType.NORMAL,
        )

        # Create new instance to ensure fresh state
        source2 = SyntheticDataSource()
        data2 = await source2.fetch(
            ticker="TEST",
            periods=100,
            seed=seed,
            distribution_type=DistributionType.NORMAL,
        )

        # Results should be identical
        df1 = data1.to_pandas()
        df2 = data2.to_pandas()

        np.testing.assert_array_almost_equal(df1["Close"].values, df2["Close"].values)
        np.testing.assert_array_almost_equal(df1["Open"].values, df2["Open"].values)

    async def test_invalid_distribution_type(self):
        """Test that invalid distribution type raises error."""
        with self.assertRaises(ValueError):
            await self.source.fetch(
                ticker="TEST", distribution_type="invalid_distribution", periods=100
            )

    async def test_invalid_frequency(self):
        """Test that invalid frequency raises error."""
        with self.assertRaises(ValueError):
            await self.source.fetch(
                ticker="TEST",
                periods=100,
                freq="X",  # Invalid frequency
            )

    async def test_string_distribution_type(self):
        """Test that string distribution types are converted correctly."""
        data = await self.source.fetch(
            ticker="TEST",
            distribution_type="normal",  # String instead of enum
            periods=50,
            seed=self.seed,
        )

        self.assertEqual(len(data), 50)

    async def test_zero_periods(self):
        """Test behavior with zero periods."""
        data = await self.source.fetch(ticker="TEST", periods=0, seed=self.seed)
        self.assertEqual(len(data), 0)

    async def test_large_periods(self):
        """Test behavior with large number of periods."""
        data = await self.source.fetch(ticker="TEST", periods=5000, seed=self.seed)
        self.assertEqual(len(data), 5000)

    def test_generate_returns_methods(self):
        """Test individual return generation methods."""
        periods = 100

        # Test normal returns
        returns = self.source._generate_returns(
            DistributionType.NORMAL, periods, return_mean=0.001, return_std=0.02
        )
        self.assertEqual(len(returns), periods)
        self.assertTrue(np.isfinite(returns).all())

        # Test Student-t returns
        returns_t = self.source._generate_returns(
            DistributionType.STUDENT_T,
            periods,
            return_mean=0.001,
            return_std=0.02,
            df=3.0,
        )
        self.assertEqual(len(returns_t), periods)
        self.assertTrue(np.isfinite(returns_t).all())

    def test_generate_jump_diffusion_method(self):
        """Test jump diffusion generation method."""
        returns = self.source._generate_jump_diffusion(
            periods=100,
            drift=0.001,
            volatility=0.02,
            jump_intensity=0.1,
            jump_size_mean=0.02,
            jump_size_std=0.01,
        )

        self.assertEqual(len(returns), 100)
        self.assertTrue(np.isfinite(returns).all())

    def test_generate_garch_method(self):
        """Test GARCH generation method."""
        returns = self.source._generate_garch(
            periods=100, mean_return=0.001, omega=0.0001, alpha=0.1, beta=0.85
        )

        self.assertEqual(len(returns), 100)
        self.assertTrue(np.isfinite(returns).all())

    def test_generate_regime_switching_method(self):
        """Test regime switching generation method."""
        returns = self.source._generate_regime_switching(
            periods=100,
            switch_prob=0.02,
            regime_1_mean=0.001,
            regime_1_std=0.015,
            regime_2_mean=-0.002,
            regime_2_std=0.035,
        )

        self.assertEqual(len(returns), 100)
        self.assertTrue(np.isfinite(returns).all())

    def test_generate_ohlc_method(self):
        """Test OHLC generation from close prices."""
        close_prices = np.array([100, 101, 102, 99, 98])
        params = IntradayPriceParams()

        ohlc = self.source._generate_ohlc(close_prices, params)

        self.assertIn("open", ohlc)
        self.assertIn("high", ohlc)
        self.assertIn("low", ohlc)

        self.assertEqual(len(ohlc["open"]), 5)
        self.assertEqual(len(ohlc["high"]), 5)
        self.assertEqual(len(ohlc["low"]), 5)

        # Check OHLC relationships
        for i in range(5):
            self.assertGreaterEqual(
                ohlc["high"][i], max(ohlc["open"][i], close_prices[i])
            )
            self.assertLessEqual(ohlc["low"][i], min(ohlc["open"][i], close_prices[i]))

    def test_generate_volume_method(self):
        """Test volume generation method."""
        periods = 10
        date_index = pd.bdate_range("2023-01-01", periods=periods)
        params = VolumeParams(base_volume=1000000, volume_std=0.3)

        volumes = self.source._generate_volume(periods, date_index, params)

        self.assertEqual(len(volumes), periods)
        self.assertTrue(all(volumes > 0))
        self.assertTrue(all(volumes == volumes.astype(int)))  # Should be integers

    def test_gamma_spread_distribution(self):
        """Test gamma distribution for spreads."""
        params = IntradayPriceParams(spread_distribution="gamma")
        close_prices = np.array([100] * 50)

        ohlc = self.source._generate_ohlc(close_prices, params)

        # All spreads should be positive with gamma distribution
        high_spreads = ohlc["high"] / close_prices - 1
        low_spreads = close_prices / ohlc["low"] - 1

        self.assertTrue(all(high_spreads > 0))
        self.assertTrue(all(low_spreads > 0))

    def test_normal_spread_distribution(self):
        """Test normal distribution for spreads."""
        params = IntradayPriceParams(spread_distribution="normal")
        close_prices = np.array([100] * 50)

        ohlc = self.source._generate_ohlc(close_prices, params)

        # Spreads should be positive after clipping
        high_spreads = ohlc["high"] / close_prices - 1
        low_spreads = close_prices / ohlc["low"] - 1

        self.assertTrue(all(high_spreads > 0))
        self.assertTrue(all(low_spreads > 0))


class TestDistributionType(unittest.TestCase):
    """Test the DistributionType enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        expected_values = [
            "normal",
            "log_normal",
            "student_t",
            "laplace",
            "geometric_brownian_motion",
            "jump_diffusion",
            "garch",
            "regime_switching",
        ]

        for value in expected_values:
            self.assertIn(value, [dt.value for dt in DistributionType])

    def test_enum_conversion(self):
        """Test string to enum conversion."""
        self.assertEqual(DistributionType("normal"), DistributionType.NORMAL)
        self.assertEqual(DistributionType("garch"), DistributionType.GARCH)


class TestDataClasses(unittest.TestCase):
    """Test the data classes used for parameters."""

    def test_intraday_price_params_defaults(self):
        """Test IntradayPriceParams default values."""
        params = IntradayPriceParams()

        self.assertEqual(params.high_spread_mean, 0.015)
        self.assertEqual(params.low_spread_mean, 0.015)
        self.assertEqual(params.gap_mean, 0.0)
        self.assertEqual(params.spread_distribution, "gamma")

    def test_volume_params_defaults(self):
        """Test VolumeParams default values."""
        params = VolumeParams()

        self.assertEqual(params.base_volume, 1000000)
        self.assertEqual(params.volume_std, 0.3)
        self.assertEqual(params.volume_trend, 0.0)
        self.assertTrue(params.volume_weekday_effect)

    def test_market_params_defaults(self):
        """Test MarketParams default values."""
        params = MarketParams()

        self.assertEqual(params.trading_days_per_year, 252)
        self.assertEqual(params.start_price, 100.0)
        self.assertEqual(params.start_date, "2023-01-01")


class TestAsyncIntegration(unittest.TestCase):
    """Test async integration aspects."""

    def setUp(self):
        """Set up async test environment."""
        self.source = SyntheticDataSource()

    def test_async_fetch_runs(self):
        """Test that async fetch method runs without errors."""

        async def run_test():
            data = await self.source.fetch(ticker="TEST", periods=10, seed=42)
            return data

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            data = loop.run_until_complete(run_test())
            self.assertEqual(len(data), 10)
        finally:
            loop.close()


# Integration test runner for async tests
class AsyncTestCase(unittest.TestCase):
    """Base class for async tests."""

    def setUp(self):
        """Set up event loop for async tests."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up event loop."""
        self.loop.close()

    def run_async(self, coro):
        """Helper to run async coroutines in tests."""
        return self.loop.run_until_complete(coro)


# Run async tests
class TestSyntheticAsync(AsyncTestCase):
    """Async version of main tests for better test organization."""

    def setUp(self):
        super().setUp()
        self.source = SyntheticDataSource()

    def test_all_distributions_async(self):
        """Test all distribution types work in async context."""

        async def test_distributions():
            for dist_type in DistributionType:
                data = await self.source.fetch(
                    ticker="TEST", distribution_type=dist_type, periods=50, seed=42
                )
                self.assertEqual(len(data), 50)
                self.assertIsInstance(data, pl.DataFrame)

        self.run_async(test_distributions())


if __name__ == "__main__":
    # Create a test suite that handles both sync and async tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticDataSource))
    suite.addTests(loader.loadTestsFromTestCase(TestDistributionType))
    suite.addTests(loader.loadTestsFromTestCase(TestDataClasses))
    suite.addTests(loader.loadTestsFromTestCase(TestAsyncIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticAsync))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
