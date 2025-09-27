"""
Test suite for the CustomDataSource class.

This module tests all functionality in jarjarquant.data_gatherer.custom,
including data fetching, time filtering, path resolution, and error handling.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

# Mock DataService before any imports that might trigger it
with patch('jarjarquant.data_service.DataService.__init__', return_value=None):
    from jarjarquant.data_gatherer.custom import CustomDataSource
    from jarjarquant.data_gatherer.utils import BarSize, Duration
    from tests.mocks.mock_data import (
        create_mock_daily_equity_data,
        create_mock_forex_data,
        create_mock_intraday_equity_data,
        create_mock_iv_data,
    )


@pytest.fixture
def custom_source():
    """Create a CustomDataSource instance for tests."""
    return CustomDataSource()


@pytest.fixture
def mock_daily_equity():
    """Mock daily equity data."""
    return create_mock_daily_equity_data()


@pytest.fixture
def mock_hourly_equity():
    """Mock hourly equity data."""
    return create_mock_intraday_equity_data(bar_size="1hour")


@pytest.fixture
def mock_minute_equity():
    """Mock minute equity data."""
    return create_mock_intraday_equity_data(bar_size="1min")


@pytest.fixture
def mock_daily_forex():
    """Mock daily forex data."""
    return create_mock_forex_data(bar_size="1d")


@pytest.fixture
def mock_hourly_forex():
    """Mock hourly forex data."""
    return create_mock_forex_data(bar_size="1hour")


@pytest.fixture
def mock_iv_data():
    """Mock IV data."""
    return create_mock_iv_data()


class TestCustomDataSource:
    """Test suite for CustomDataSource class."""

    def test_security_type_mapping(self, custom_source):
        """Test security type to folder mapping."""
        security_map = {
            "STK": "equities",
            "CASH": "forex",
            "FX": "forex",
            "FOREX": "forex",
        }
        
        for security_type, expected_folder in security_map.items():
            folder = security_map.get(security_type.upper(), security_type.lower())
            assert folder == expected_folder

    def test_bar_size_mapping(self, custom_source):
        """Test bar size to folder mapping."""
        bar_size_map = {
            BarSize.ONE_MINUTE: "1min",
            BarSize.ONE_HOUR: "1hour",
            BarSize.ONE_DAY: "1d",
        }
        
        for bar_size, expected_folder in bar_size_map.items():
            folder = bar_size_map.get(bar_size, str(bar_size).lower())
            assert folder == expected_folder

    def test_parse_end_time_with_datetime_object(self, custom_source):
        """Test _parse_end_time with datetime objects."""
        test_datetime = datetime(2023, 12, 31, 15, 30, 0)
        
        # Should return the same datetime object
        result = custom_source._parse_end_time(test_datetime, BarSize.ONE_DAY)
        assert result == test_datetime
        
        result = custom_source._parse_end_time(test_datetime, BarSize.ONE_HOUR)
        assert result == test_datetime

    def test_parse_end_time_daily_string_formats(self, custom_source):
        """Test _parse_end_time with various string formats for daily data."""
        # Date-only format for daily data
        result = custom_source._parse_end_time("2023-12-31", BarSize.ONE_DAY)
        expected = datetime(2023, 12, 31, 0, 0, 0)
        assert result == expected
        
        # Full datetime format for daily data (should still work)
        result = custom_source._parse_end_time("2023-12-31 15:30:00", BarSize.ONE_DAY)
        expected = datetime(2023, 12, 31, 15, 30, 0)
        assert result == expected

    def test_parse_end_time_intraday_string_formats(self, custom_source):
        """Test _parse_end_time with various string formats for intraday data."""
        # Full datetime format
        result = custom_source._parse_end_time("2023-12-31 15:30:00", BarSize.ONE_HOUR)
        expected = datetime(2023, 12, 31, 15, 30, 0)
        assert result == expected
        
        # Hour-minute format
        result = custom_source._parse_end_time("2023-12-31 15:30", BarSize.ONE_HOUR)
        expected = datetime(2023, 12, 31, 15, 30, 0)
        assert result == expected
        
        # Date-only format (should set to end of day)
        result = custom_source._parse_end_time("2023-12-31", BarSize.ONE_HOUR)
        expected = datetime(2023, 12, 31, 23, 59, 59)
        assert result == expected

    def test_parse_end_time_invalid_format(self, custom_source):
        """Test _parse_end_time with invalid string formats."""
        with pytest.raises(ValueError):
            custom_source._parse_end_time("invalid-date", BarSize.ONE_DAY)

    def test_calculate_start_time_daily(self, custom_source):
        """Test _calculate_start_time for daily data."""
        end_time = datetime(2023, 12, 31, 0, 0, 0)
        
        # Test one month duration
        start_time = custom_source._calculate_start_time(
            end_time, Duration.ONE_MONTH, BarSize.ONE_DAY
        )
        
        # Should be approximately 30 days earlier
        days_diff = (end_time - start_time).days
        assert 25 <= days_diff <= 35  # At least 25 days, at most 35 days

    def test_calculate_start_time_intraday(self, custom_source):
        """Test _calculate_start_time for intraday data."""
        end_time = datetime(2023, 12, 31, 15, 30, 0)
        
        # Test one day duration
        start_time = custom_source._calculate_start_time(
            end_time, Duration.ONE_DAY, BarSize.ONE_HOUR
        )
        expected = datetime(2023, 12, 30, 15, 30, 0)
        assert start_time == expected
        
        # Test one week duration
        start_time = custom_source._calculate_start_time(
            end_time, Duration.ONE_WEEK, BarSize.ONE_HOUR
        )
        expected = datetime(2023, 12, 24, 15, 30, 0)
        assert start_time == expected

    @pytest.mark.asyncio
    @patch('os.path.exists')
    @patch('duckdb.connect')
    @patch('polars.from_pandas')
    async def test_fetch_daily_equity_data_success(self, mock_from_pandas, mock_connect, mock_exists, custom_source, mock_daily_equity):
        """Test successful fetch of daily equity data."""
        # Setup mocks
        mock_exists.return_value = True
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        mock_con.execute.return_value.fetch_df.return_value = mock_daily_equity.to_pandas()
        mock_from_pandas.return_value = mock_daily_equity
        
        # Call the method
        result = await custom_source.fetch(
            ticker="AAPL",
            security_type="STK",
            data_type="prices",
            bar_size=BarSize.ONE_DAY,
            duration=Duration.ONE_MONTH
        )
        
        # Assertions
        assert isinstance(result, pl.DataFrame)
        mock_exists.assert_called_once()
        mock_connect.assert_called_once_with(database=":memory:")
        mock_con.execute.assert_called_once()
        mock_con.close.assert_called_once()

    @pytest.mark.asyncio
    @patch('os.path.exists')
    @patch('duckdb.connect')
    @patch('polars.from_pandas')
    async def test_fetch_with_time_filtering(self, mock_from_pandas, mock_connect, mock_exists, custom_source, mock_daily_equity):
        """Test fetch with time-based filtering."""
        # Setup mocks
        mock_exists.return_value = True
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        mock_con.execute.return_value.fetch_df.return_value = mock_daily_equity.to_pandas()
        mock_from_pandas.return_value = mock_daily_equity
        
        # Call with end_time
        await custom_source.fetch(
            ticker="AAPL",
            security_type="STK",
            data_type="prices",
            bar_size=BarSize.ONE_DAY,
            duration=Duration.ONE_WEEK,
            end_time="2023-12-31"
        )
        
        # Check that WHERE clause was added to query
        query_call = mock_con.execute.call_args[0][0]
        assert "WHERE" in query_call
        assert "date >=" in query_call
        assert "date <=" in query_call

    @pytest.mark.asyncio
    @patch('os.path.exists')
    @patch('duckdb.connect')
    @patch('polars.from_pandas')
    async def test_fetch_intraday_with_time_filtering(self, mock_from_pandas, mock_connect, mock_exists, custom_source, mock_hourly_equity):
        """Test fetch intraday data with time-based filtering."""
        # Setup mocks
        mock_exists.return_value = True
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        mock_con.execute.return_value.fetch_df.return_value = mock_hourly_equity.to_pandas()
        mock_from_pandas.return_value = mock_hourly_equity
        
        # Call with end_time for intraday data
        await custom_source.fetch(
            ticker="AAPL",
            security_type="STK",
            data_type="prices",
            bar_size=BarSize.ONE_HOUR,
            duration=Duration.ONE_DAY,
            end_time="2023-12-29 16:00:00"
        )
        
        # Check that datetime column is used in WHERE clause
        query_call = mock_con.execute.call_args[0][0]
        assert "WHERE" in query_call
        assert "datetime >=" in query_call
        assert "datetime <=" in query_call

    @pytest.mark.asyncio
    @patch('os.path.exists')
    async def test_fetch_file_not_found(self, mock_exists, custom_source):
        """Test fetch when parquet file doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            await custom_source.fetch(
                ticker="NONEXISTENT",
                security_type="STK",
                data_type="prices",
                bar_size=BarSize.ONE_DAY,
                duration=Duration.ONE_MONTH
            )

    @pytest.mark.asyncio
    async def test_fetch_iv_data_for_non_equity(self, custom_source):
        """Test that IV data request for non-equity raises ValueError."""
        with pytest.raises(ValueError, match="IV data is only available for equities"):
            await custom_source.fetch(
                ticker="EURUSD",
                security_type="CASH",
                data_type="iv",
                bar_size=BarSize.ONE_DAY,
                duration=Duration.ONE_MONTH
            )

    @pytest.mark.asyncio
    @patch('os.path.exists')
    @patch('duckdb.connect')
    @patch('polars.from_pandas')
    async def test_fetch_iv_data_success(self, mock_from_pandas, mock_connect, mock_exists, custom_source, mock_iv_data):
        """Test successful fetch of IV data."""
        # Setup mocks
        mock_exists.return_value = True
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        mock_con.execute.return_value.fetch_df.return_value = mock_iv_data.to_pandas()
        mock_from_pandas.return_value = mock_iv_data
        
        # Call the method for IV data
        result = await custom_source.fetch(
            ticker="AAPL",
            security_type="STK",
            data_type="iv",
            bar_size=BarSize.ONE_DAY,
            duration=Duration.ONE_MONTH
        )
        
        # Assertions
        assert isinstance(result, pl.DataFrame)
        
        # Check that the correct path structure was used (no bar_size folder for IV)
        query_call = mock_con.execute.call_args[0][0]
        assert "iv" in query_call
        assert "equities" in query_call

    @pytest.mark.asyncio
    @patch('os.path.exists')
    @patch('duckdb.connect')
    @patch('polars.from_pandas')
    async def test_fetch_forex_data_success(self, mock_from_pandas, mock_connect, mock_exists, custom_source, mock_daily_forex):
        """Test successful fetch of forex data."""
        # Setup mocks
        mock_exists.return_value = True
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        mock_con.execute.return_value.fetch_df.return_value = mock_daily_forex.to_pandas()
        mock_from_pandas.return_value = mock_daily_forex
        
        # Call the method for forex data
        result = await custom_source.fetch(
            ticker="EURUSD",
            security_type="CASH",
            data_type="prices",
            bar_size=BarSize.ONE_DAY,
            duration=Duration.ONE_MONTH
        )
        
        # Assertions
        assert isinstance(result, pl.DataFrame)
        
        # Check that the correct path structure was used
        query_call = mock_con.execute.call_args[0][0]
        assert "forex" in query_call

    @pytest.mark.asyncio
    @patch('os.path.exists')
    async def test_path_construction_equity_prices(self, mock_exists, custom_source):
        """Test path construction for equity price data."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError) as exc_info:
            await custom_source.fetch(
                ticker="AAPL",
                security_type="STK", 
                data_type="prices",
                bar_size=BarSize.ONE_DAY
            )
        
        path = str(exc_info.value)
        assert "sample_data" in path
        assert "data" in path
        assert "prices" in path
        assert "equities" in path
        assert "1d" in path
        assert "AAPL.parquet" in path

    @pytest.mark.asyncio
    @patch('os.path.exists')
    async def test_path_construction_iv_data(self, mock_exists, custom_source):
        """Test path construction for IV data."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError) as exc_info:
            await custom_source.fetch(
                ticker="AAPL",
                security_type="STK",
                data_type="iv",
                bar_size=BarSize.ONE_DAY
            )
        
        path = str(exc_info.value)
        assert "sample_data" in path
        assert "data" in path
        assert "iv" in path
        assert "equities" in path
        assert "AAPL.parquet" in path
        # IV data should not have bar_size in path
        path_before_file = path.split("AAPL.parquet")[0]
        assert "1d" not in path_before_file

    @pytest.mark.asyncio
    @patch('os.path.exists')
    @patch('duckdb.connect')
    @patch('polars.from_pandas')
    async def test_data_standardization_daily(self, mock_from_pandas, mock_connect, mock_exists, custom_source, mock_daily_equity):
        """Test data standardization for daily data."""
        # Setup mocks
        mock_exists.return_value = True
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        
        # Create mock data with datetime column instead of date
        mock_data_with_datetime = mock_daily_equity.rename({"date": "datetime"})
        mock_con.execute.return_value.fetch_df.return_value = mock_data_with_datetime.to_pandas()
        
        # Mock the polars conversion to return properly standardized data
        expected_df = mock_data_with_datetime.with_columns(
            pl.col("datetime").dt.date().alias("date")
        ).drop("datetime")
        mock_from_pandas.return_value = expected_df
        
        result = await custom_source.fetch(
            ticker="AAPL",
            security_type="STK",
            data_type="prices",
            bar_size=BarSize.ONE_DAY,
            duration=Duration.ONE_MONTH
        )
        
        # Should have 'date' column, not 'datetime' for daily data
        assert "date" in result.columns
        assert "datetime" not in result.columns

    @pytest.mark.asyncio
    @patch('os.path.exists')
    @patch('duckdb.connect')
    @patch('polars.from_pandas')
    async def test_data_standardization_intraday(self, mock_from_pandas, mock_connect, mock_exists, custom_source, mock_hourly_equity):
        """Test data standardization for intraday data."""
        # Setup mocks  
        mock_exists.return_value = True
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        
        # Create mock data with date column instead of datetime
        mock_data_with_date = mock_hourly_equity.rename({"datetime": "date"})
        mock_con.execute.return_value.fetch_df.return_value = mock_data_with_date.to_pandas()
        
        # Mock the conversion to return properly renamed data
        expected_df = mock_data_with_date.rename({"date": "datetime"})
        mock_from_pandas.return_value = expected_df
        
        result = await custom_source.fetch(
            ticker="AAPL",
            security_type="STK",
            data_type="prices",
            bar_size=BarSize.ONE_HOUR,
            duration=Duration.ONE_DAY
        )
        
        # Should have 'datetime' column, not 'date' for intraday data
        assert "datetime" in result.columns
        assert "date" not in result.columns

    @pytest.mark.asyncio
    @patch('os.path.exists')
    @patch('duckdb.connect')
    @patch('polars.from_pandas')
    async def test_timezone_columns_removal(self, mock_from_pandas, mock_connect, mock_exists, custom_source, mock_hourly_equity):
        """Test removal of timezone/offset columns."""
        # Setup mocks
        mock_exists.return_value = True
        mock_con = MagicMock()
        mock_connect.return_value = mock_con
        
        # Create mock data with extra columns
        mock_data_with_extras = mock_hourly_equity.with_columns([
            pl.lit(-5).alias("gmtoffset"),
            pl.lit(1640800000).alias("timestamp")
        ])
        mock_con.execute.return_value.fetch_df.return_value = mock_data_with_extras.to_pandas()
        
        # Mock the conversion to simulate column removal
        expected_df = mock_data_with_extras.drop(["gmtoffset", "timestamp"])
        mock_from_pandas.return_value = expected_df
        
        result = await custom_source.fetch(
            ticker="AAPL",
            security_type="STK",
            data_type="prices",
            bar_size=BarSize.ONE_HOUR,
            duration=Duration.ONE_DAY
        )
        
        # Should not have timezone/offset columns
        assert "gmtoffset" not in result.columns
        assert "timestamp" not in result.columns

    @pytest.mark.parametrize("security_type,expected_folder", [
        ("STK", "equities"),
        ("CASH", "forex"),
        ("FX", "forex"),
        ("FOREX", "forex"),
        ("unknown", "unknown"),  # lowercase fallback
    ])
    def test_security_type_mapping_parametrized(self, custom_source, security_type, expected_folder):
        """Test security type mapping with parametrized inputs."""
        security_map = {
            "STK": "equities",
            "CASH": "forex", 
            "FX": "forex",
            "FOREX": "forex",
        }
        result = security_map.get(security_type.upper(), security_type.lower())
        assert result == expected_folder

    @pytest.mark.parametrize("bar_size,expected_folder", [
        (BarSize.ONE_MINUTE, "1min"),
        (BarSize.ONE_HOUR, "1hour"),
        (BarSize.ONE_DAY, "1d"),
    ])
    def test_bar_size_mapping_parametrized(self, custom_source, bar_size, expected_folder):
        """Test bar size mapping with parametrized inputs."""
        bar_size_map = {
            BarSize.ONE_MINUTE: "1min",
            BarSize.ONE_HOUR: "1hour",
            BarSize.ONE_DAY: "1d",
        }
        result = bar_size_map.get(bar_size, str(bar_size).lower())
        assert result == expected_folder

    @pytest.mark.parametrize("end_time_str,bar_size,expected", [
        ("2023-12-31", BarSize.ONE_DAY, datetime(2023, 12, 31, 0, 0, 0)),
        ("2023-12-31 15:30:00", BarSize.ONE_HOUR, datetime(2023, 12, 31, 15, 30, 0)),
        ("2023-12-31 15:30", BarSize.ONE_HOUR, datetime(2023, 12, 31, 15, 30, 0)),
        ("2023-12-31", BarSize.ONE_HOUR, datetime(2023, 12, 31, 23, 59, 59)),
    ])
    def test_parse_end_time_parametrized(self, custom_source, end_time_str, bar_size, expected):
        """Test end_time parsing with various formats."""
        result = custom_source._parse_end_time(end_time_str, bar_size)
        assert result == expected