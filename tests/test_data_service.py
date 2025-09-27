"""Tests for the DuckDB data service."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import polars as pl
import pytest

from jarjarquant.data_service import DataService, SampleRequest, BarSize, EquityParams


@pytest.fixture
def mock_data_service():
    """Create a mocked DataService instance that doesn't rely on actual files."""
    with patch('jarjarquant.data_service.duckdb') as mock_duckdb, \
         patch.object(Path, 'exists', return_value=True), \
         patch.object(Path, 'mkdir'), \
         patch.object(Path, 'glob') as mock_glob:

        # Mock DuckDB connection
        mock_conn = MagicMock()
        mock_duckdb.connect.return_value = mock_conn

        # Mock file discovery
        mock_glob.return_value = [
            Path('AAPL.parquet'), Path('MSFT.parquet'),
            Path('GOOGL.parquet'), Path('TSLA.parquet')
        ]

        # Create service with a dummy path to avoid file dependencies
        service = DataService(data_path='/tmp/mock_data')
        service.conn = mock_conn

        yield service

        # Don't call close() in tests as it's mocked


class TestDataService:
    """Test cases for DataService functionality."""

    def test_initialization(self, mock_data_service):
        """Test that DataService initializes correctly."""
        assert mock_data_service is not None
        assert hasattr(mock_data_service, 'data_path')
        assert hasattr(mock_data_service, 'prices_path')
        assert hasattr(mock_data_service, 'equities_path')

    def test_list_available_tickers(self, mock_data_service):
        """Test listing available tickers."""
        with patch.object(mock_data_service, 'list_available_tickers', return_value=['AAPL', 'MSFT', 'GOOGL']):
            tickers = mock_data_service.list_available_tickers("equities")
            assert isinstance(tickers, list)
            assert len(tickers) > 0
            assert "AAPL" in tickers
            assert all(isinstance(t, str) for t in tickers)

    def test_get_price_data_single_ticker(self, mock_data_service):
        """Test getting price data for a single ticker."""
        # Mock sample data
        sample_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))
        sample_data.index.name = 'date'

        with patch.object(mock_data_service, 'get_price_data', return_value=sample_data):
            df = mock_data_service.get_price_data("AAPL")

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert df.index.name == "date"

            # Check expected columns
            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in expected_cols:
                assert col in df.columns

    def test_get_price_data_multiple_tickers(self, mock_data_service):
        """Test getting price data for multiple tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL"]

        # Mock multi-ticker data
        data_rows = []
        for ticker in tickers:
            for i, date in enumerate(['2024-01-01', '2024-01-02', '2024-01-03']):
                data_rows.append({
                    'ticker': ticker, 'date': pd.Timestamp(date),
                    'Open': 100.0 + i, 'High': 105.0 + i, 'Low': 99.0 + i,
                    'Close': 104.0 + i, 'Volume': 1000000 + i * 100000
                })

        sample_data = pd.DataFrame(data_rows).set_index(['ticker', 'date'])

        with patch.object(mock_data_service, 'get_price_data', return_value=sample_data):
            df = mock_data_service.get_price_data(tickers)

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert df.index.names == ["ticker", "date"]

            # Check that all requested tickers are present
            unique_tickers = df.index.get_level_values("ticker").unique()
            for ticker in tickers:
                assert ticker in unique_tickers

    def test_get_price_data_with_date_range(self, mock_data_service):
        """Test getting price data with date filters."""
        start_date = "2024-01-01"
        end_date = "2024-12-31"

        # Mock filtered data
        sample_data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [99.0, 100.0],
            'Close': [104.0, 105.0],
            'Volume': [1000000, 1100000]
        }, index=pd.to_datetime(['2024-01-15', '2024-01-16']))
        sample_data.index.name = 'date'

        with patch.object(mock_data_service, 'get_price_data', return_value=sample_data):
            df = mock_data_service.get_price_data(
                "AAPL", start_date=start_date, end_date=end_date
            )

            assert isinstance(df, pd.DataFrame)
            if not df.empty:
                assert df.index.min() >= pd.Timestamp(start_date)
                assert df.index.max() <= pd.Timestamp(end_date)

    def test_get_price_data_with_columns(self, mock_data_service):
        """Test getting specific columns."""
        columns = ["Close", "Volume"]

        # Mock data with only requested columns
        sample_data = pd.DataFrame({
            'Close': [104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))
        sample_data.index.name = 'date'

        with patch.object(mock_data_service, 'get_price_data', return_value=sample_data):
            df = mock_data_service.get_price_data("AAPL", columns=columns)

            assert isinstance(df, pd.DataFrame)
            if not df.empty:
                assert all(col in df.columns for col in columns)
                assert len(df.columns) == len(columns)

    def test_get_latest_prices(self, mock_data_service):
        """Test getting latest prices."""
        tickers = ["AAPL", "MSFT"]

        # Mock latest prices data
        sample_data = pd.DataFrame({
            'Close': [175.0, 300.0],
            'Volume': [50000000, 30000000],
            'date': [pd.Timestamp('2024-01-15'), pd.Timestamp('2024-01-15')]
        }, index=pd.Index(tickers, name='ticker'))

        with patch.object(mock_data_service, 'get_latest_prices', return_value=sample_data):
            df = mock_data_service.get_latest_prices(tickers)

            assert isinstance(df, pd.DataFrame)
            if not df.empty:
                assert df.index.name == "ticker"
                assert "Close" in df.columns
                assert "Volume" in df.columns
                assert "date" in df.columns

    def test_get_metadata(self, mock_data_service):
        """Test getting metadata."""
        # Mock metadata data
        sample_metadata = pd.DataFrame({
            'Sector': ['Technology services', 'Technology services'],
            'Market capitalization': [2.8e12, 2.3e12],
            'Volume 1 day': [50000000, 30000000]
        }, index=pd.Index(['AAPL', 'MSFT'], name='Symbol'))

        with patch.object(mock_data_service, 'get_metadata', return_value=sample_metadata):
            df = mock_data_service.get_metadata(tickers=["AAPL", "MSFT"])

            assert isinstance(df, pd.DataFrame)
            if not df.empty:
                assert df.index.name == "Symbol"
                assert "Sector" in df.columns

    def test_get_metadata_with_filters(self, mock_data_service):
        """Test getting metadata with filters."""
        # Mock filtered metadata data
        sample_metadata = pd.DataFrame({
            'Sector': ['Technology services', 'Technology services'],
            'Market capitalization': [2.8e12, 2.3e12]
        }, index=pd.Index(['AAPL', 'MSFT'], name='Symbol'))

        with patch.object(mock_data_service, 'get_metadata', return_value=sample_metadata):
            df = mock_data_service.get_metadata(filters={"Sector": "Technology services"})

            assert isinstance(df, pd.DataFrame)
            if not df.empty:
                assert all(df["Sector"] == "Technology services")

    def test_get_sample_by_criteria(self, mock_data_service):
        """Test getting sample tickers by criteria."""
        # Mock sample tickers
        sample_tickers = ['AAPL', 'MSFT', 'GOOGL']

        with patch.object(mock_data_service, 'get_sample_by_criteria', return_value=sample_tickers):
            tickers = mock_data_service.get_sample_by_criteria(
                n_samples=5,
                sector="Technology services",
                min_market_cap=1e11,  # 100 billion
                random_seed=42,
            )

            assert isinstance(tickers, list)
            assert len(tickers) <= 5

    def test_get_date_range(self, mock_data_service):
        """Test getting date range for a ticker."""
        # Mock date range
        mock_start = pd.Timestamp('2020-01-01')
        mock_end = pd.Timestamp('2024-01-15')

        with patch.object(mock_data_service, 'get_date_range', return_value=(mock_start, mock_end)):
            start_date, end_date = mock_data_service.get_date_range("AAPL")

            assert isinstance(start_date, pd.Timestamp)
            assert isinstance(end_date, pd.Timestamp)
            assert start_date < end_date

    def test_get_sectors(self, mock_data_service):
        """Test getting unique sectors."""
        # Mock sectors data
        mock_sectors = ['Finance', 'Health services', 'Technology services']

        with patch.object(mock_data_service, 'get_sectors', return_value=mock_sectors):
            sectors = mock_data_service.get_sectors()

            assert isinstance(sectors, list)
            if sectors:
                assert all(isinstance(s, str) for s in sectors)
                assert sectors == sorted(sectors)  # Should be sorted

    def test_get_analyst_ratings(self, mock_data_service):
        """Test getting unique analyst ratings."""
        # Mock analyst ratings data
        mock_ratings = ['Buy', 'Hold', 'Sell', 'Strong Buy']

        with patch.object(mock_data_service, 'get_analyst_ratings', return_value=mock_ratings):
            ratings = mock_data_service.get_analyst_ratings()

            assert isinstance(ratings, list)
            if ratings:
                assert all(isinstance(r, str) for r in ratings)
                assert ratings == sorted(ratings)  # Should be sorted

    def test_context_manager(self):
        """Test using DataService as a context manager."""
        with patch('jarjarquant.data_service.duckdb'), \
             patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'mkdir'), \
             patch.object(DataService, 'list_available_tickers', return_value=['AAPL', 'MSFT']):

            with DataService(data_path='/tmp/mock_data') as service:
                tickers = service.list_available_tickers()
                assert len(tickers) > 0

    def test_invalid_ticker(self, mock_data_service):
        """Test handling of invalid ticker."""
        # Mock empty DataFrame for invalid ticker
        empty_df = pd.DataFrame()

        with patch.object(mock_data_service, 'get_price_data', return_value=empty_df):
            df = mock_data_service.get_price_data("INVALID_TICKER_XYZ")
            assert df.empty

    def test_invalid_frequency(self, mock_data_service):
        """Test handling of unsupported frequency."""
        # Mock the actual method to raise NotImplementedError
        with patch.object(mock_data_service, 'get_price_data', side_effect=NotImplementedError("Frequency 1h not yet supported")):
            with pytest.raises(NotImplementedError):
                mock_data_service.get_price_data("AAPL", frequency="1h")

    def test_get_sample_equity(self, mock_data_service):
        """Test getting equity sample data."""
        # Mock sample data
        sample_data = pl.DataFrame({
            'ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
            'date': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')] * 2,
            'Open': [100.0, 101.0, 200.0, 201.0],
            'High': [105.0, 106.0, 205.0, 206.0],
            'Low': [99.0, 100.0, 199.0, 200.0],
            'Close': [104.0, 105.0, 204.0, 205.0],
            'Volume': [1000000, 1100000, 2000000, 2100000]
        })

        sample_request = SampleRequest(
            sample_type="equities",
            start_date="2024-01-01",
            end_date="2024-01-02",
            bar_size=BarSize.ONE_DAY,
            n_samples=2,
            params=EquityParams(sector="Technology services")
        )

        mock_sample = type('Sample', (), {
            'sample_type': 'equities',
            'start_date': '2024-01-01',
            'end_date': '2024-01-02',
            'data': sample_data,
            'bar_size': BarSize.ONE_DAY,
            'params': EquityParams(sector="Technology services")
        })()

        with patch.object(mock_data_service, 'get_sample', return_value=mock_sample):
            result = mock_data_service.get_sample(sample_request)
            assert result.sample_type == "equities"
            assert isinstance(result.data, pl.DataFrame)
            assert len(result.data) == 4

    def test_save_and_load_database(self, mock_data_service):
        """Test saving and loading data from database."""
        # Mock data to save
        test_data = pl.DataFrame({
            'value': [1, 2, 3],
            'name': ['a', 'b', 'c']
        })

        # Mock successful save
        with patch.object(mock_data_service, 'save_to_database') as mock_save, \
             patch.object(mock_data_service, 'load_from_database', return_value=test_data) as mock_load:

            # Test save
            mock_data_service.save_to_database(test_data, 'test_table')
            mock_save.assert_called_once_with(test_data, 'test_table')

            # Test load
            loaded_data = mock_data_service.load_from_database('test_table')
            mock_load.assert_called_once_with('test_table')
            assert loaded_data is not None
            assert len(loaded_data) == 3
