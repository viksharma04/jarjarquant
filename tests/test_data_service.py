"""Tests for the DuckDB data service."""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
from jarjarquant.data_service import DataService


@pytest.fixture
def data_service():
    """Create a DataService instance with the sample data path."""
    # Use the actual sample data path
    sample_data_path = Path(__file__).parent.parent / "jarjarquant" / "sample_data" / "data"
    
    if not sample_data_path.exists():
        pytest.skip(f"Sample data not found at {sample_data_path}")
    
    service = DataService(data_path=sample_data_path)
    yield service
    service.close()


class TestDataService:
    """Test cases for DataService functionality."""
    
    def test_initialization(self, data_service):
        """Test that DataService initializes correctly."""
        assert data_service is not None
        assert data_service.data_path.exists()
        assert data_service.prices_path.exists()
        assert data_service.equities_path.exists()
    
    def test_list_available_tickers(self, data_service):
        """Test listing available tickers."""
        tickers = data_service.list_available_tickers("equities")
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert "AAPL" in tickers
        assert all(isinstance(t, str) for t in tickers)
    
    def test_get_price_data_single_ticker(self, data_service):
        """Test getting price data for a single ticker."""
        df = data_service.get_price_data("AAPL")
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.index.name == "date"
        
        # Check expected columns
        expected_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected_cols:
            assert col in df.columns
    
    def test_get_price_data_multiple_tickers(self, data_service):
        """Test getting price data for multiple tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        df = data_service.get_price_data(tickers)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.index.names == ["ticker", "date"]
        
        # Check that all requested tickers are present
        unique_tickers = df.index.get_level_values("ticker").unique()
        for ticker in tickers:
            if Path(data_service.equities_path / "1d" / f"{ticker}.parquet").exists():
                assert ticker in unique_tickers
    
    def test_get_price_data_with_date_range(self, data_service):
        """Test getting price data with date filters."""
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        
        df = data_service.get_price_data(
            "AAPL",
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert df.index.min() >= pd.Timestamp(start_date)
            assert df.index.max() <= pd.Timestamp(end_date)
    
    def test_get_price_data_with_columns(self, data_service):
        """Test getting specific columns."""
        columns = ["Close", "Volume"]
        df = data_service.get_price_data("AAPL", columns=columns)
        
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert all(col in df.columns for col in columns)
            assert len(df.columns) == len(columns)
    
    def test_get_latest_prices(self, data_service):
        """Test getting latest prices."""
        tickers = ["AAPL", "MSFT"]
        df = data_service.get_latest_prices(tickers)
        
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert df.index.name == "ticker"
            assert "Close" in df.columns
            assert "Volume" in df.columns
            assert "date" in df.columns
    
    def test_get_metadata(self, data_service):
        """Test getting metadata."""
        # Skip if metadata doesn't exist
        metadata_path = data_service.equities_path / "equities_metadata.csv"
        if not metadata_path.exists():
            pytest.skip("Metadata file not found")
        
        df = data_service.get_metadata(tickers=["AAPL", "MSFT"])
        
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert df.index.name == "Symbol"
            assert "Sector" in df.columns
    
    def test_get_metadata_with_filters(self, data_service):
        """Test getting metadata with filters."""
        # Skip if metadata doesn't exist
        metadata_path = data_service.equities_path / "equities_metadata.csv"
        if not metadata_path.exists():
            pytest.skip("Metadata file not found")
        
        df = data_service.get_metadata(filters={"Sector": "Technology services"})
        
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert all(df["Sector"] == "Technology services")
    
    def test_get_sample_by_criteria(self, data_service):
        """Test getting sample tickers by criteria."""
        # Skip if metadata doesn't exist
        metadata_path = data_service.equities_path / "equities_metadata.csv"
        if not metadata_path.exists():
            pytest.skip("Metadata file not found")
        
        tickers = data_service.get_sample_by_criteria(
            n_samples=5,
            sector="Technology services",
            min_market_cap=1e11,  # 100 billion
            random_seed=42
        )
        
        assert isinstance(tickers, list)
        assert len(tickers) <= 5
    
    def test_get_date_range(self, data_service):
        """Test getting date range for a ticker."""
        start_date, end_date = data_service.get_date_range("AAPL")
        
        assert isinstance(start_date, pd.Timestamp)
        assert isinstance(end_date, pd.Timestamp)
        assert start_date < end_date
    
    def test_get_sectors(self, data_service):
        """Test getting unique sectors."""
        # Skip if metadata doesn't exist
        metadata_path = data_service.equities_path / "equities_metadata.csv"
        if not metadata_path.exists():
            pytest.skip("Metadata file not found")
        
        sectors = data_service.get_sectors()
        
        assert isinstance(sectors, list)
        if sectors:
            assert all(isinstance(s, str) for s in sectors)
            assert sectors == sorted(sectors)  # Should be sorted
    
    def test_get_analyst_ratings(self, data_service):
        """Test getting unique analyst ratings."""
        # Skip if metadata doesn't exist
        metadata_path = data_service.equities_path / "equities_metadata.csv"
        if not metadata_path.exists():
            pytest.skip("Metadata file not found")
        
        ratings = data_service.get_analyst_ratings()
        
        assert isinstance(ratings, list)
        if ratings:
            assert all(isinstance(r, str) for r in ratings)
            assert ratings == sorted(ratings)  # Should be sorted
    
    def test_context_manager(self):
        """Test using DataService as a context manager."""
        sample_data_path = Path(__file__).parent.parent / "jarjarquant" / "sample_data" / "data"
        
        if not sample_data_path.exists():
            pytest.skip(f"Sample data not found at {sample_data_path}")
        
        with DataService(data_path=sample_data_path) as service:
            tickers = service.list_available_tickers()
            assert len(tickers) > 0
    
    def test_invalid_ticker(self, data_service):
        """Test handling of invalid ticker."""
        df = data_service.get_price_data("INVALID_TICKER_XYZ")
        assert df.empty
    
    def test_invalid_frequency(self, data_service):
        """Test handling of unsupported frequency."""
        with pytest.raises(NotImplementedError):
            data_service.get_price_data("AAPL", frequency="1h")