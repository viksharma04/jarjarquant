"""
Test suite for the data gatherer module.

This module tests all functionality in jarjarquant.data_gatherer,
including data source registration, fetching, and random sampling.
"""

import asyncio
from unittest.mock import AsyncMock, mock_open, patch

import pytest

import pandas as pd

from jarjarquant.data_gatherer import DataGatherer
from jarjarquant.data_gatherer.base import get_data_source, list_data_sources
from jarjarquant.data_gatherer.tws import TWSDataSource
from jarjarquant.data_gatherer.yahoo import YFinanceDataSource




@pytest.fixture
def data_gatherer():
    """Create a DataGatherer instance for tests."""
    return DataGatherer()


def test_data_gatherer_initialization(data_gatherer):
    """Test that DataGatherer initializes correctly."""
    assert isinstance(data_gatherer, DataGatherer)
    assert hasattr(data_gatherer, "_sources")
    assert len(data_gatherer._sources) > 0


def test_available_sources():
    """Test available_sources class method."""
    sources = DataGatherer.available_sources()
    assert isinstance(sources, list)
    assert "yahoo" in sources
    assert "synthetic" in sources


def test_get_random_tickers(data_gatherer):
    """Test get_random_tickers method."""
    with (
        patch("os.path.exists", return_value=True),
        patch(
            "builtins.open",
            mock_open(
                read_data="Symbol,Name\nAAPL,Apple\nMSFT,Microsoft\nGOOGL,Google"
            ),
        ),
    ):
        tickers = data_gatherer.get_random_tickers(2)
        assert len(tickers) == 2
        assert all(isinstance(ticker, str) for ticker in tickers)


@pytest.mark.asyncio
async def test_get_invalid_source(data_gatherer):
    """Test get method with invalid source."""
    with pytest.raises(ValueError):
        await data_gatherer.get("AAPL", source="invalid_source")


def test_sync_and_async_consistency(data_gatherer):
    """Test that sync and async methods are consistent."""
    mock_result = pd.DataFrame({"Close": [100, 101, 102]})

    with patch.object(
        data_gatherer, "get", new_callable=AsyncMock
    ) as mock_async:
        mock_async.return_value = mock_result

        # Call sync version
        sync_result = data_gatherer.get_sync(ticker="AAPL", source="yahoo")

        # Verify async method was called with correct parameters
        mock_async.assert_called_once_with("AAPL", "yahoo")

        # Results should match
        pd.testing.assert_frame_equal(sync_result, mock_result)


def test_data_source_registry_integration():
    """Test that random sampling works with registered data sources."""
    sources = list_data_sources()

    # Should have at least yahoo and synthetic
    assert "yahoo" in sources
    assert "synthetic" in sources

    # Should be able to get source classes
    yahoo_class = get_data_source("yahoo")
    assert yahoo_class == YFinanceDataSource

    tws_class = get_data_source("tws")
    assert tws_class == TWSDataSource


@pytest.mark.asyncio
async def test_get_random_ticker_async(data_gatherer):
    """Test async get method with 'random' ticker."""
    mock_data = pd.DataFrame({"Close": [100, 101, 102]})
    
    with (
        patch.object(data_gatherer, "get_random_tickers", return_value=["AAPL"]),
        patch.object(data_gatherer._sources["synthetic"], "fetch", new_callable=AsyncMock, return_value=mock_data)
    ):
        result = await data_gatherer.get("random", source="synthetic")
        
        # Verify get_random_tickers was called with 1
        data_gatherer.get_random_tickers.assert_called_once_with(1)
        
        # Verify fetch was called with the random ticker
        data_gatherer._sources["synthetic"].fetch.assert_called_once_with("AAPL")
        
        # Verify result is correct
        pd.testing.assert_frame_equal(result, mock_data)


def test_get_random_ticker_sync(data_gatherer):
    """Test sync get_sync method with 'random' ticker."""
    mock_data = pd.DataFrame({"Close": [100, 101, 102]})
    
    with (
        patch.object(data_gatherer, "get_random_tickers", return_value=["MSFT"]),
        patch.object(data_gatherer, "get", new_callable=AsyncMock, return_value=mock_data)
    ):
        result = data_gatherer.get_sync("random", source="yahoo")
        
        # Verify async get was called with correct parameters
        data_gatherer.get.assert_called_once_with("random", "yahoo")
        
        # Verify result is correct
        pd.testing.assert_frame_equal(result, mock_data)


@pytest.mark.asyncio
async def test_random_ticker_case_insensitive(data_gatherer):
    """Test that 'random' ticker works case-insensitively."""
    mock_data = pd.DataFrame({"Close": [100, 101, 102]})
    
    test_cases = ["random", "Random", "RANDOM", "RaNdOm"]
    
    for random_ticker in test_cases:
        with (
            patch.object(data_gatherer, "get_random_tickers", return_value=["GOOGL"]),
            patch.object(data_gatherer._sources["synthetic"], "fetch", new_callable=AsyncMock, return_value=mock_data)
        ):
            result = await data_gatherer.get(random_ticker, source="synthetic")
            
            # Verify get_random_tickers was called
            data_gatherer.get_random_tickers.assert_called_with(1)
            
            # Verify fetch was called with the random ticker
            data_gatherer._sources["synthetic"].fetch.assert_called_with("GOOGL")
            
            # Verify result is correct
            pd.testing.assert_frame_equal(result, mock_data)


