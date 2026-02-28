"""Tests for DuckDB storage backend."""

import polars as pl
import pytest
from jarjarquant.storage.duckdb import DuckDBRepository
from jarjarquant.schemas import BarSize


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a minimal Parquet file structure for testing."""
    prices_dir = tmp_path / "prices" / "equities" / "1d"
    prices_dir.mkdir(parents=True)
    df = pl.DataFrame(
        {
            "date": pl.date_range(
                pl.date(2020, 1, 1), pl.date(2020, 12, 31), eager=True
            ),
            "Open": [100.0] * 366,
            "High": [105.0] * 366,
            "Low": [95.0] * 366,
            "Close": [102.0] * 366,
            "Volume": [1000.0] * 366,
        }
    )
    df.write_parquet(str(prices_dir / "TEST.parquet"))
    return tmp_path


def test_duckdb_repo_list_tickers(tmp_data_dir):
    repo = DuckDBRepository(str(tmp_data_dir))
    tickers = repo.list_tickers()
    assert "TEST" in tickers


def test_duckdb_repo_get_prices(tmp_data_dir):
    repo = DuckDBRepository(str(tmp_data_dir))
    df = repo.get_prices("TEST", "2020-01-01", "2020-06-30")
    assert isinstance(df, pl.DataFrame)
    assert len(df) > 0
    assert "Close" in df.columns


def test_duckdb_repo_save_and_load(tmp_data_dir):
    repo = DuckDBRepository(str(tmp_data_dir))
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    repo.save("test_table", df)
    loaded = repo.load("test_table")
    assert loaded.shape == (3, 2)
