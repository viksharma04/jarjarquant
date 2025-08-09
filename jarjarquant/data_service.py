"""
DuckDB-based data service for accessing financial data.

This module provides a centralized interface for querying financial data
stored in Parquet files using DuckDB for efficient columnar operations.
"""

import logging
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DataService:
    """Manages access to financial data using DuckDB."""

    def __init__(self, data_path: Union[str, Path] = "sample_data/data/"):
        """
        Initialize the DataService.

        Args:
            data_path: Base path to the data directory. Defaults to sample_data/data/
        """
        self.data_path = Path(data_path)
        self.prices_path = self.data_path / "prices"
        self.equities_path = self.prices_path / "equities"
        self.forex_path = self.prices_path / "forex"

        # Verify paths exist
        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")

        # Initialize DuckDB connection (in-memory by default)
        self.conn = duckdb.connect(":memory:")

        # Configure DuckDB for better performance
        self.conn.execute("SET threads TO 4")
        self.conn.execute("SET memory_limit = '1GB'")

        # Register views for different data sources
        self._register_data_views()

    def _register_data_views(self) -> None:
        """Register DuckDB views for available data sources."""
        # Register equities view if 1d data exists
        equities_1d_path = self.equities_path / "1d"
        if equities_1d_path.exists():
            try:
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW equities_1d AS
                    SELECT * FROM read_parquet('{equities_1d_path}/*.parquet', 
                                             filename=true,
                                             union_by_name=true)
                """)
                logger.info("Registered equities_1d view")
            except Exception as e:
                logger.warning(f"Could not register equities_1d view: {e}")

        # Register metadata view if exists
        metadata_path = self.equities_path / "equities_metadata.csv"
        if metadata_path.exists():
            try:
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW equities_metadata AS
                    SELECT * FROM read_csv_auto('{metadata_path}')
                """)
                logger.info("Registered equities_metadata view")
            except Exception as e:
                logger.warning(f"Could not register equities_metadata view: {e}")

    def get_price_data(
        self,
        ticker: Union[str, List[str]],
        start_date: Optional[Union[datetime, date, str]] = None,
        end_date: Optional[Union[datetime, date, str]] = None,
        columns: Optional[List[str]] = None,
        frequency: str = "1d",
    ) -> pd.DataFrame:
        """
        Get price data for one or more tickers.

        Args:
            ticker: Single ticker symbol or list of tickers
            start_date: Start date for data (inclusive)
            end_date: End date for data (inclusive)
            columns: Specific columns to return. If None, returns all columns
            frequency: Data frequency (currently only "1d" supported)

        Returns:
            DataFrame with requested price data
        """
        if frequency != "1d":
            raise NotImplementedError(f"Frequency {frequency} not yet supported")

        # Convert single ticker to list
        if isinstance(ticker, str):
            tickers = [ticker]
        else:
            tickers = ticker

        # Build query
        ticker_files = [
            str(self.equities_path / "1d" / f"{t}.parquet") for t in tickers
        ]

        # Create union query for multiple tickers
        union_parts = []
        for t, file in zip(tickers, ticker_files):
            if Path(file).exists():
                part = f"SELECT '{t}' as ticker, * FROM read_parquet('{file}')"
                union_parts.append(part)

        if not union_parts:
            return pd.DataFrame()

        query = " UNION ALL ".join(union_parts)

        # Add date filters if provided
        conditions = []
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            conditions.append(f"date >= '{start_date}'")

        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            conditions.append(f"date <= '{end_date}'")

        if conditions:
            query = f"SELECT * FROM ({query}) WHERE {' AND '.join(conditions)}"

        # Select specific columns if requested
        if columns:
            col_list = ", ".join(["ticker", "date"] + columns)
            query = f"SELECT {col_list} FROM ({query})"

        # Execute query
        result = self.conn.execute(query).df()

        # Set index
        if not result.empty:
            if len(tickers) == 1:
                result = result.drop("ticker", axis=1)
                result = result.set_index("date")
            else:
                result = result.set_index(["ticker", "date"])

        return result

    def get_latest_prices(
        self,
        tickers: Optional[List[str]] = None,
        columns: List[str] = ["Close", "Volume"],
    ) -> pd.DataFrame:
        """
        Get the latest available price for each ticker.

        Args:
            tickers: List of tickers. If None, returns all available tickers
            columns: Columns to return

        Returns:
            DataFrame with latest prices
        """
        if tickers is None:
            # Get all available tickers
            tickers = self.list_available_tickers()

        if not tickers:
            return pd.DataFrame()

        # Build query to get latest date for each ticker
        ticker_files = [
            (t, str(self.equities_path / "1d" / f"{t}.parquet")) for t in tickers
        ]

        union_parts = []
        for t, file in ticker_files:
            if Path(file).exists():
                cols = ", ".join(columns)
                part = f"""
                WITH latest AS (
                    SELECT date, {cols}
                    FROM read_parquet('{file}')
                    ORDER BY date DESC
                    LIMIT 1
                )
                SELECT '{t}' as ticker, date, {cols} FROM latest
                """
                union_parts.append(f"({part})")

        if not union_parts:
            return pd.DataFrame()

        query = " UNION ALL ".join(union_parts)
        result = self.conn.execute(query).df()

        if not result.empty:
            result = result.set_index("ticker")

        return result

    def get_metadata(
        self,
        tickers: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Get metadata for tickers.

        Args:
            tickers: List of tickers to get metadata for
            filters: Dictionary of column:value filters to apply

        Returns:
            DataFrame with metadata
        """
        query = "SELECT * FROM equities_metadata"

        conditions = []

        if tickers:
            ticker_list = "', '".join(tickers)
            conditions.append(f"Symbol IN ('{ticker_list}')")

        if filters:
            for col, val in filters.items():
                # Simple column quoting for SQL safety
                quoted_col = f'"{col}"'
                if isinstance(val, str):
                    conditions.append(f"{quoted_col} = '{val}'")
                elif isinstance(val, (list, tuple)):
                    val_list = "', '".join(str(v) for v in val)
                    conditions.append(f"{quoted_col} IN ('{val_list}')")
                else:
                    conditions.append(f"{quoted_col} = {val}")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            result = self.conn.execute(query).df()
            if not result.empty and "Symbol" in result.columns:
                result = result.set_index("Symbol")
            return result
        except Exception as e:
            logger.warning(f"Could not fetch metadata: {e}")
            return pd.DataFrame()

    def get_sample_by_criteria(
        self,
        n_samples: int = 10,
        sector: Optional[str] = None,
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        min_volume: Optional[float] = None,
        analyst_rating: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> List[str]:
        """
        Get a sample of tickers based on metadata criteria.

        Args:
            n_samples: Number of samples to return
            sector: Filter by sector
            min_market_cap: Minimum market capitalization
            max_market_cap: Maximum market capitalization
            min_volume: Minimum average volume
            analyst_rating: Filter by analyst rating
            random_seed: Random seed for reproducible sampling

        Returns:
            List of ticker symbols matching criteria
        """
        query = "SELECT Symbol FROM equities_metadata"

        conditions = []

        if sector:
            conditions.append(f"Sector = '{sector}'")

        if min_market_cap:
            conditions.append(f'"Market capitalization" >= {min_market_cap}')

        if max_market_cap:
            conditions.append(f'"Market capitalization" <= {max_market_cap}')

        if min_volume:
            conditions.append(f'"Volume 1 day" >= {min_volume}')

        if analyst_rating:
            conditions.append(f"\"Analyst Rating\" = '{analyst_rating}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Add sampling
        if random_seed:
            query += f" USING SAMPLE {n_samples} (SYSTEM, {random_seed})"
        else:
            query += f" USING SAMPLE {n_samples}"

        try:
            result = self.conn.execute(query).df()
            return result["Symbol"].tolist() if not result.empty else []
        except Exception as e:
            logger.warning(f"Could not get sample: {e}")
            return []

    @lru_cache(maxsize=1)
    def list_available_tickers(self, asset_type: str = "equities") -> List[str]:
        """
        List all available tickers for a given asset type.

        Args:
            asset_type: Type of asset ("equities", "forex", etc.)

        Returns:
            List of available ticker symbols
        """
        if asset_type == "equities":
            path = self.equities_path / "1d"
            if path.exists():
                return sorted([f.stem for f in path.glob("*.parquet")])
        elif asset_type == "forex":
            path = self.forex_path / "1d"
            if path.exists():
                return sorted([f.stem for f in path.glob("*.parquet")])

        return []

    def get_date_range(self, ticker: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the available date range for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Tuple of (start_date, end_date)
        """
        file_path = self.equities_path / "1d" / f"{ticker}.parquet"

        if not file_path.exists():
            raise ValueError(f"No data available for ticker {ticker}")

        query = f"""
        SELECT MIN(date) as start_date, MAX(date) as end_date
        FROM read_parquet('{file_path}')
        """

        result = self.conn.execute(query).df()
        return result.iloc[0]["start_date"], result.iloc[0]["end_date"]

    def get_sectors(self) -> List[str]:
        """Get unique sectors from metadata."""
        try:
            result = self.conn.execute(
                "SELECT DISTINCT Sector FROM equities_metadata"
            ).df()
            return sorted(result["Sector"].dropna().tolist())
        except Exception:
            return []

    def get_analyst_ratings(self) -> List[str]:
        """Get unique analyst ratings from metadata."""
        try:
            result = self.conn.execute(
                'SELECT DISTINCT "Analyst Rating" FROM equities_metadata'
            ).df()
            return sorted(result["Analyst Rating"].dropna().tolist())
        except Exception:
            return []

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
