"""
DuckDB-based data service for accessing financial data.

This module provides a centralized interface for querying financial data
stored in Parquet files using DuckDB for efficient columnar operations.
"""

import logging
import os
import random
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

import duckdb
import pandas as pd
import polars as pl

from jarjarquant.config import LOCAL_DB_PATH
from jarjarquant.data_gatherer.utils import BarSize

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BaseParams:
    pass


@dataclass(slots=True)
class EquityParams(BaseParams):
    sector: Optional[str] = None
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    min_volume: Optional[int] = None
    analyst_rating: Optional[str] = None


@dataclass(slots=True)
class ForexParams(BaseParams):
    pass


@dataclass(slots=True)
class SampleRequest:
    sample_type: Literal["equities", "forex", "crypto", "precious_metals"]
    start_date: str
    end_date: str
    bar_size: BarSize = BarSize.ONE_DAY
    n_samples: int = 10
    params: Optional[BaseParams] = None

    def __post_init__(self):
        if self.params is None:
            if self.sample_type == "equities":
                self.params = EquityParams()
            elif self.sample_type == "forex":
                self.params = ForexParams()
            else:
                self.params = BaseParams()


TParams = TypeVar("TParams", bound=BaseParams)


@dataclass(slots=True)
class Sample(Generic[TParams]):
    sample_type: Literal["equities", "forex", "crypto", "precious_metals"]
    start_date: str
    end_date: str
    data: pl.DataFrame
    bar_size: BarSize
    params: TParams


# Type aliases for different sample types
EquitySample = Sample[EquityParams]
ForexSample = Sample[ForexParams]


class DataService:
    """Manages access to financial data using DuckDB."""

    # Class-level constants
    BAR_SIZE_MAP = {
        BarSize.ONE_MINUTE: "1min",
        BarSize.ONE_HOUR: "1hour",
        BarSize.ONE_DAY: "1d",
    }

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        """
        Initialize the DataService.

        Args:
            data_path: Base path to the data directory. Defaults to LOCAL_DB_PATH from config
        """
        # If using default path, use the configured LOCAL_DB_PATH
        if data_path is None:
            local_db_path = Path(LOCAL_DB_PATH)
            if local_db_path.is_absolute():
                # LOCAL_DB_PATH is an absolute path, use as-is
                self.data_path = local_db_path
            else:
                # LOCAL_DB_PATH is relative, resolve relative to current directory
                self.data_path = Path.cwd() / local_db_path
        else:
            self.data_path = Path(data_path)

        self.prices_path = self.data_path / "prices"
        self.equities_path = self.prices_path / "equities"
        self.forex_path = self.prices_path / "forex"
        self.db_path = Path(__file__).parent / "db"

        # Verify paths exist
        if not self.data_path.exists():
            logger.error(f"Current working directory: {Path.cwd()}")
            logger.error(f"Resolved data path: {self.data_path.resolve()}")
            raise ValueError(f"Data path {self.data_path} does not exist")

        # Ensure db path exists
        self.db_path.mkdir(exist_ok=True)

        # Initialize DuckDB connection (in-memory by default)
        try:
            self.conn = duckdb.connect(":memory:")
            # Configure DuckDB for better performance with env variables
            threads = int(os.getenv("DUCKDB_THREADS", "4"))
            memory_limit = os.getenv("DUCKDB_MEMORY_LIMIT", "1GB")
            self.conn.execute(f"SET threads TO {threads}")
            self.conn.execute(f"SET memory_limit = '{memory_limit}'")
            logger.debug(
                f"DuckDB configured with {threads} threads and {memory_limit} memory"
            )
        except duckdb.Error as e:
            logger.error(f"Failed to initialize DuckDB connection: {e}")
            raise

        # Register views for different data sources
        self._register_data_views()

    def _get_date_column_name(self, bar_size: BarSize) -> str:
        """Get the appropriate date column name based on bar size."""
        return "date" if bar_size in (BarSize.ONE_DAY, BarSize.ONE_WEEK) else "datetime"

    def _build_ticker_file_list(
        self, tickers: List[str], asset_path: Path, bar_folder: str
    ) -> List[tuple[str, str]]:
        """Build and filter list of ticker files that exist."""
        ticker_files = [
            (ticker, str(asset_path / bar_folder / f"{ticker}.parquet"))
            for ticker in tickers
        ]
        return [(ticker, file) for ticker, file in ticker_files if Path(file).exists()]

    def _build_union_query(self, ticker_files: List[tuple[str, str]]) -> str:
        """Build a union query from ticker file list."""
        if not ticker_files:
            raise ValueError("No ticker files provided for union query")

        union_parts = [
            f"SELECT '{ticker}' as ticker, * FROM read_parquet('{file}')"
            for ticker, file in ticker_files
        ]
        return " UNION ALL ".join(union_parts)

    def _register_data_views(self) -> None:
        """Register DuckDB views for available data sources."""
        # Register metadata view if exists
        metadata_path = self.equities_path / "equities_metadata.csv"
        if metadata_path.exists():
            try:
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW equities_metadata AS
                    SELECT * FROM read_csv_auto('{metadata_path}')
                """)
                logger.info("Registered equities_metadata view")
            except (duckdb.Error, FileNotFoundError, OSError) as e:
                logger.warning(f"Could not register equities_metadata view: {e}")
        else:
            logger.warning(
                f"Metadata file not found at {metadata_path}. Some features may not work."
            )

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

        Raises:
            ValueError: If ticker is empty or invalid date range provided
            NotImplementedError: If unsupported frequency requested
        """
        # Parameter validation
        if not ticker or (isinstance(ticker, list) and not ticker):
            raise ValueError("Ticker cannot be empty")

        if frequency != "1d":
            raise NotImplementedError(f"Frequency {frequency} not yet supported")

        # Validate date range if both provided
        if start_date and end_date:
            start_dt = start_date
            end_dt = end_date

            if isinstance(start_date, str):
                try:
                    start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                except ValueError:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            elif isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_dt = datetime.combine(start_date, datetime.min.time())

            if isinstance(end_date, str):
                try:
                    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                except ValueError:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            elif isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_dt = datetime.combine(
                    end_date, datetime.max.time().replace(microsecond=0)
                )

            if start_dt > end_dt:
                raise ValueError("start_date cannot be after end_date")

        # Convert single ticker to list
        if isinstance(ticker, str):
            tickers = [ticker]
        else:
            tickers = ticker

        # Build query
        ticker_files = [
            str(self.equities_path / "1d" / f"{t}.parquet") for t in tickers
        ]

        # Build ticker file list and create union query
        ticker_files = [
            (t, str(self.equities_path / "1d" / f"{t}.parquet")) for t in tickers
        ]
        existing_files = [(t, f) for t, f in ticker_files if Path(f).exists()]

        if not existing_files:
            return pd.DataFrame()

        query = self._build_union_query(existing_files)

        # Add date filters if provided
        conditions = []
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(start_date, date):
                start_date = start_date.strftime("%Y-%m-%d")
            conditions.append(f"date >= '{start_date}'")

        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(end_date, date):
                end_date = end_date.strftime("%Y-%m-%d")
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
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get the latest available price for each ticker.

        Args:
            tickers: List of tickers. If None, returns all available tickers
            columns: Columns to return

        Returns:
            DataFrame with latest prices
        """
        if columns is None:
            columns = ["Close", "Volume"]

        if tickers is None:
            # Get all available tickers
            tickers = self.list_available_tickers()

        if not tickers:
            return pd.DataFrame()

        # Build query to get latest date for each ticker
        ticker_files = self._build_ticker_file_list(tickers, self.equities_path, "1d")

        if not ticker_files:
            return pd.DataFrame()

        union_parts = []
        for ticker, file in ticker_files:
            cols = ", ".join(columns)
            part = f"""
            WITH latest AS (
                SELECT date, {cols}
                FROM read_parquet('{file}')
                ORDER BY date DESC
                LIMIT 1
            )
            SELECT '{ticker}' as ticker, date, {cols} FROM latest
            """
            union_parts.append(f"({part})")

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
            placeholders = ", ".join(["?"] * len(tickers))
            conditions.append(f"Symbol IN ({placeholders})")

        if filters:
            for col, val in filters.items():
                # Properly quote column name and parameterize values
                quoted_col = f'"{col}"'
                if isinstance(val, str):
                    conditions.append(f"{quoted_col} = ?")
                elif isinstance(val, (list, tuple)):
                    placeholders = ", ".join(["?"] * len(val))
                    conditions.append(f"{quoted_col} IN ({placeholders})")
                else:
                    conditions.append(f"{quoted_col} = ?")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            # Build parameters list for parameterized query
            params = []
            if tickers:
                params.extend(tickers)
            if filters:
                for col, val in filters.items():
                    if isinstance(val, str):
                        params.append(val)
                    elif isinstance(val, (list, tuple)):
                        params.extend(val)
                    else:
                        params.append(val)

            result = (
                self.conn.execute(query, params).df()
                if params
                else self.conn.execute(query).df()
            )
            if not result.empty and "Symbol" in result.columns:
                result = result.set_index("Symbol")
            return result
        except (duckdb.Error, ValueError, KeyError) as e:
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
        params = []

        if sector:
            conditions.append("Sector = ?")
            params.append(sector)

        if min_market_cap:
            conditions.append('"Market capitalization" >= ?')
            params.append(min_market_cap)

        if max_market_cap:
            conditions.append('"Market capitalization" <= ?')
            params.append(max_market_cap)

        if min_volume:
            conditions.append('"Volume 1 day" >= ?')
            params.append(min_volume)

        if analyst_rating:
            conditions.append('"Analyst Rating" = ?')
            params.append(analyst_rating)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Add sampling
        if random_seed:
            query += f" USING SAMPLE {n_samples} (SYSTEM, {random_seed})"
        else:
            query += f" USING SAMPLE {n_samples}"

        try:
            result = (
                self.conn.execute(query, params).df()
                if params
                else self.conn.execute(query).df()
            )
            return result["Symbol"].tolist() if not result.empty else []
        except (duckdb.Error, ValueError) as e:
            logger.warning(f"Could not get sample: {e}")
            return []

    @lru_cache(maxsize=16)
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
        if result.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return result.iloc[0]["start_date"], result.iloc[0]["end_date"]

    def get_sectors(self) -> List[str]:
        """Get unique sectors from metadata."""
        try:
            result = self.conn.execute(
                "SELECT DISTINCT Sector FROM equities_metadata"
            ).df()
            return sorted(result["Sector"].dropna().tolist())
        except (duckdb.Error, KeyError) as e:
            logger.warning(f"Could not fetch sectors: {e}")
            return []

    def get_analyst_ratings(self) -> List[str]:
        """Get unique analyst ratings from metadata."""
        try:
            result = self.conn.execute(
                'SELECT DISTINCT "Analyst Rating" FROM equities_metadata'
            ).df()
            return sorted(result["Analyst Rating"].dropna().tolist())
        except (duckdb.Error, KeyError) as e:
            logger.warning(f"Could not fetch analyst ratings: {e}")
            return []

    def get_sample(self, sample_request: SampleRequest) -> Sample:
        """
        Get a sample of financial data based on the request parameters.

        Args:
            sample_request: Request parameters including sample type, date range,
                           bar size, number of samples, and filtering criteria

        Returns:
            Sample object containing the requested data as a Polars DataFrame

        Raises:
            ValueError: If invalid sample request parameters
            NotImplementedError: If unsupported sample type requested
        """
        # Parameter validation
        if sample_request.n_samples <= 0:
            raise ValueError("n_samples must be positive")

        if not sample_request.start_date or not sample_request.end_date:
            raise ValueError("Both start_date and end_date must be provided")
        if sample_request.sample_type == "equities":
            return self._get_equity_sample(sample_request)
        elif sample_request.sample_type == "forex":
            return self._get_forex_sample(sample_request)
        else:
            raise NotImplementedError(
                f"Sample type {sample_request.sample_type} not yet supported"
            )

    def _get_equity_sample(self, sample_request: SampleRequest) -> EquitySample:
        """
        Get a sample of equity data based on the request parameters.

        Args:
            sample_request: Request parameters for equity data

        Returns:
            Sample object containing equity data as a Polars DataFrame
        """
        # Get sample tickers based on criteria
        if not isinstance(sample_request.params, EquityParams):
            raise ValueError("EquityParams required for equity samples")

        tickers = self.get_sample_by_criteria(
            n_samples=sample_request.n_samples,
            sector=sample_request.params.sector,
            min_market_cap=sample_request.params.min_market_cap,
            max_market_cap=sample_request.params.max_market_cap,
            min_volume=sample_request.params.min_volume,
            analyst_rating=sample_request.params.analyst_rating,
        )

        if not tickers:
            raise ValueError("No equity tickers found matching the specified criteria")

        # Build DuckDB query to fetch data for all tickers
        bar_folder = self.BAR_SIZE_MAP.get(sample_request.bar_size, "1d")

        # Filter to only existing files
        existing_files = self._build_ticker_file_list(
            tickers, self.equities_path, bar_folder
        )

        if not existing_files:
            raise ValueError("No equity data files found for the sampled tickers")

        return self._build_sample_from_files(existing_files, sample_request)

    def _get_forex_sample(self, sample_request: SampleRequest) -> ForexSample:
        """
        Get a sample of forex data based on the request parameters.

        Args:
            sample_request: Request parameters for forex data

        Returns:
            Sample object containing forex data as a Polars DataFrame
        """
        # Get available forex tickers (no metadata filtering for forex)
        available_forex = self.list_available_tickers("forex")

        if not available_forex:
            raise ValueError("No forex data available")

        # Sample random tickers if we have more than requested
        if len(available_forex) > sample_request.n_samples:
            tickers = random.sample(available_forex, sample_request.n_samples)
        else:
            tickers = available_forex

        # Build DuckDB query to fetch data for all tickers
        bar_folder = self.BAR_SIZE_MAP.get(sample_request.bar_size, "1d")

        # Filter to only existing files
        existing_files = self._build_ticker_file_list(
            tickers, self.forex_path, bar_folder
        )

        if not existing_files:
            raise ValueError("No forex data files found for the sampled tickers")

        return self._build_sample_from_files(existing_files, sample_request)

    def _build_sample_from_files(
        self, existing_files: List[tuple], sample_request: SampleRequest
    ) -> Sample:
        """
        Build a Sample object from a list of ticker/file pairs.

        Args:
            existing_files: List of (ticker, file_path) tuples
            sample_request: Original sample request

        Returns:
            Sample object with the requested data
        """
        # Build union query for multiple tickers
        query = self._build_union_query(existing_files)

        # Add date range filters
        conditions = []
        date_column = self._get_date_column_name(sample_request.bar_size)

        if sample_request.start_date:
            conditions.append(f"{date_column} >= '{sample_request.start_date}'")
        if sample_request.end_date:
            conditions.append(f"{date_column} <= '{sample_request.end_date}'")

        if conditions:
            query = f"SELECT * FROM ({query}) WHERE {' AND '.join(conditions)}"

        # Add ordering for consistency
        query += f" ORDER BY ticker, {date_column}"

        # Execute query and get result as Polars DataFrame using DuckDB's native pl() method
        try:
            pl_df = self.conn.execute(query).pl()
        except AttributeError:
            # Fallback to Arrow if pl() not available
            try:
                arrow_result = self.conn.execute(query).arrow()
                pl_df = pl.from_arrow(arrow_result)
            except AttributeError:
                # Final fallback to pandas conversion
                result_df = self.conn.execute(query).df()
                pl_df = pl.from_pandas(result_df)

        # Ensure proper date/datetime type
        if "date" in pl_df.columns and sample_request.bar_size == BarSize.ONE_DAY:
            pl_df = pl_df.with_columns(pl.col("date").dt.date())
        elif "datetime" in pl_df.columns and sample_request.bar_size != BarSize.ONE_DAY:
            pl_df = pl_df.with_columns(pl.col("datetime").dt.replace_time_zone(None))

        return Sample(
            sample_type=sample_request.sample_type,
            start_date=sample_request.start_date,
            end_date=sample_request.end_date,
            data=pl_df,
            bar_size=sample_request.bar_size,
            params=sample_request.params,
        )

    def close(self) -> None:
        """Close the DuckDB connection."""
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
                logger.debug("DuckDB connection closed successfully")
        except Exception as e:
            logger.warning(f"Error closing DuckDB connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        # Don't suppress exceptions
        return False

    def save_to_database(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        table_name: str,
        append: bool = True,
    ) -> None:
        """
        Save tabular data to a DuckDB table.

        Args:
            data: DataFrame to save (pandas or polars)
            table_name: Name of the table
            append: If True, append to existing table. If False, overwrite.

        Raises:
            ValueError: If data is empty or invalid
            IOError: If unable to write to database
        """
        if (
            data is None
            or (isinstance(data, pd.DataFrame) and data.empty)
            or (isinstance(data, pl.DataFrame) and data.height == 0)
            or (hasattr(data, "__len__") and len(data) == 0)
        ):
            raise ValueError("Cannot save empty or None data")

        # Convert to polars if pandas
        if isinstance(data, pd.DataFrame):
            pl_data = pl.from_pandas(data)
        else:
            pl_data = data

        try:
            # Create database file path for persistent storage
            db_file_path = self.db_path / f"{table_name}.duckdb"
            # Create parent directory if it doesn't exist
            db_file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"db file path type: {type(db_file_path)}")
            # Use persistent database connection for this table
            # DuckDB will create the file automatically if it doesn't exist
            table_conn = duckdb.connect(db_file_path)

            try:
                if append and self._table_exists(table_conn, table_name):
                    # Insert into existing table
                    table_conn.execute(
                        f"INSERT INTO {table_name} SELECT * FROM pl_data"
                    )
                    logger.info(f"Appended {pl_data.height} rows to table {table_name}")
                else:
                    # Create new table or replace existing
                    if not append and self._table_exists(table_conn, table_name):
                        table_conn.execute(f"DROP TABLE {table_name}")

                    table_conn.execute(
                        f"CREATE TABLE {table_name} AS SELECT * FROM pl_data"
                    )
                    logger.info(
                        f"Created table {table_name} with {pl_data.height} rows"
                    )

            finally:
                table_conn.close()

        except Exception as e:
            logger.error(f"Failed to save data to table {table_name}: {e}")
            raise IOError(f"Unable to write to database table {table_name}: {e}")

    def _table_exists(self, conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            conn: DuckDB connection
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        try:
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()
            return result is not None and result[0] > 0
        except Exception:
            return False

    def load_from_database(self, table_name: str) -> Optional[pl.DataFrame]:
        """
        Load data from a DuckDB table.

        Args:
            table_name: Name of the table

        Returns:
            Polars DataFrame if table exists, None otherwise

        Raises:
            IOError: If unable to read the table
        """
        db_file_path = self.db_path / f"{table_name}.duckdb"

        if not db_file_path.exists():
            logger.warning(f"Database file {db_file_path} does not exist")
            return None

        try:
            # Use persistent database connection for this table
            table_conn = duckdb.connect(str(db_file_path))

            try:
                if not self._table_exists(table_conn, table_name):
                    logger.warning(f"Table {table_name} does not exist in database")
                    return None

                # Query the table and convert to Polars DataFrame
                result = table_conn.execute(f"SELECT * FROM {table_name}").pl()
                logger.debug(f"Loaded {result.height} rows from table {table_name}")
                return result

            finally:
                table_conn.close()

        except Exception as e:
            logger.error(f"Failed to load data from table {table_name}: {e}")
            raise IOError(f"Unable to read database table {table_name}: {e}")
