"""DuckDB-based data repository for accessing financial data stored in Parquet files."""

import logging
import random
from pathlib import Path
from typing import Optional, Union

import duckdb
import polars as pl

from jarjarquant.schemas import BarSize, Sample, SampleRequest

logger = logging.getLogger(__name__)

BAR_SIZE_MAP = {
    BarSize.ONE_MINUTE: "1min",
    BarSize.ONE_HOUR: "1hour",
    BarSize.ONE_DAY: "1d",
}


class DuckDBRepository:
    """Data repository backed by Parquet files queried via DuckDB."""

    def __init__(self, data_path: Union[str, Path]) -> None:
        """Initialize with path to the data directory.

        Args:
            data_path: Base path containing prices/ subdirectory with Parquet files.
        """
        self.data_path = Path(data_path)
        self.prices_path = self.data_path / "prices"
        self._db_dir = self.data_path / "db"
        self._db_dir.mkdir(parents=True, exist_ok=True)

        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")

        self._conn = duckdb.connect(":memory:")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        bar_size: BarSize = BarSize.ONE_DAY,
    ) -> pl.DataFrame:
        """Read OHLCV data for a single ticker filtered by date range.

        Args:
            ticker: Ticker symbol (e.g. "AAPL").
            start_date: Inclusive start date string (YYYY-MM-DD).
            end_date: Inclusive end date string (YYYY-MM-DD).
            bar_size: Bar size for the data.

        Returns:
            Polars DataFrame with OHLCV columns.
        """
        bar_folder = BAR_SIZE_MAP.get(bar_size, "1d")
        file_path = self._find_ticker_file(ticker, bar_folder)

        if file_path is None:
            return pl.DataFrame()

        date_col = "date" if bar_size in (BarSize.ONE_DAY, BarSize.ONE_WEEK) else "datetime"

        query = f"""
            SELECT * FROM read_parquet('{file_path}')
            WHERE {date_col} >= '{start_date}' AND {date_col} <= '{end_date}'
            ORDER BY {date_col}
        """
        return self._conn.execute(query).pl()

    def get_sample(self, request: SampleRequest) -> Sample:
        """Get a random sample of ticker data.

        Args:
            request: Sampling parameters.

        Returns:
            Sample containing concatenated data for sampled tickers.
        """
        available = self.list_tickers()
        if not available:
            raise ValueError("No tickers available for sampling")

        n = min(request.n_samples, len(available))
        chosen = random.sample(available, n)

        bar_folder = BAR_SIZE_MAP.get(request.bar_size, "1d")
        date_col = "date" if request.bar_size in (BarSize.ONE_DAY, BarSize.ONE_WEEK) else "datetime"

        frames: list[pl.DataFrame] = []
        for ticker in chosen:
            file_path = self._find_ticker_file(ticker, bar_folder)
            if file_path is None:
                continue
            query = f"""
                SELECT '{ticker}' AS ticker, *
                FROM read_parquet('{file_path}')
                WHERE {date_col} >= '{request.start_date}'
                  AND {date_col} <= '{request.end_date}'
                ORDER BY {date_col}
            """
            frames.append(self._conn.execute(query).pl())

        if not frames:
            raise ValueError("No data found for sampled tickers in the requested date range")

        data = pl.concat(frames)

        return Sample(
            start_date=request.start_date,
            end_date=request.end_date,
            data=data,
            bar_size=request.bar_size,
        )

    def list_tickers(self, asset_type: str = "equities") -> list[str]:
        """Discover available tickers from Parquet file names.

        Args:
            asset_type: Subdirectory under prices/ (default "equities").

        Returns:
            Sorted list of ticker symbols.
        """
        path = self.prices_path / asset_type / "1d"
        if not path.exists():
            return []
        return sorted(f.stem for f in path.glob("*.parquet"))

    def save(self, table_name: str, data: pl.DataFrame) -> None:
        """Persist a DataFrame to a DuckDB database file.

        Args:
            table_name: Logical table name (also used as the db filename).
            data: Data to save.
        """
        if data is None or data.is_empty():
            raise ValueError("Cannot save empty or None data")

        db_file = self._db_dir / f"{table_name}.duckdb"
        conn = duckdb.connect(str(db_file))
        try:
            if self._table_exists(conn, table_name):
                conn.execute(f"INSERT INTO {table_name} SELECT * FROM data")
            else:
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM data")
        finally:
            conn.close()

    def load(self, table_name: str) -> pl.DataFrame:
        """Load a table from a DuckDB database file.

        Args:
            table_name: Logical table name.

        Returns:
            Polars DataFrame with table contents.

        Raises:
            FileNotFoundError: If the database file does not exist.
            ValueError: If the table does not exist in the database.
        """
        db_file = self._db_dir / f"{table_name}.duckdb"
        if not db_file.exists():
            raise FileNotFoundError(f"Database file {db_file} does not exist")

        conn = duckdb.connect(str(db_file))
        try:
            if not self._table_exists(conn, table_name):
                raise ValueError(f"Table {table_name} does not exist")
            return conn.execute(f"SELECT * FROM {table_name}").pl()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the in-memory DuckDB connection."""
        if hasattr(self, "_conn") and self._conn:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_ticker_file(self, ticker: str, bar_folder: str) -> Optional[str]:
        """Locate the Parquet file for a ticker across asset subdirectories."""
        for asset_dir in self.prices_path.iterdir():
            if not asset_dir.is_dir():
                continue
            candidate = asset_dir / bar_folder / f"{ticker}.parquet"
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()
            return result is not None and result[0] > 0
        except Exception:
            return False
