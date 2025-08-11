import os
from datetime import datetime, timedelta
from typing import Optional, Union

import duckdb
import polars as pl
from dateutil.relativedelta import relativedelta

from .base import DataSource, register_data_source
from .utils import DURATION_TO_DAYS_MAP, BarSize, Duration


@register_data_source("custom")
class CustomDataSource(DataSource):
    async def fetch(
        self,
        ticker: str = "MSFT",
        bar_size: BarSize = BarSize.ONE_DAY,
        duration: Duration = Duration.ONE_MONTH,
        end_time: Optional[Union[str, datetime]] = None,
        security_type: str = "STK",
        database_folder: str = "sample_data/",
        data_type: str = "prices",
    ) -> pl.DataFrame:
        """
        Fetch data from local parquet files.

        Args:
            ticker: Symbol to fetch (e.g., "AAPL" for equities, "EURUSD" for forex)
            bar_size: Time resolution of the data
            duration: Historical period to fetch
            end_time: End time for historical data. Can be:
                     - String: "YYYY-MM-DD" for daily, "YYYY-MM-DD HH:MM:SS" for intraday
                     - datetime object (automatically formatted based on bar_size)
                     - None: Uses latest available data
            security_type: Type of security ("STK" for stocks, "CASH" for forex)
            database_folder: Base folder containing the data
            data_type: Type of data to fetch ("prices" or "iv" for implied volatility)

        Returns:
            Polars DataFrame with the requested data
        """
        # Map security types to folder names
        security_map = {
            "STK": "equities",
            "CASH": "forex",
            "FX": "forex",
            "FOREX": "forex",
        }
        sec_folder = security_map.get(security_type.upper(), security_type.lower())

        # Map bar_size to folder name
        bar_size_map = {
            BarSize.ONE_MINUTE: "1min",
            BarSize.ONE_HOUR: "1hour",
            BarSize.ONE_DAY: "1d",
        }
        bar_folder = bar_size_map.get(bar_size, str(bar_size).lower())

        # Build path to parquet file
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Handle different data types and folder structures
        if data_type.lower() == "iv":
            # IV data is only available for equities and doesn't have bar_size subfolders
            if sec_folder != "equities":
                raise ValueError(
                    f"IV data is only available for equities, not {security_type}"
                )
            parquet_path = os.path.join(
                parent_dir,
                database_folder,
                "data",
                "iv",
                sec_folder,
                f"{ticker}.parquet",
            )
        else:
            # Regular price data
            parquet_path = os.path.join(
                parent_dir,
                database_folder,
                "data",
                "prices",
                sec_folder,
                bar_folder,
                f"{ticker}.parquet",
            )

        parquet_path = os.path.normpath(parquet_path)

        # Check if file exists
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Data file not found: {parquet_path}. "
                f"Available data types: {data_type}, security: {security_type}, "
                f"bar_size: {bar_size}, ticker: {ticker}"
            )

        con = duckdb.connect(database=":memory:")
        query = f"SELECT * FROM '{parquet_path}'"

        # Apply time-based filtering if end_time is specified
        if end_time is not None:
            # Parse end_time based on type and bar_size
            parsed_end_time = self._parse_end_time(end_time, bar_size)
            start_time = self._calculate_start_time(parsed_end_time, duration, bar_size)

            # Determine column name and format for filtering
            if bar_size == BarSize.ONE_DAY:
                date_column = "date"
                start_str = start_time.strftime("%Y-%m-%d")
                end_str = parsed_end_time.strftime("%Y-%m-%d")
            else:
                # For intraday data, use datetime column
                date_column = "datetime"
                start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
                end_str = parsed_end_time.strftime("%Y-%m-%d %H:%M:%S")

            # Use >= for start to be inclusive, <= for end to be inclusive
            query += (
                f" WHERE {date_column} >= '{start_str}'"
                f" AND {date_column} <= '{end_str}'"
            )

        df = con.execute(query).fetch_df()
        con.close()

        # Convert pandas DataFrame to Polars DataFrame
        pl_df = pl.from_pandas(df)

        # Standardize date/datetime columns and types
        if "datetime" in pl_df.columns and "date" not in pl_df.columns:
            # Some files have datetime column - standardize naming
            if bar_size == BarSize.ONE_DAY:
                # Daily data should use 'date' column with Date dtype
                pl_df = pl_df.with_columns(
                    pl.col("datetime").dt.date().alias("date")
                ).drop("datetime")
            else:
                # Intraday data keeps 'datetime' column as Datetime
                pass
        elif "date" in pl_df.columns:
            if bar_size == BarSize.ONE_DAY:
                # Daily data should have Date dtype
                pl_df = pl_df.with_columns(pl.col("date").dt.date().alias("date"))
            else:
                # Intraday data should rename to 'datetime' and keep as Datetime
                pl_df = pl_df.rename({"date": "datetime"})

        # Remove timezone/offset columns if present (inconsistent across files)
        if "gmtoffset" in pl_df.columns:
            pl_df = pl_df.drop("gmtoffset")
        if "timestamp" in pl_df.columns:
            pl_df = pl_df.drop("timestamp")

        return pl_df

    def _parse_end_time(
        self, end_time: Union[str, datetime], bar_size: BarSize
    ) -> datetime:
        """Parse end_time parameter into datetime object with appropriate formatting."""
        if isinstance(end_time, datetime):
            return end_time

        # Parse string input
        if isinstance(end_time, str):
            # Try different datetime formats based on bar_size and input
            if bar_size == BarSize.ONE_DAY:
                # For daily data, accept date-only format
                try:
                    return datetime.strptime(end_time, "%Y-%m-%d")
                except ValueError:
                    # Also try datetime format for daily (will truncate to date later)
                    return datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            else:
                # For intraday data, try full datetime format first
                try:
                    return datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        return datetime.strptime(end_time, "%Y-%m-%d %H:%M")
                    except ValueError:
                        # Fall back to date-only and set to end of day
                        dt = datetime.strptime(end_time, "%Y-%m-%d")
                        return dt.replace(hour=23, minute=59, second=59)

        raise ValueError(f"Invalid end_time format: {end_time}")

    def _calculate_start_time(
        self, end_time: datetime, duration: Duration, bar_size: BarSize
    ) -> datetime:
        """Calculate start time based on end time, duration, and bar size."""
        if bar_size == BarSize.ONE_DAY:
            # For daily data, use days from the duration mapping
            duration_days = DURATION_TO_DAYS_MAP[duration]
            return end_time - relativedelta(days=duration_days)
        else:
            # For intraday data, be more precise with duration calculation
            if duration == Duration.ONE_DAY:
                return end_time - timedelta(days=1)
            elif duration == Duration.ONE_WEEK:
                return end_time - timedelta(days=7)
            elif duration == Duration.ONE_MONTH:
                return end_time - relativedelta(months=1)
            elif duration == Duration.THREE_MONTHS:
                return end_time - relativedelta(months=3)
            elif duration == Duration.SIX_MONTHS:
                return end_time - relativedelta(months=6)
            elif duration == Duration.ONE_YEAR:
                return end_time - relativedelta(years=1)
            else:
                # Fallback to days-based calculation
                duration_days = DURATION_TO_DAYS_MAP.get(duration, 30)
                return end_time - timedelta(days=duration_days)
