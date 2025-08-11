from datetime import datetime
from typing import Optional

import httpx
import polars as pl
from dateutil.relativedelta import relativedelta

from .base import DataSource, register_data_source
from .utils import (
    BAR_SIZE_TO_STR_MAP,
    DURATION_TO_DAYS_MAP,
    BarSize,
    Duration,
    convert_date_to_unixtime,
)


@register_data_source("eodhd")
class EODHDDataSource(DataSource):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a reusable HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20,
                    keepalive_expiry=60.0,
                ),
            )
        return self._client

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def close(self):
        """Explicitly close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def fetch(
        self,
        ticker: str = "SPY",
        bar_size: BarSize = BarSize.ONE_DAY,
        duration: Duration = Duration.ONE_MONTH,
        end_date: Optional[str] = None,
        security_type: str = "STK",
        **kwargs,
    ) -> pl.DataFrame:
        if not self.api_key:
            raise ValueError("EODHD API key not configured.")

        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        if security_type == "STK":
            ticker = ticker + ".US"

        bar_size_map = BAR_SIZE_TO_STR_MAP["eodhd"]
        if bar_size not in list(bar_size_map.keys()):
            raise ValueError(
                "bar_size can only be 1/5 min, 1 hour, day, week, or month"
            )
        eodhd_period = bar_size_map.get(bar_size, "d")

        if duration is not Duration.MAX:
            # Convert duration to days (simple approximation)
            if duration not in list(DURATION_TO_DAYS_MAP.keys()):
                raise ValueError("Invalid duration")
            duration_days = DURATION_TO_DAYS_MAP.get(duration, 30)

            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - relativedelta(days=duration_days)
            start_date = start_dt.strftime("%Y-%m-%d")
        else:
            start_date = None
            end_date = None

        if bar_size not in [BarSize.ONE_MINUTE, BarSize.FIVE_MINUTES, BarSize.ONE_HOUR]:
            url = f"https://eodhd.com/api/eod/{ticker}?period={eodhd_period}&api_token={self.api_key}&fmt=json"
            if start_date is not None:
                url += f"&from={start_date}"
            if end_date is not None:
                url += f"&to={end_date}"
            try:
                client = await self._get_client()
                r = await client.get(url)
                r.raise_for_status()
                series = r.json()
            except Exception:
                return pl.DataFrame()
        else:
            if start_date and end_date is not None:
                from_unix_time, to_unix_time = convert_date_to_unixtime(
                    start_date, end_date
                )
            else:
                from_unix_time = None
                to_unix_time = None
            try:
                url = f"https://eodhd.com/api/intraday/{ticker}?api_token={self.api_key}&interval={eodhd_period}&fmt=json"
                if from_unix_time is not None:
                    url += f"&from={from_unix_time}"
                if to_unix_time is not None:
                    url += f"&to={to_unix_time}"
                client = await self._get_client()
                r = await client.get(url)
                r.raise_for_status()
                series = r.json()

            except Exception as e:
                print(f"{ticker}: Data fetching error: {e}")
                return pl.DataFrame()

        if not series:
            return pl.DataFrame()

        df = pl.DataFrame(series)
        df = df.rename(
            mapping={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        # Ensure Volume is Float64 for consistency across all data sources
        if "Volume" in df.columns:
            df = df.with_columns(pl.col("Volume").cast(pl.Float64))
        if "date" in df.columns:
            df = df.with_columns(pl.col("date").str.strptime(pl.Date(), "%Y-%m-%d"))
        elif "datetime" in df.columns:
            # Parse datetime and keep in native UTC format (no timezone conversion)
            df = df.with_columns(
                pl.col("datetime").str.strptime(pl.Datetime("ns"), "%Y-%m-%d %H:%M:%S")
            )
        else:
            print("Warning: date or datetime column not present")

        return df
