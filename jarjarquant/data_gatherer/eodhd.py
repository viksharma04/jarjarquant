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
        if not api_key:
            raise ValueError("EODHD API key not configured.")
        self.api_key = api_key

    async def fetch(
        self,
        ticker: str = "SPY",
        bar_size: BarSize = BarSize.ONE_DAY,
        duration: Duration = Duration.ONE_MONTH,
        end_date: Optional[str] = None,
        security_type: str = "STK",
        **kwargs,
    ) -> pl.DataFrame:
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
                r = httpx.get(url)
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
                r = httpx.get(url)
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
        if "date" in df.columns:
            df = df.with_columns(pl.col("date").str.strptime(pl.Date(), "%Y-%m-%d"))
        elif "datetime" in df.columns:
            # Convert index to datetime and localize to UTC, then convert to US/Eastern
            df = df.with_columns(
                pl.col("datetime")
                .str.strptime(pl.Datetime("ns"), "%Y-%m-%d %H:%M:%S")
                .dt.convert_time_zone("US/Eastern")
            )
        else:
            print("Warning: date or datetime column not present")

        return df
