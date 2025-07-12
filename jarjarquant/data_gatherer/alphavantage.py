from typing import Optional

import httpx
import polars as pl

from .base import DataSource, register_data_source
from .utils import BAR_SIZE_TO_STR_MAP, BarSize

API_FUNCTION_MAP = {
    BarSize.ONE_DAY: "TIME_SERIES_DAILY_ADJUSTED",
    BarSize.ONE_WEEK: "TIME_SERIES_WEEKLY_ADJUSTED",
    BarSize.ONE_MONTH: "TIME_SERIES_MONTHLY_ADJUSTED",
}

API_COLUMN_MAP = {
    BarSize.ONE_DAY: "Time Series (Daily)",
    BarSize.ONE_WEEK: "Weekly Adjusted Time Series",
    BarSize.ONE_MONTH: "Monthly Adjusted Time Series",
}


@register_data_source("alphavantage")
class AlphaVantageDataSource(DataSource):
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def fetch(
        self,
        ticker: str = "SPY",
        month: Optional[str] = None,
        bar_size: BarSize = BarSize.ONE_DAY,
    ):
        """
        Fetches historical price data for a given ticker from Alpha Vantage.
        Parameters:
            ticker (str): The ticker symbol to fetch data for. Defaults to "SPY".
            month (Optional[str]): The month for the data in "YYYY-MM" format. Defaults to today if not provided.
            duration (Duration): The duration of data to fetch (e.g., ONE_MONTH, ONE_YEAR). Defaults to Duration.ONE_MONTH.
            bar_size (BarSize): The granularity of the data (e.g., ONE_DAY, ONE_MINUTE). Defaults to BarSize.ONE_DAY.
        Returns:
            pl.DataFrame: A Polars DataFrame containing the historical price data with columns for datetime/date, Open, High, Low, Close, and Volume.
        Raises:
            ValueError: If the Alpha Vantage API key is not provided, or if an invalid bar size or duration is specified.
        Notes:
            - For intraday data (minute/hour bars), only the most recent data (up to 30 days) is available due to Alpha Vantage API limitations.
            - For daily, weekly, or monthly bars, adjusted close and volume are returned.
            - The function prints a warning and returns an empty DataFrame if data fetching fails.
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not configured.")

        period_map = BAR_SIZE_TO_STR_MAP["alphavantage"]
        if bar_size not in list(period_map.keys()):
            raise ValueError("invalid bar size")
        intraday_interval = period_map.get(bar_size, "")

        if bar_size in [
            BarSize.ONE_MINUTE,
            BarSize.FIVE_MINUTES,
            BarSize.FIFTEEN_MINUTES,
            BarSize.THIRTY_MINUTES,
            BarSize.ONE_HOUR,
        ]:
            # Get each month between start date and end date as "YYYY-MM"
            # Call the API parallely to fetch data for each month
            # Concatenate the data into a single dataframe
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={intraday_interval}&apikey={self.api_key}&outputsize=full"
            if month:
                url += f"&month={month}"
            try:
                r = httpx.get(url, timeout=30.0)
                r.raise_for_status()
                result = r.json()
            except Exception as e:
                print(f"Warning: {e}")
                return pl.DataFrame()

            data = result[f"Time Series ({intraday_interval})"]
            df = (
                pl.DataFrame(
                    {
                        "datetime": list(data.keys()),
                        "Open": [float(d["1. open"]) for d in data.values()],
                        "High": [float(d["2. high"]) for d in data.values()],
                        "Low": [float(d["3. low"]) for d in data.values()],
                        "Close": [float(d["4. close"]) for d in data.values()],
                        "Volume": [int(d["5. volume"]) for d in data.values()],
                    }
                )
                .with_columns(
                    pl.col("datetime")
                    .str.strptime(pl.Datetime("ns"), "%Y-%m-%d %H:%M:%S")
                    .dt.convert_time_zone("US/Eastern")
                )
                .sort("datetime")
            )
        else:
            time_series_function = API_FUNCTION_MAP[bar_size]
            url = f"https://www.alphavantage.co/query?function={time_series_function}&symbol={ticker}&apikey={self.api_key}"
            try:
                r = httpx.get(url)
                result = r.json()
            except Exception as e:
                print(f"Warning: {e}")
                return pl.DataFrame()

            data = result[f"{API_COLUMN_MAP[bar_size]}"]
            df = pl.DataFrame(
                {
                    "date": list(data.keys()),
                    "Open": [float(d["1. open"]) for d in data.values()],
                    "High": [float(d["2. high"]) for d in data.values()],
                    "Low": [float(d["3. low"]) for d in data.values()],
                    "Close": [float(d["5. adjusted close"]) for d in data.values()],
                    "Volume": [int(d["6. volume"]) for d in data.values()],
                }
            ).with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

        if not r:
            print("Warning: No data fetched for request")
            return pl.DataFrame()

        return df
