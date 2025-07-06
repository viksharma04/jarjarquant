"""The data gatherer minimizes code duplication and simplifies gathering samples for experimentation."""

import asyncio
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import polars as pl
import pytz
import yfinance as yf
from dotenv import load_dotenv
from eodhd import APIClient
from ib_async import IB, Contract, Forex, Index, Stock, util

from jarjarquant.utilities import BarSize, Duration

load_dotenv()
logger = logging.getLogger(__name__)


class DataGatherer:
    def __init__(
        self,
        eodhd_api_key: Optional[str] = None,
        alpha_vantage_api_key: Optional[str] = None,
    ):
        self.data = []
        self.eodhd_api_key = os.environ.get("EODHD_API_KEY", eodhd_api_key)
        self.alpha_vantage_api_key = os.environ.get(
            "ALPHA_VANTAGE_API_KEY", alpha_vantage_api_key
        )
        ib = IB()
        if ib.isConnected():
            ib.disconnect()

    @staticmethod
    def get_random_tickers(num_tickers: int = 30, options: Optional[list] = None):
        # Use a default list of tickers if none provided
        if options is None:
            options = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "KO",
                "MCD",
                "NVDA",
                "JPM",
                "DIS",
                "BAC",
                "CVX",
                "INTC",
                "CSCO",
                "PEP",
                "WMT",
                "PG",
                "ADBE",
                "PFE",
                "XOM",
                "T",
                "NKE",
                "MRK",
                "IBM",
                "HON",
                "BA",
                "MMM",
                "UNH",
                "GS",
                "LMT",
                "ABT",
                "MO",
                "AXP",
                "CL",
                "COP",
                "DOW",
                "GE",
                "HD",
                "JNJ",
                "TRV",
                "VZ",
                "WFC",
            ]

        return random.sample(options, num_tickers)

    def generate_random_normal(
        self,
        loc: float = 0.005,
        volatility: float = 0.05,
        periods: int = 252,
        freq: str = "B",
        start: Optional[str] = "2100-01-01",
        name: Optional[str] = "price",
        persist: Optional[bool] = False,
        **kwags,
    ) -> pd.DataFrame:
        date_range = pd.date_range(start=start, periods=periods, freq=freq)
        returns = np.random.normal(loc=loc, scale=volatility, size=periods, **kwags)
        prices = 100 * (1 + returns).cumprod()
        df = pd.DataFrame({name: prices}, index=date_range)

        if persist:
            self.data.append(df)

        return df

    def get_custom_sample(
        self,
        sample_name: str,
        return_metadata: Optional[bool] = False,
        persist: Optional[bool] = False,
    ):
        # Read all csv files in the data/sample_name folder and return a list of DataFrames

        folder_path = os.path.join(
            os.path.dirname(__file__), "ticker_samples", "data", sample_name
        )
        data = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)

                df = pd.read_csv(
                    file_path,
                    index_col=0,
                    parse_dates=["date"],
                    date_format="%Y-%m-%d %H:%M:%S%z",
                )
                df.set_index("date", inplace=True)
                df.index = pd.to_datetime(df.index, utc=True)
                df.index = df.index.tz_localize(None)
                if return_metadata:
                    metadata = file_name.split("_")
                    data.append(
                        {"market_cap": metadata[0], "sector": metadata[1], "df": df}
                    )
                else:
                    data.append(df)

        if persist:
            self.data.append(data)

        return data

    @staticmethod
    def get_yf_ticker(ticker: str = "SPY", **kwags):
        ticker = yf.Ticker(ticker)
        series = ticker.history(auto_adjust=True, **kwags)[
            ["Open", "High", "Low", "Close", "Volume"]
        ]

        return series

    async def get_eodhd_ticker(
        self,
        ticker: str = "SPY",
        bar_size: BarSize = BarSize.ONE_DAY,
        duration: Duration = Duration.ONE_MONTH,
        end: Optional[str] = None,
        security_type: str = "STK",
        **kwags,
    ) -> pl.DataFrame:
        if self.eodhd_api_key is None:
            raise ValueError("EODHD API key not provided.")

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        if security_type == "STK":
            ticker = ticker + ".US"

        period_map = {
            BarSize.ONE_MINUTE: "1m",
            BarSize.FIVE_MINUTES: "5m",
            BarSize.ONE_HOUR: "1h",
            BarSize.ONE_DAY: "d",
            BarSize.ONE_WEEK: "w",
            BarSize.ONE_MONTH: "m",
        }
        if bar_size not in list(period_map.keys()):
            raise ValueError(
                "bar_size can only be 1/5 min, 1 hour, day, week, or month"
            )
        eodhd_period = period_map.get(bar_size, "d")

        if duration is not Duration.MAX:
            # Convert duration to days (simple approximation)
            duration_days_map = {
                Duration.ONE_DAY: 1,
                Duration.ONE_WEEK: 7,
                Duration.ONE_MONTH: 30,
                Duration.TWO_MONTHS: 60,
                Duration.THREE_MONTHS: 90,
                Duration.SIX_MONTHS: 180,
                Duration.ONE_YEAR: 365,
                Duration.FIVE_YEARS: 1825,
                Duration.TEN_YEARS: 3650,
            }
            if duration not in list(duration_days_map.keys()):
                raise ValueError("Invalid duration")
            duration_days = duration_days_map.get(duration, 30)

            end_dt = datetime.strptime(end, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=duration_days)
            start = start_dt.strftime("%Y-%m-%d")
        else:
            start = None
            end = None

        api = APIClient(self.eodhd_api_key)

        if bar_size not in [BarSize.ONE_MINUTE, BarSize.FIVE_MINUTES, BarSize.ONE_HOUR]:
            series = api.get_eod_historical_stock_market_data(
                symbol=ticker,
                from_date=start,
                to_date=end,
                period=eodhd_period,
                order="a",
                **kwags,
            )
        else:
            if start is not None:
                # Convert start and end dates to UNIX timestamps at 12:00 am Eastern Time
                from_dt = datetime.strptime(start, "%Y-%m-%d")
                to_dt = datetime.strptime(end, "%Y-%m-%d")
                # Set time to 12:00 am and localize to US/Eastern, then convert to UTC
                eastern = pytz.timezone("US/Eastern")
                from_dt = eastern.localize(
                    from_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                ).astimezone(pytz.UTC)
                to_dt = eastern.localize(
                    to_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                ).astimezone(pytz.UTC)
                from_unix_time = int(from_dt.timestamp())
                to_unix_time = int(to_dt.timestamp())
            else:
                from_unix_time = None
                to_unix_time = None
            try:
                series = api.get_intraday_historical_data(
                    symbol=ticker,
                    from_unix_time=from_unix_time,
                    to_unix_time=to_unix_time,
                    interval=eodhd_period,
                    **kwags,
                )

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
            df = df.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d"))
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

    # TODO: Support end dates
    def get_alpha_vantage_ticker(
        self,
        ticker: str = "SPY",
        end_date: Optional[str] = None,
        duration: Duration = Duration.ONE_MONTH,
        bar_size: BarSize = BarSize.ONE_DAY,
    ):
        """
        Fetches historical price data for a given ticker from Alpha Vantage.
        Parameters:
            ticker (str): The ticker symbol to fetch data for. Defaults to "SPY".
            end_date (Optional[str]): The end date for the data in "YYYY-MM-DD" format. Defaults to today if not provided.
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
        if self.alpha_vantage_api_key is None:
            raise ValueError("Alpha Vantage API key is not provided!")

        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        period_map = {
            BarSize.ONE_MINUTE: "1min",
            BarSize.FIVE_MINUTES: "5min",
            BarSize.FIFTEEN_MINUTES: "15min",
            BarSize.THIRTY_MINUTES: "30min",
            BarSize.ONE_HOUR: "60min",
            BarSize.ONE_DAY: "",
            BarSize.ONE_WEEK: "",
            BarSize.ONE_MONTH: "",
        }
        if bar_size not in list(period_map.keys()):
            raise ValueError("invalid bar size")
        intraday_interval = period_map.get(bar_size, "")

        # Convert duration to days (simple approximation)
        duration_days_map = {
            Duration.ONE_DAY: 1,
            Duration.ONE_WEEK: 7,
            Duration.ONE_MONTH: 30,
            Duration.THREE_MONTHS: 90,
            Duration.SIX_MONTHS: 180,
            Duration.ONE_YEAR: 365,
            Duration.TEN_YEARS: 3650,
        }
        if duration not in list(duration_days_map.keys()):
            raise ValueError("Invalid duration")
        duration_days = duration_days_map.get(duration, 30)

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=duration_days)
        start_date = start_dt.strftime("%Y-%m-%d")

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
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={intraday_interval}&apikey={self.alpha_vantage_api_key}"
            try:
                r = httpx.get(url)
                result = r.json()
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
            except Exception as e:
                print(f"Warning: {e}")
                return pl.DataFrame()
        else:
            function_map = {
                BarSize.ONE_DAY: "TIME_SERIES_DAILY_ADJUSTED",
                BarSize.ONE_WEEK: "TIME_SERIES_WEEKLY_ADJUSTED",
                BarSize.ONE_MONTH: "TIME_SERIES_MONTHLY_ADJUSTED",
            }
            time_series_function = function_map[bar_size]
            url = f"https://www.alphavantage.co/query?function={time_series_function}&symbol={ticker}&apikey={self.alpha_vantage_api_key}"
            try:
                r = httpx.get(url)
                result = r.json()
                time_series_column_map = {
                    BarSize.ONE_DAY: "Time Series (Daily)",
                    BarSize.ONE_WEEK: "Weekly Adjusted Time Series",
                    BarSize.ONE_MONTH: "Monthly Adjusted Time Series",
                }
                data = result[f"{time_series_column_map[bar_size]}"]
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
            except Exception as e:
                print(f"Warning: {e}")
                return pl.DataFrame()

        if not r:
            print("Warning: No data fetched for request")
            return pl.DataFrame()

        return df

    @staticmethod
    async def _get_tws_ticker(
        ticker: str = "",
        exchange: str = "SMART",
        currency: str = "USD",
        end_date: str = "",
        duration: Duration = Duration.ONE_MONTH,
        bar_size: BarSize = BarSize.ONE_DAY,
        what_to_show="TRADES",
        security_type="STK",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Asynchronously fetches historical market data from Interactive Brokers TWS or Gateway.
        Parameters:
            ticker (str): The ticker symbol of the security (default: "").
            exchange (str): The exchange to use (default: "SMART").
            currency (str): The currency of the security (default: "USD").
            end_date (str): The end date/time for the data request format "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" (default: "").
            duration (str): The duration of data to fetch, e.g., '1 M' for 1 month (default: "1 M").
            bar_size (str): The size of each bar, e.g., '1 day' (default: "1 day").
            what_to_show (str): The type of data to show, e.g., "TRADES" (default: "TRADES").
            security_type (str): The type of security, e.g., "STK" for stock (default: "STK").
            **kwags: Additional keyword arguments.
        Returns:
            pandas.DataFrame: DataFrame containing the historical data with columns renamed to standard format.
        Raises:
            RuntimeError: If data cannot be fetched from TWS.
        """
        ib = IB()

        # Connect to the IB Gateway or TWS
        try:
            await ib.connectAsync("127.0.0.1", 7496, clientId=1)
        except Exception as e:
            raise RuntimeError(f"TWSError: Cannot establish TWS connection: {e}")

        # Define the stock contract
        if security_type == "STK":
            contract = Stock(ticker, exchange, currency)
        elif security_type == "IDX":
            exchange = "CBOE" if exchange == "SMART" else exchange
            contract = Index(ticker, exchange, currency)
        elif security_type == "FX":
            exchange = "IDEALPRO"
            contract = Forex(ticker, exchange, currency)
        else:
            contract = Contract(security_type, 33887599, ticker, exchange=exchange)

        if end_date.strip():
            try:
                dt = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                # Try alternate format if the first one fails
                dt = datetime.strptime(end_date, "%Y%m%d %H:%M:%S")

            # Format date according to IB API requirements: yyyymmdd hh:mm:ss TZ
            end_date = dt.strftime("%Y%m%d %H:%M:%S") + " US/Eastern"

        # Request historical implied volatility data
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_date,
                durationStr=duration,  # Duration of data, e.g., '1 M' for 1 month
                barSizeSetting=bar_size,  # Bar size, e.g., '1 day'
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1,
            )
            # Convert bars to a DataFrame and display
            df = util.df(bars)
            # Disconnect from IB
            ib.disconnect()
        except Exception as e:
            ib.disconnect()  # Disconnect to make sure next call works
            df = None
            logger.warning(f"TWSError: Cannot fetch data from TWS: {e}")

        # Rename 'open', 'high', 'low', 'close' columns to 'Open', 'High', 'Low', 'Close'
        if df is not None:
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
            df.index = pd.DatetimeIndex(df["date"])
            df.drop(columns=["date"], inplace=True)

            return df

        return pd.DataFrame()

    def get_tws_ticker(
        self,
        ticker: str = "",
        exchange: str = "SMART",
        currency: str = "USD",
        end_date: str = "",
        duration: Duration = Duration.ONE_MONTH,
        bar_size: BarSize = BarSize.ONE_DAY,
        what_to_show="TRADES",
        security_type="STK",
        **kwags,
    ) -> pd.DataFrame:
        """
        Synchronous wrapper to fetch historical market data from Interactive Brokers TWS or Gateway.
        Parameters:
            ticker (str): The ticker symbol of the security (default: "").
            exchange (str): The exchange to use (default: "SMART").
            currency (str): The currency of the security (default: "USD").
            end_date (str): The end date/time for the data request format "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" (default: "").
            duration (str): The duration of data to fetch, e.g., '1 M' for 1 month (default: "1 M").
            bar_size (str): The size of each bar, e.g., '1 day' (default: "1 day").
            what_to_show (str): The type of data to show, e.g., "TRADES" (default: "TRADES").
            security_type (str): The type of security, e.g., "STK" for stock (default: "STK").
            **kwags: Additional keyword arguments.
        Returns:
            pandas.DataFrame: DataFrame containing the historical data with columns renamed to standard format.
        Raises:
            RuntimeError: If data cannot be fetched from TWS.
        """
        return asyncio.run(
            self._get_tws_ticker(
                ticker=ticker,
                exchange=exchange,
                currency=currency,
                end_date=end_date,
                duration=duration,
                bar_size=bar_size,
                what_to_show=what_to_show,
                security_type=security_type,
                **kwags,
            )
        )

    def get_random_price_samples_yf(
        self,
        years_in_sample: int = 10,
        resolution: str = "1 day",
        tickers: Optional[list] = None,
        num_tickers_to_sample: Optional[int] = 30,
        persist: Optional[bool] = False,
    ):
        # Set today's date (October 23, 2024) and the time range for the last 30 years
        today = datetime.today()
        num_years = timedelta(days=years_in_sample * 365)

        # List of 30 stock tickers with replacements for problematic ones
        if tickers is None:
            tickers = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "KO",
                "MCD",
                "NVDA",
                "JPM",
                "DIS",
                "BAC",
                "CVX",
                "INTC",
                "CSCO",
                "PEP",
                "WMT",
                "PG",
                "ADBE",
                "PFE",
                "XOM",
                "T",
                "NKE",
                "MRK",
                "ORCL",
                "IBM",
                "HON",
                "BA",
                "MMM",
                "UNH",
                "GS",
                "LMT",
                "ABT",
                "MO",
                "AXP",
                "CL",
                "COP",
                "DOW",
                "GE",
                "HD",
                "JNJ",
                "TRV",
                "VZ",
                "WFC",
            ]

        tickers = random.sample(tickers, num_tickers_to_sample)

        # Function to get random start and end dates
        def get_random_date(ticker):
            ticker = yf.Ticker(ticker)
            ticker_metadata = ticker.history_metadata

            start_limit = datetime.fromtimestamp(
                np.abs(ticker_metadata["firstTradeDate"])
            )

            # If today - start limit is less than years to sample, then we can't sample that far back
            if int((today - start_limit).days) < years_in_sample * 365:
                print(
                    f"WARNING: {ticker} does not have enough data to sample {
                        years_in_sample
                    } years back. Sampling all available data"
                )
                start_date = start_limit
                end_date = today

            else:
                random_start = start_limit + timedelta(
                    days=np.random.randint(
                        0, (today - start_limit).days - years_in_sample * 365
                    )
                )
                random_end = random_start + num_years
                start_date = random_start
                end_date = random_end

            return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

        # List to store the DataFrames
        dataframes = []

        # Loop over each ticker and fetch 2 years of data with random start dates
        for ticker in tickers:
            start_date, end_date = get_random_date(ticker)
            df = self.get_yf_ticker(ticker, start=start_date, end=end_date)
            df.index = df.index.tz_localize(None)
            dataframes.append(df)

        if persist:
            self.data.append(dataframes)

        return dataframes

    @staticmethod
    async def async_get_tws_ticker(ib, ticker, duration, bar_size, years_in_sample):
        today = datetime.today()
        num_years = timedelta(days=years_in_sample * 365)

        contract = Stock(ticker, "SMART", "USD")
        # Get the start timestamp for the ticker
        start_limit = await ib.reqHeadTimeStampAsync(
            contract=contract, whatToShow="TRADES", useRTH=True, formatDate=1
        )

        # Determine the end date for data sampling
        if int((today - start_limit).days) < years_in_sample * 365:
            print(
                f"WARNING: {ticker} does not have enough data to sample {years_in_sample} years back. Sampling all available data"
            )
            end_date = today
        else:
            random_start = start_limit + timedelta(
                days=np.random.randint(
                    0, (today - start_limit).days - years_in_sample * 365
                )
            )
            random_end = random_start + num_years
            end_date = random_end

        if duration is None:
            duration = f"{years_in_sample} Y"

        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_date,
            durationStr=duration,  # e.g., '10 Y'
            barSizeSetting=bar_size,  # Bar size: '1 day'
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        # Convert the bars to a DataFrame
        df = util.df(bars)
        if df is not None:
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
            df.index = pd.DatetimeIndex(df["date"])
            df.drop(columns=["date"], inplace=True)

        return df

    async def _get_random_price_samples_tws(
        self,
        years_in_sample: int = 1,
        tickers: Optional[list] = None,
        num_tickers_to_sample: int = 30,
        persist: Optional[bool] = False,
        bar_size: BarSize = BarSize.ONE_DAY,
        duration: Optional[str] = None,
        verbose: Optional[bool] = False,
    ):
        ib = IB()
        await ib.connectAsync("127.0.0.1", 7496, clientId=1)

        if tickers is None:
            tickers = self.get_random_tickers(num_tickers_to_sample, options=None)
        else:
            tickers = random.sample(tickers, num_tickers_to_sample)
        # tasks = [self.async_get_tws_ticker(
        #     ib, ticker, duration, bar_size, years_in_sample) for ticker in tickers]
        # results = await asyncio.gather(*tasks)

        results = []
        for ticker in tickers:
            try:
                result = await self.async_get_tws_ticker(
                    ib, ticker, duration, bar_size, years_in_sample
                )
                if verbose:
                    results.append({ticker: result})
                else:
                    results.append(result)
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                continue

        ib.disconnect()

        return results

    def get_random_price_samples_tws(
        self,
        years_in_sample: int = 1,
        tickers: Optional[list] = None,
        num_tickers_to_sample: int = 30,
        persist: Optional[bool] = False,
        bar_size: BarSize = BarSize.ONE_DAY,
        duration: Optional[str] = None,
        verbose: Optional[bool] = False,
    ):
        return asyncio.run(
            self._get_random_price_samples_tws(
                years_in_sample=years_in_sample,
                tickers=tickers,
                num_tickers_to_sample=num_tickers_to_sample,
                persist=persist,
                bar_size=bar_size,
                duration=duration,
                verbose=verbose,
            )
        )
