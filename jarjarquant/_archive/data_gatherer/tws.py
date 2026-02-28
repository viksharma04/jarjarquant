from datetime import datetime

import polars as pl
from ib_async import IB, Contract, Forex, Index, Stock, util

from .base import DataSource, register_data_source
from .utils import BarSize, Duration


@register_data_source("tws")
class TWSDataSource(DataSource):
    async def fetch(
        self,
        ticker: str = "",
        exchange: str = "SMART",
        currency: str = "USD",
        end_date: str = "",
        duration: Duration = Duration.ONE_MONTH,
        bar_size: BarSize = BarSize.ONE_DAY,
        what_to_show="TRADES",
        security_type="STK",
        client_id: int = 1,
        **kwargs,
    ) -> pl.DataFrame:
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
            client_id (int): The client ID to use for TWS connection (default: 1).
            **kwags: Additional keyword arguments.
        Returns:
            polars.DataFrame: DataFrame containing the historical data with columns renamed to standard format.
        Raises:
            RuntimeError: If data cannot be fetched from TWS.
        """
        ib = IB()

        # Connect to the IB Gateway or TWS
        try:
            await ib.connectAsync("127.0.0.1", 7497, clientId=client_id)
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
            # Use UTC to avoid timezone conversion issues
            end_date = dt.strftime("%Y%m%d %H:%M:%S") + " UTC"

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
        except Exception:
            ib.disconnect()  # Disconnect to make sure next call works
            df = None

        # Convert to Polars and rename columns to standard format
        if df is not None and not df.empty:
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

            # Convert to Polars DataFrame first
            pl_df = pl.from_pandas(df)

            # Handle date/datetime column based on bar size
            if bar_size == BarSize.ONE_DAY:
                # Daily data should have 'date' column (cast datetime to date)
                pl_df = pl_df.with_columns(pl.col("date").dt.date().alias("date"))
            else:
                # Intraday data should have 'datetime' column
                pl_df = pl_df.rename({"date": "datetime"})
                # Ensure timezone-aware UTC
                try:
                    pl_df = pl_df.with_columns(
                        pl.col("datetime").dt.convert_time_zone("UTC").alias("datetime")
                    )
                except Exception:
                    pl_df = pl_df.with_columns(
                        pl.col("datetime").dt.replace_time_zone("UTC").alias("datetime")
                    )

            return pl_df

        return pl.DataFrame()
