from datetime import datetime

import pandas as pd
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
        except Exception:
            ib.disconnect()  # Disconnect to make sure next call works
            df = None

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
