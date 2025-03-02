"""The data gatherer minimizes code duplication and simplifies gathering samples for experimentation."""
import asyncio
import concurrent.futures
import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from ib_async import IB, Stock, util


class DataGatherer:

    def __init__(self):

        self.series = []
        ib = IB()
        if ib.isConnected():
            ib.disconnect()

    @staticmethod
    def get_random_tickers(num_tickers: int = 30, options: Optional[list] = None):

        # Use a default list of tickers if none provided
        if options is None:
            options = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'KO', 'MCD', 'NVDA', 'JPM', 'DIS',
                'BAC', 'CVX', 'INTC', 'CSCO', 'PEP', 'WMT', 'PG', 'ADBE', 'PFE',
                'XOM', 'T', 'NKE', 'MRK', 'IBM', 'HON', 'BA', 'MMM', 'UNH',
                'GS', 'LMT', 'ABT', 'MO', 'AXP', 'CL', 'COP', 'DOW', 'GE', 'HD',
                'JNJ', 'TRV', 'VZ', 'WFC'
            ]

        return random.sample(options, num_tickers)

    def generate_random_normal(self, loc: float = 0.005, volatility: float = 0.05, periods: int = 252, freq: str = 'B', start: Optional[str] = '2100-01-01', name: Optional[str] = 'price', persist: Optional[bool] = False, **kwags):

        date_range = pd.date_range(start=start, periods=periods, freq=freq)
        returns = np.random.normal(
            loc=loc, scale=volatility, size=periods, **kwags)
        prices = 100 * (1 + returns).cumprod()

        if persist:
            self.series.append(pd.DataFrame(prices, index=date_range))

        return pd.Series(prices, index=date_range, name=name)

    @staticmethod
    def get_yf_ticker(ticker: str = "SPY", **kwags):

        ticker = yf.Ticker(ticker)
        series = ticker.history(
            auto_adjust=True, **kwags)[['Open', 'High', 'Low', 'Close', 'Volume']]

        return series

    @staticmethod
    async def _get_tws_ticker(ticker: str = '', exchange: str = 'SMART', currency: str = 'USD', end_date: str = '', duration: str = '1 M', bar_size: str = '1 day', what_to_show='TRADES', **kwags):

        ib = IB()

        # Connect to the IB Gateway or TWS
        await ib.connectAsync('127.0.0.1', 7496, clientId=1)

        # Define the stock contract
        contract = Stock(ticker, exchange, currency)

        # Request historical implied volatility data
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_date,
            durationStr=duration,  # Duration of data, e.g., '1 M' for 1 month
            barSizeSetting=bar_size,  # Bar size, e.g., '1 day'
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1
        )

        # Convert bars to a DataFrame and display
        df = util.df(bars)
        # Rename 'open', 'high', 'low', 'close' columns to 'Open', 'High', 'Low', 'Close'
        df.rename(columns={'open': 'Open', 'high': 'High',
                  'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

        # Disconnect from IB
        ib.disconnect()

        return df

    def get_tws_ticker(self, ticker: str = '', exchange: str = 'SMART', currency: str = 'USD', end_date: str = '', duration: str = '1 M', bar_size: str = '1 day', what_to_show='TRADES', **kwags):

        return asyncio.run(self._get_tws_ticker(ticker=ticker, exchange=exchange, currency=currency, end_date=end_date, duration=duration, bar_size=bar_size, what_to_show=what_to_show, **kwags))

    def get_random_price_samples_yf(self, years_in_sample: int = 10, resolution: str = '1 day', tickers: Optional[list] = None, num_tickers_to_sample: Optional[int] = 30, persist: Optional[bool] = False):

        # Set today's date (October 23, 2024) and the time range for the last 30 years
        today = datetime.today()
        num_years = timedelta(days=years_in_sample*365)

        # List of 30 stock tickers with replacements for problematic ones
        if tickers is None:
            tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'KO', 'MCD', 'NVDA', 'JPM', 'DIS',
                'BAC', 'CVX', 'INTC', 'CSCO', 'PEP', 'WMT', 'PG', 'ADBE', 'PFE',
                'XOM', 'T', 'NKE', 'MRK', 'ORCL', 'IBM', 'HON', 'BA', 'MMM', 'UNH',
                'GS', 'LMT', 'ABT', 'MO', 'AXP', 'CL', 'COP', 'DOW', 'GE', 'HD',
                'JNJ', 'TRV', 'VZ', 'WFC'
            ]

        tickers = random.sample(tickers, num_tickers_to_sample)

        # Function to get random start and end dates
        def get_random_date(ticker):
            ticker = yf.Ticker(ticker)
            ticker_metadata = ticker.history_metadata

            start_limit = datetime.fromtimestamp(
                np.abs(ticker_metadata['firstTradeDate']))

            # If today - start limit is less than years to sample, then we can't sample that far back
            if int((today - start_limit).days) < years_in_sample*365:
                print(f"WARNING: {ticker} does not have enough data to sample {
                      years_in_sample} years back. Sampling all available data")
                start_date = start_limit
                end_date = today

            else:
                random_start = start_limit + \
                    timedelta(days=np.random.randint(
                        0, (today - start_limit).days - years_in_sample*365))
                random_end = random_start + num_years
                start_date = random_start
                end_date = random_end

            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

        # List to store the DataFrames
        dataframes = []

        # Loop over each ticker and fetch 2 years of data with random start dates
        for ticker in tickers:
            start_date, end_date = get_random_date(ticker)
            df = self.get_yf_ticker(ticker, start=start_date, end=end_date)
            df.index = df.index.tz_localize(None)
            dataframes.append(df)

        if persist:
            self.series.append(dataframes)

        return dataframes

    # async def _get_random_price_samples_tws(self,
    #                                         years_in_sample: int = 10,
    #                                         tickers: Optional[list] = None,
    #                                         num_tickers_to_sample: Optional[int] = 30,
    #                                         persist: Optional[bool] = False,
    #                                         bar_size: Optional[str] = '1 day',
    #                                         duration: Optional[str] = None):
    #     # Set today's date and compute the time delta
    #     today = datetime.today()
    #     num_years = timedelta(days=years_in_sample * 365)

    #     # Use a default list of tickers if none provided
    #     if tickers is None:
    #         tickers = [
    #             'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'KO', 'MCD', 'NVDA', 'JPM', 'DIS',
    #             'BAC', 'CVX', 'INTC', 'CSCO', 'PEP', 'WMT', 'PG', 'ADBE', 'PFE',
    #             'XOM', 'T', 'NKE', 'MRK', 'IBM', 'HON', 'BA', 'MMM', 'UNH',
    #             'GS', 'LMT', 'ABT', 'MO', 'AXP', 'CL', 'COP', 'DOW', 'GE', 'HD',
    #             'JNJ', 'TRV', 'VZ', 'WFC'
    #         ]
    #     tickers = random.sample(tickers, num_tickers_to_sample)

    #     # Define a synchronous helper function that fetches data for one ticker.
    #     # We create a separate IB connection (with its own clientId) in each thread.
    #     def fetch_data_for_ticker(ticker, client_id, duration=None):
    #         ib_local = IB()
    #         # Connect synchronously (using a unique clientId per thread)
    #         ib_local.connect('127.0.0.1', 7496, clientId=client_id)

    #         contract = Stock(ticker, 'SMART', 'USD')
    #         # Get the start timestamp for the ticker
    #         start_limit = ib_local.reqHeadTimeStamp(
    #             contract=contract, whatToShow='TRADES', useRTH=True)

    #         # Determine the end date for data sampling
    #         if int((today - start_limit).days) < years_in_sample * 365:
    #             print(
    #                 f"WARNING: {ticker} does not have enough data to sample {years_in_sample} years back. Sampling all available data")
    #             end_date = today
    #         else:
    #             random_start = start_limit + timedelta(
    #                 days=np.random.randint(
    #                     0, (today - start_limit).days - years_in_sample * 365)
    #             )
    #             random_end = random_start + num_years
    #             end_date = random_end

    #         if duration is None:
    #             duration = f"{years_in_sample} Y"

    #         bars = ib_local.reqHistoricalData(
    #             contract,
    #             endDateTime=end_date,
    #             durationStr=duration,          # e.g., '10 Y'
    #             barSizeSetting=bar_size,        # Bar size: '1 day'
    #             whatToShow='TRADES',
    #             useRTH=True,
    #             formatDate=1
    #         )

    #         # Convert the bars to a DataFrame
    #         df = util.df(bars)
    #         df.rename(columns={'open': 'Open', 'high': 'High',
    #                            'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    #         ib_local.disconnect()
    #         return df

    #     # Use a ThreadPoolExecutor to run fetch_data_for_ticker concurrently.
    #     util.startLoop()
    #     loop = asyncio.get_running_loop()
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_tickers_to_sample) as executor:
    #         tasks = [
    #             loop.run_in_executor(
    #                 executor, fetch_data_for_ticker, ticker, 2 + i, duration)
    #             for i, ticker in enumerate(tickers)
    #         ]
    #         dataframes = await asyncio.gather(*tasks)

    #     if persist:
    #         self.series.append(dataframes)
    #     return dataframes

    @staticmethod
    async def async_get_tws_ticker(ib, ticker, duration, bar_size, years_in_sample):

        today = datetime.today()
        num_years = timedelta(days=years_in_sample * 365)

        contract = Stock(ticker, 'SMART', 'USD')
        # Get the start timestamp for the ticker
        start_limit = await ib.reqHeadTimeStampAsync(
            contract=contract, whatToShow='TRADES', useRTH=True, formatDate=1)

        # Determine the end date for data sampling
        if int((today - start_limit).days) < years_in_sample * 365:
            print(
                f"WARNING: {ticker} does not have enough data to sample {years_in_sample} years back. Sampling all available data")
            end_date = today
        else:
            random_start = start_limit + timedelta(
                days=np.random.randint(
                    0, (today - start_limit).days - years_in_sample * 365)
            )
            random_end = random_start + num_years
            end_date = random_end

        if duration is None:
            duration = f"{years_in_sample} Y"

        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_date,
            durationStr=duration,          # e.g., '10 Y'
            barSizeSetting=bar_size,        # Bar size: '1 day'
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )

        # Convert the bars to a DataFrame
        df = util.df(bars)
        df.rename(columns={'open': 'Open', 'high': 'High',
                           'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

        return df

    async def _get_random_price_samples_tws(self,
                                            years_in_sample: int = 10,
                                            tickers: Optional[list] = None,
                                            num_tickers_to_sample: Optional[int] = 30,
                                            persist: Optional[bool] = False,
                                            bar_size: Optional[str] = '1 day',
                                            duration: Optional[str] = None,
                                            verbose: Optional[bool] = False):

        ib = IB()
        await ib.connectAsync('127.0.0.1', 7496, clientId=1)

        if tickers is None:
            tickers = self.get_random_tickers(
                num_tickers_to_sample, options=None)
        else:
            tickers = random.sample(tickers, num_tickers_to_sample)
        # tasks = [self.async_get_tws_ticker(
        #     ib, ticker, duration, bar_size, years_in_sample) for ticker in tickers]
        # results = await asyncio.gather(*tasks)

        results = []
        for ticker in tickers:
            try:
                result = await self.async_get_tws_ticker(
                    ib, ticker, duration, bar_size, years_in_sample)
                if verbose:
                    results.append({ticker: result})
                else:
                    results.append(result)
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                continue

        ib.disconnect()

        return results

    def get_random_price_samples_tws(self,
                                     years_in_sample: int = 10,
                                     tickers: Optional[list] = None,
                                     num_tickers_to_sample: Optional[int] = 30,
                                     persist: Optional[bool] = False,
                                     bar_size: Optional[str] = '1 day',
                                     duration: Optional[str] = None,
                                     verbose: Optional[bool] = False):

        return asyncio.run(self._get_random_price_samples_tws(
            years_in_sample=years_in_sample,
            tickers=tickers,
            num_tickers_to_sample=num_tickers_to_sample,
            persist=persist,
            bar_size=bar_size,
            duration=duration,
            verbose=verbose
        ))
