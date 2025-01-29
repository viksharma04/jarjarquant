import asyncio
from ib_async import IB, Stock, util


async def main():
    # Create an IB instance
    ib = IB()

    # Connect to the IB Gateway or TWS
    await ib.connectAsync('127.0.0.1', 7496, clientId=1)

    # Define the AAPL stock contract
    contract = Stock('AAPL', 'SMART', 'USD')

    # Request historical implied volatility data
    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime='',
        durationStr='1 D',  # Duration of data, e.g., '1 M' for 1 month
        barSizeSetting='2 mins',  # Bar size, e.g., '1 day'
        whatToShow='OPTION_IMPLIED_VOLATILITY',
        useRTH=True,
        formatDate=1
    )

    # Convert bars to a DataFrame and display
    df = util.df(bars)
    print(df)

    # Disconnect from IB
    ib.disconnect()

# Run the main function
if __name__ == '__main__':
    asyncio.run(main())
