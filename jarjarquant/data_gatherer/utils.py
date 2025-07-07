"""This module provides utility classes and functions for the jarjarquant package."""

from datetime import datetime
from enum import StrEnum
from typing import List

import pytz


class BarSize(StrEnum):
    """A string enum for the bar size parameter in the TWS API."""

    ONE_SECOND = "1 sec"
    FIVE_SECONDS = "5 secs"
    TEN_SECONDS = "10 secs"
    FIFTEEN_SECONDS = "15 secs"
    THIRTY_SECONDS = "30 secs"
    ONE_MINUTE = "1 min"
    TWO_MINUTES = "2 mins"
    THREE_MINUTES = "3 mins"
    FIVE_MINUTES = "5 mins"
    TEN_MINUTES = "10 mins"
    FIFTEEN_MINUTES = "15 mins"
    TWENTY_MINUTES = "20 mins"
    THIRTY_MINUTES = "30 mins"
    ONE_HOUR = "1 hour"
    TWO_HOURS = "2 hours"
    THREE_HOURS = "3 hours"
    FOUR_HOURS = "4 hours"
    EIGHT_HOURS = "8 hours"
    ONE_DAY = "1 day"
    ONE_WEEK = "1 week"
    ONE_MONTH = "1 month"


class Duration(StrEnum):
    """A string enum for the duration parameter in the TWS API."""

    ONE_DAY = "1 D"
    TWO_DAYS = "2 D"
    ONE_WEEK = "1 W"
    ONE_MONTH = "1 M"
    TWO_MONTHS = "2 M"
    THREE_MONTHS = "3 M"
    SIX_MONTHS = "6 M"
    ONE_YEAR = "1 Y"
    TWO_YEARS = "2 Y"
    THREE_YEARS = "3 Y"
    FOUR_YEARS = "4 Y"
    FIVE_YEARS = "5 Y"
    TEN_YEARS = "10 Y"
    FIFTY_YEARS = "50 Y"
    HUNDRED_YEARS = "100 Y"
    MAX = "MAX"


BAR_SIZE_TO_STR_MAP = {
    "eodhd": {
        BarSize.ONE_MINUTE: "1m",
        BarSize.FIVE_MINUTES: "5m",
        BarSize.ONE_HOUR: "1h",
        BarSize.ONE_DAY: "d",
        BarSize.ONE_WEEK: "w",
        BarSize.ONE_MONTH: "m",
    },
    "alphavantage": {
        BarSize.ONE_MINUTE: "1min",
        BarSize.FIVE_MINUTES: "5min",
        BarSize.FIFTEEN_MINUTES: "15min",
        BarSize.THIRTY_MINUTES: "30min",
        BarSize.ONE_HOUR: "60min",
        BarSize.ONE_DAY: "",
        BarSize.ONE_WEEK: "",
        BarSize.ONE_MONTH: "",
    },
}

DURATION_TO_DAYS_MAP = {
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


def convert_date_to_unixtime(start_date: str, end_date: str) -> List[int]:
    # Convert start and end dates to UNIX timestamps at 12:00 am Eastern Time
    from_dt = datetime.strptime(start_date, "%Y-%m-%d")
    to_dt = datetime.strptime(end_date, "%Y-%m-%d")
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

    return [from_unix_time, to_unix_time]
