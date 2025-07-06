"""This module provides utility classes and functions for the jarjarquant package."""

from enum import StrEnum


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
