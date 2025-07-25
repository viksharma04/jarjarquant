import os

from dotenv import load_dotenv

load_dotenv()

# pull keys from env once
EODHD_API_KEY = os.getenv("EODHD_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# data source api keys
DATA_SOURCE_CONFIG = {
    "eodhd": {"api_key": EODHD_API_KEY},
    "alphavantage": {"api_key": ALPHA_VANTAGE_API_KEY},
}
