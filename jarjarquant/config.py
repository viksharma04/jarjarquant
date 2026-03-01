import os

from dotenv import load_dotenv

load_dotenv()

# Database path - can be absolute or relative to current working directory
LOCAL_DB_PATH = os.getenv("LOCAL_DB_PATH", "db/sample_data")
