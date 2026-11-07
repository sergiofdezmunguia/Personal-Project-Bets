import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Season 2025/26 onwards: final test and live simulation. Not for development.
FINAL_TEST_START = pd.Timestamp("2025-07-01")

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

BESOCCER_API_KEY = os.getenv("BESOCCER_API_KEY", "")
BESOCCER_BASE_URL = os.getenv("BESOCCER_BASE_URL", "https://apiclient.besoccerapi.com")
