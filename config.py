"""
config.py — Central configuration for the Commodity Price Prediction Engine.

Focused on vegetable price prediction in the Indian market.
"""

import os
from datetime import datetime, timedelta

# ─────────────────────────── API Configuration ───────────────────────────
API_BASE_URL = "https://api.agmarknet.gov.in/v1"
FILTERS_ENDPOINT = f"{API_BASE_URL}/daily-price-arrival/filters"
REPORT_ENDPOINT = f"{API_BASE_URL}/daily-price-arrival/report"

# ─────────────────────────── Web Portal URLs ─────────────────────────────
PORTAL_HOME_URL = "https://www.agmarknet.gov.in/home"
PRICE_REPORT_URL = "https://www.agmarknet.gov.in/pricearrivalreportlist"

# ─────────────────────────── Request Settings ────────────────────────────
REQUEST_TIMEOUT = 60  # seconds
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.agmarknet.gov.in/",
    "Origin": "https://www.agmarknet.gov.in",
}
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries
PAGE_SIZE = 500  # records per API page

# ─────────────────────────── Selenium Settings ───────────────────────────
SELENIUM_HEADLESS = True
SELENIUM_IMPLICIT_WAIT = 10  # seconds
SELENIUM_PAGE_LOAD_TIMEOUT = 30  # seconds

# ─────────────────────────── Default Query Parameters ────────────────────
DEFAULT_DAYS_BACK = 365  # fetch 1 year of data by default
DEFAULT_TO_DATE = datetime.now().strftime("%Y-%m-%d")

# Agmarknet 2.0 went live on Nov 7, 2025.
# "Both" (Price+Arrival) data only available from this date onward.
# "Price" or "Arrival" individually are available from 2021-01-01.
AGMARKNET_V2_GO_LIVE = "2025-11-07"
DEFAULT_FROM_DATE = max(
    (datetime.now() - timedelta(days=DEFAULT_DAYS_BACK)).strftime("%Y-%m-%d"),
    "2021-01-01",
)

# Mandatory API parameters with "All" defaults
DEFAULT_DATA_TYPE = 100004     # "Price" (100004=Price, 100005=Arrival, 100006=Both)
DEFAULT_DISTRICT_ID = 100001   # "All Districts"
DEFAULT_MARKET_ID = 100002     # "All Markets"
DEFAULT_GRADE_ID = 100003      # "All Grades"
DEFAULT_VARIETY_ID = 100007    # "All Varieties"

# ─────────────────────────── Vegetable Presets ───────────────────────────
# Curated list of commonly traded vegetables in Indian wholesale markets.
# These are the vegetables available on Agmarknet with high data availability.
VEGETABLES = [
    "Tomato",
    "Onion",
    "Potato",
    "Green Chilli",
    "Cabbage",
    "Cauliflower",
    "Brinjal",
    "Lady Finger(Bhindi)",
    "Capsicum",
    "Bottle Gourd",
    "Bitter Gourd",
    "Cucumber",
    "Pumpkin",
    "Radish",
    "Carrot",
    "Beans",
    "Peas",
    "Spinach",
    "Methi(Fenugreek Leaves)",
    "Coriander(Leaves)",
    "Ginger(Green)",
    "Garlic",
    "Drumstick",
    "Ridge Gourd(Tori)",
    "Pointed Gourd(Parval)",
    "Cluster Beans",
    "Ash Gourd",
    "Snake Gourd",
    "Round Gourd",
    "Papaya(Raw)",
]

# Default commodity for the dashboard — a highly volatile vegetable
DEFAULT_COMMODITY = "Tomato"
DEFAULT_COMMODITY_GROUP = "Vegetables"

# Major agricultural states with high mandi data availability
VEGETABLE_STATES = [
    "Maharashtra",
    "Karnataka",
    "Madhya Pradesh",
    "Uttar Pradesh",
    "Gujarat",
    "Rajasthan",
    "Tamil Nadu",
    "Andhra Pradesh",
    "Telangana",
    "West Bengal",
    "Punjab",
    "Haryana",
    "Bihar",
    "Odisha",
    "Chhattisgarh",
    "Jharkhand",
    "Kerala",
    "Himachal Pradesh",
    "Assam",
]
DEFAULT_STATE = "Maharashtra"

# ─────────────────────────── Data Storage ────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────── Feature Engineering ─────────────────────────
LAG_PERIODS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 30]

# ─────────────────────────── Model Settings ──────────────────────────────
FORECAST_HORIZON = 7  # days to forecast
N_OPTUNA_TRIALS = 50  # hyperparameter search trials
TIME_SERIES_SPLITS = 5  # for walk-forward validation

# ─────────────────────────── Logging ─────────────────────────────────────
LOG_LEVEL = "INFO"
