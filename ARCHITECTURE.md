# 🏗️ Architecture — Vegetable Price Prediction Engine

This document describes the system architecture, data flow, and design decisions of the Vegetable Price Prediction Engine.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                        │
│  ┌─────────────────────────┐   ┌─────────────────────────────┐  │
│  │  Streamlit Dashboard    │   │  CLI (main.py --args)       │  │
│  │  (dashboard.py)         │   │                             │  │
│  └────────────┬────────────┘   └──────────────┬──────────────┘  │
│               │                               │                 │
│               └───────────────┬───────────────┘                 │
│                               ▼                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │            PIPELINE ORCHESTRATOR (main.py)                 │  │
│  │   Scrape → Clean → Feature Engineer → Train → Forecast    │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    ▼                          ▼                          ▼
┌──────────┐         ┌─────────────────┐         ┌──────────────┐
│  DATA    │         │   PROCESSING    │         │    MODEL     │
│  LAYER   │         │   LAYER         │         │    LAYER     │
│          │         │                 │         │              │
│ scraper  │────────►│ data_processing │────────►│  model.py    │
│   .py    │  raw    │      .py        │ clean   │              │
│          │  CSV    │                 │  CSV    │  XGBoost +   │
│ Agmarknet│         │ feature_        │         │  Optuna      │
│ 2.0 API  │         │ engineering.py  │         │              │
└──────────┘         └─────────────────┘         └──────────────┘
                                                       │
                                         ┌─────────────┼─────────┐
                                         ▼             ▼         ▼
                                    ┌─────────┐  ┌─────────┐ ┌──────┐
                                    │ Metrics │  │ Forecast│ │Model │
                                    │ (RMSE,  │  │ (n-day  │ │.joblib│
                                    │  MAPE)  │  │  ahead) │ │      │
                                    └─────────┘  └─────────┘ └──────┘
```

---

## Module Details

### 1. `config.py` — Central Configuration

**Role:** Single source of truth for all configurable parameters.

| Section              | Key Parameters                                                    |
|----------------------|-------------------------------------------------------------------|
| API Config           | `API_BASE_URL`, `REPORT_ENDPOINT`, `FILTERS_ENDPOINT`            |
| Request Settings     | `REQUEST_TIMEOUT=60s`, `MAX_RETRIES=3`, `PAGE_SIZE=500`          |
| Vegetable Presets    | `VEGETABLES` (30 items), `VEGETABLE_STATES` (19 states)          |
| Feature Config       | `LAG_PERIODS=[1,7,14,30]`, `ROLLING_WINDOWS=[7,30]`             |
| Model Config         | `FORECAST_HORIZON=7`, `N_OPTUNA_TRIALS=50`, `TIME_SERIES_SPLITS=5` |
| Storage              | `DATA_DIR`, `MODEL_DIR` (auto-created)                           |

---

### 2. `scraper.py` — Agmarknet Data Extraction

**Role:** Fetches wholesale vegetable price data from the Govt. of India's Agmarknet 2.0 portal.

#### API Strategy (Primary)
- **Method:** `POST` with JSON body to `https://api.agmarknet.gov.in/v1/daily-price-arrival/report`
- **Authentication:** None (public API with browser-like headers)
- **Pagination:** Server returns `total_count` and `total_pages`; client iterates with `page` parameter
- **Rate limiting:** 0.5s polite delay between pages; 3 retries with 2s backoff

#### Request Body Structure
```json
{
  "from_date": "2025-04-25",
  "to_date": "2026-04-25",
  "data_type": 100004,
  "commodity": 65,
  "state": [20],
  "district": [100001],
  "market": [100002],
  "grade": [100003],
  "variety": [100007],
  "limit": 500,
  "page": 1,
  "group": 6
}
```

#### Response Structure
```json
{
  "status": true,
  "data": {
    "columns": [...],
    "records": [{
      "data": [
        {
          "cmdt_name": "Tomato",
          "model_price": "3,000.00",
          "min_price": "1,500.00",
          "max_price": "5,000.00",
          "arrival_qty": "2.00",
          "arrival_date": "25-04-2026",
          "market_name": "...",
          "state_name": "...",
          ...
        }
      ],
      "pagination": [{
        "total_count": 10718,
        "total_pages": 22,
        "current_page": 1,
        "items_per_page": 500
      }]
    }]
  }
}
```

#### Filter Resolution
Before fetching data, the scraper calls `/v1/daily-price-arrival/filters` to resolve human-readable names (e.g., "Tomato" → ID 65, "Maharashtra" → ID 20).

#### Selenium Fallback
If the API fails after all retries, falls back to headless Chrome browser automation:
- Uses `webdriver-manager` for automatic ChromeDriver setup
- Navigates the portal's React UI, fills dropdowns, and scrapes result tables
- Handles pagination via "Next" button clicks

#### Column Standardization
Raw API field names are mapped to a standard schema:

| Raw API Field    | Standard Column     |
|------------------|---------------------|
| `model_price`    | `Modal_Price`       |
| `min_price`      | `Min_Price`         |
| `max_price`      | `Max_Price`         |
| `arrival_qty`    | `Arrivals_Tonnes`   |
| `arrival_date`   | `Date`              |
| `cmdt_name`      | `Commodity`         |
| `market_name`    | `Market`            |
| `state_name`     | `State`             |
| `variety_name`   | `Variety`           |

Prices arrive as comma-formatted strings (`"3,000.00"`) and are stripped before numeric conversion.

---

### 3. `data_processing.py` — Data Cleaning Pipeline

**Role:** Transforms raw scraped data into a clean, continuous daily time series.

#### Pipeline Steps (`clean_pipeline()`)

```
Raw DataFrame (1770 rows, multiple markets/varieties)
    │
    ├─ 1. normalize_names()      → Title-case, alias mapping
    ├─ 2. handle_missing_values()→ Forward-fill + interpolate prices, fill arrivals with 0
    ├─ 3. fill_non_trading_days()→ Create continuous daily range per market,
    │                              aggregate duplicates (mean prices, sum arrivals)
    ├─ 4. remove_outliers()      → IQR-based clipping (1.5× factor)
    └─ 5. aggregate_by_date()    → Collapse multi-market data to daily averages
                                   │
                                   ▼
                        Clean DataFrame (~365 rows, daily)
```

#### Key Design Decision: Duplicate Handling
Multiple varieties/grades at the same market on the same date (e.g., "Tomato - Other" and "Tomato - Hybrid") are aggregated per date **before** reindexing to avoid pandas' "cannot reindex on an axis with duplicate labels" error.

---

### 4. `feature_engineering.py` — Feature Generation

**Role:** Creates ~30 predictive features from the cleaned daily price series.

#### Feature Categories

| Category        | Features Created                                    | Count |
|-----------------|-----------------------------------------------------|-------|
| Temporal        | Year, Month, Day, DayOfWeek, Quarter, WeekOfYear, DayOfYear, IsWeekend | 8 |
| Cyclical        | Month_Sin/Cos, DayOfWeek_Sin/Cos                   | 4     |
| Price Lags      | Price_Lag_1/7/14/30                                 | 4     |
| Arrivals Lags   | Arrivals_Lag_1/7                                    | 2     |
| Rolling Stats   | Mean/Std/Min/Max for windows 7 and 30               | 8     |
| Arrivals Rolling| Arrivals_Rolling_Mean_7/30                          | 2     |
| Momentum        | Price_Change_1d/7d, Pct_Change_1d/7d, Spread, Spread_Pct | 6 |
| **Target**      | Target_Price_t+1 (shifted forward)                  | 1     |

#### Adaptive Sizing
For small datasets (< 50 rows), lag periods and rolling windows are **automatically reduced** to preserve enough rows for training:

```python
# Example: 31 rows → only lags [1, 7, 14] kept (30 dropped), windows [7] kept
lags = [l for l in LAG_PERIODS if l < n_rows * 0.6]
windows = [w for w in ROLLING_WINDOWS if w <= n_rows * 0.6]
```

---

### 5. `model.py` — XGBoost Training & Forecasting

**Role:** Trains, tunes, evaluates, and deploys the XGBoost model.

#### Training Pipeline

```
Feature DataFrame (n rows × 30 features)
    │
    ├─ Train/Test Split (80/20, chronological)
    │
    ├─ [Optional] Optuna Hyperparameter Tuning
    │   └─ TPE Sampler, 50 trials
    │   └─ 5-fold TimeSeriesSplit CV
    │   └─ Minimizes average RMSE
    │
    ├─ Final Model Training
    │   └─ XGBRegressor with best params
    │   └─ Early stopping on test set
    │
    ├─ Evaluation
    │   └─ Train/Test RMSE and MAPE
    │   └─ Feature importance extraction
    │
    └─ Persistence
        └─ model.joblib + meta.joblib → models/
```

#### Hyperparameter Search Space (Optuna)

| Parameter          | Range              | Scale |
|--------------------|--------------------|-------|
| learning_rate      | 0.01 – 0.3        | log   |
| max_depth          | 3 – 10             | int   |
| n_estimators       | 100 – 1000         | step=50 |
| subsample          | 0.6 – 1.0          | float |
| colsample_bytree   | 0.6 – 1.0          | float |
| reg_alpha          | 1e-8 – 10.0        | log   |
| reg_lambda         | 1e-8 – 10.0        | log   |
| min_child_weight   | 1 – 10             | int   |
| gamma              | 1e-8 – 1.0         | log   |

#### Iterative Forecasting
For multi-day forecasts, the model predicts one day at a time and feeds each prediction back as input:

```
Day 1: features(historical) → predict price_t+1
Day 2: features(historical + day1_prediction) → predict price_t+2
Day 3: features(historical + day1 + day2) → predict price_t+3
...
```

This approach correctly handles lag and rolling features that depend on recent prices.

---

### 6. `dashboard.py` — Streamlit UI

**Role:** Interactive web interface for the entire pipeline.

#### Dashboard Sections

| Section              | Description                                                  |
|----------------------|--------------------------------------------------------------|
| Sidebar Config       | Vegetable dropdown (30+), State dropdown (19), date range slider, model settings |
| Pipeline Status      | Real-time step-by-step progress (scrape → clean → features → train → forecast) |
| Model Metrics        | Color-coded cards for RMSE, MAPE, training samples, feature count |
| Price Chart          | Plotly line chart with historical prices + forecast + confidence band |
| Forecast Table       | Day-by-day predictions with trend summary                    |
| Feature Importance   | Horizontal bar chart of top 15 features                      |
| Arrivals Chart       | Bar chart of daily market arrivals (supply volume)           |
| Data Explorer        | Expandable raw data table                                    |

#### Error Handling
- Insufficient data after cleaning: shows error asking for wider date range
- Feature engineering produces too few rows: suggests 180+ days
- Model training failure: catches ValueError with actionable message

---

### 7. `main.py` — CLI Orchestrator

**Role:** Command-line entry point that runs the full pipeline sequentially.

```
python main.py --commodity Tomato --state Maharashtra
                    │
                    ▼
        ┌──── Resolve dates ────┐
        │ from_date = today-365 │
        │ to_date = today       │
        └───────────┬───────────┘
                    ▼
        [1/5] Scrape from Agmarknet
        [2/5] Clean & preprocess
        [3/5] Engineer features
        [4/5] Train XGBoost
        [5/5] Generate forecast
                    │
                    ▼
        Output: CSV files + saved model + terminal report
```

---

## Data Flow Diagram

```
Agmarknet 2.0 API                                    User
      │                                                │
      │  POST /v1/daily-price-arrival/report           │
      │  JSON body with commodity/state/dates           │
      │◄───────────────────────────────────────── scraper.py
      │                                                │
      │  JSON response (paginated, 500/page)           │
      │────────────────────────────────────────►        │
      │                                                │
      │                                                ▼
      │                                    ┌─── data/raw.csv ───┐
      │                                    │   1770 records      │
      │                                    │   12 columns        │
      │                                    └────────┬────────────┘
      │                                             │
      │                                    data_processing.py
      │                                             │
      │                                    ┌─── data/clean.csv ──┐
      │                                    │   ~300 rows (daily) │
      │                                    │   6 columns         │
      │                                    └────────┬────────────┘
      │                                             │
      │                                  feature_engineering.py
      │                                             │
      │                                    ┌─── feature_df ──────┐
      │                                    │   ~260 rows         │
      │                                    │   37 columns        │
      │                                    └────────┬────────────┘
      │                                             │
      │                                         model.py
      │                                             │
      │                                    ┌────────┴────────┐
      │                                    ▼                 ▼
      │                           models/model.joblib   forecast.csv
      │                           models/meta.joblib    (7-day ahead)
```

---

## Key Design Decisions

### 1. POST API vs GET
The Agmarknet 2.0 API **only accepts POST** requests with a JSON body for the report endpoint. GET requests return 405 Method Not Allowed. This was discovered through network inspection of the official portal.

### 2. Adaptive Feature Engineering
Rather than a fixed lag-30 that would eliminate all data for small datasets, lags and windows are automatically reduced to fit the available data, ensuring at least ~40% of rows survive for training.

### 3. Per-Market Gap Filling
Non-trading days (weekends, holidays) are filled **per market** before aggregation, ensuring the daily time series is continuous. This is critical for lag features to represent actual temporal distances.

### 4. Iterative Forecasting
Multi-day forecasts use iterative 1-step prediction rather than a single multi-output model. This allows lag and rolling features to incorporate each predicted value, maintaining feature consistency with training data.

### 5. Walk-Forward Validation
Standard k-fold CV is inappropriate for time series. The system uses `TimeSeriesSplit` for expanding-window validation, ensuring the model is never evaluated on past data.
