# 🥬 Vegetable Price Prediction Engine

An end-to-end Machine Learning pipeline and interactive dashboard for forecasting **vegetable prices in Indian wholesale markets (Mandis)**. This project scrapes daily market price data from the Govt. of India's [Agmarknet 2.0](https://agmarknet.gov.in) portal, engineers time-series features, and uses an XGBoost model optimized with Optuna to predict future prices.

> **Data Source:** Directorate of Marketing & Inspection, Ministry of Agriculture & Farmers Welfare, Govt. of India — via the Agmarknet 2.0 REST API.

## 🎯 Target Domain

This project is specifically designed for the **Indian vegetable market** — predicting wholesale prices of commonly traded vegetables like Tomato, Onion, Potato, Cabbage, Cauliflower, Brinjal, Lady Finger, Green Chilli, Capsicum, and 20+ more vegetables across 19 major Indian states.

## 🚀 Features

- **30+ Indian Vegetables** — Pre-configured with commonly traded vegetables available on Agmarknet.
- **Automated Data Extraction** — Scrapes live data via the Agmarknet 2.0 POST API with Selenium fallback.
- **Smart Cleaning & Preprocessing** — Handles missing dates, outlier clipping, non-trading day gap filling, and multi-market aggregation.
- **Adaptive Feature Engineering** — Auto-adapts lag periods and rolling windows based on dataset size.
- **ML Training Pipeline** — XGBoost regressor with optional Optuna hyperparameter optimization and walk-forward validation.
- **Interactive Dashboard** — Streamlit dashboard with vegetable/state dropdowns, Plotly price charts, feature importance, and forecast visualization.

## 📁 Project Structure

```text
Commodity-PP/
├── config.py                 # Central configs, vegetable presets, state lists
├── scraper.py                # Agmarknet 2.0 API data extraction (POST + Selenium fallback)
├── data_processing.py        # Data cleaning, imputation, outlier handling
├── feature_engineering.py    # Lag, rolling, temporal, and momentum features
├── model.py                  # XGBoost training, Optuna tuning, forecasting
├── main.py                   # CLI orchestrator for end-to-end pipeline
├── dashboard.py              # Interactive Streamlit dashboard UI
├── test_scraper.py           # Quick scraper validation script
├── requirements.txt          # Python dependencies
├── ARCHITECTURE.md           # Detailed system architecture documentation
├── .gitignore                # Git ignore rules
├── data/                     # [Generated] Raw and cleaned CSVs
└── models/                   # [Generated] Saved XGBoost models + metadata
```

## 🥦 Supported Vegetables

| Category        | Vegetables                                                        |
|-----------------|-------------------------------------------------------------------|
| **Staples**     | Tomato, Onion, Potato, Green Chilli, Garlic, Ginger               |
| **Leafy**       | Spinach, Methi (Fenugreek), Coriander                             |
| **Gourds**      | Bottle Gourd, Bitter Gourd, Ridge Gourd, Snake Gourd, Ash Gourd   |
| **Cruciferous** | Cabbage, Cauliflower, Capsicum                                    |
| **Others**      | Brinjal, Lady Finger, Beans, Peas, Drumstick, Cucumber, Carrot   |

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ayush1860/Commodity-PP.git
   cd Commodity-PP
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **macOS only — install OpenMP** (required by XGBoost):
   ```bash
   brew install libomp
   ```

## 💻 Usage

### 1. Interactive Dashboard (Recommended)
```bash
streamlit run dashboard.py
```
Opens a browser dashboard where you can select from 30+ vegetables and 19 states, run the pipeline, and view interactive Plotly charts with forecasts.

### 2. Command Line Interface

**Predict Tomato prices in Maharashtra:**
```bash
python main.py --commodity Tomato --state Maharashtra
```

**Predict Onion prices with shorter history:**
```bash
python main.py --commodity Onion --state Karnataka --days 180 --no-tune
```

**Use existing data, skip scraping:**
```bash
python main.py --commodity Potato --state "Uttar Pradesh" --skip-scrape --csv data/potato_uttar_pradesh_cleaned.csv
```

**14-day forecast for Green Chilli:**
```bash
python main.py --commodity "Green Chilli" --state "Madhya Pradesh" --forecast 14
```

**All CLI options:**
```bash
python main.py --help
```

## 📈 ML Pipeline Overview

| Stage                | Method                                                                 |
|----------------------|------------------------------------------------------------------------|
| **Data Source**       | Agmarknet 2.0 REST API (POST) — `Modal_Price`, `Min_Price`, `Max_Price`, `Arrivals_Tonnes` |
| **Preprocessing**    | Outlier clipping (IQR), non-trading day filling, daily aggregation     |
| **Feature Engineering** | Adaptive lags, rolling statistics, cyclical encoding, momentum       |
| **Algorithm**        | XGBoost Regressor                                                      |
| **Tuning**           | Optuna (TPE sampler) with TimeSeriesSplit cross-validation             |
| **Evaluation**       | Walk-forward validation — RMSE and MAPE                               |
| **Forecasting**      | Iterative 1-step-ahead with feature recalculation                     |

> For detailed architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 📊 Feature Categories

| Category           | Features                                                            |
|--------------------|----------------------------------------------------------------------|
| **Temporal**       | Year, Month, Day, DayOfWeek, Quarter, WeekOfYear, IsWeekend         |
| **Cyclical**       | Month_Sin, Month_Cos, DayOfWeek_Sin, DayOfWeek_Cos                  |
| **Lags**           | Price_Lag_1, Price_Lag_7, Price_Lag_14, Price_Lag_30 (auto-adapted)  |
| **Rolling Stats**  | Rolling_Mean_7/30, Rolling_Std_7/30, Rolling_Min/Max_7/30           |
| **Momentum**       | Price_Change_1d/7d, Price_Pct_Change_1d/7d, Price_Spread            |
| **Supply**         | Arrivals_Lag_1/7, Arrivals_Rolling_Mean_7/30                        |

## 📋 Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt) for all dependencies
- macOS: `brew install libomp` for XGBoost OpenMP support
