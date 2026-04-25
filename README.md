# 🥬 Vegetable Price Prediction Engine

An end-to-end Machine Learning pipeline and interactive dashboard for forecasting **vegetable prices in Indian wholesale markets**. This project scrapes daily market price data from the Govt. of India's [Agmarknet](https://agmarknet.gov.in) portal, engineers time-series features, and uses an XGBoost model optimized with Optuna to predict future prices.

## 🎯 Target Domain

This project is specifically designed for the **Indian vegetable market** — predicting prices of commonly traded vegetables like Tomato, Onion, Potato, Cabbage, Cauliflower, Brinjal, Lady Finger, Green Chilli, Capsicum, and 20+ more vegetables across major Indian states.

## 🚀 Features

- **30+ Indian Vegetables**: Pre-configured with commonly traded vegetables available on Agmarknet.
- **Automated Data Extraction**: Scrapes live Agmarknet data via APIs (with Selenium fallback).
- **Automated Cleaning & Preprocessing**: Handles missing dates, outliers, and formats data for time-series forecasting.
- **Advanced Feature Engineering**: Automatically generates rolling windows, lags, and date-time features.
- **ML Training Pipeline**: Trains an XGBoost model, with optional Optuna hyperparameter optimization.
- **Interactive Dashboard**: A beautiful Streamlit dashboard with vegetable/state dropdowns, price trends, feature importance, and forecast visualization.

## 📁 Project Structure

```text
Commodity-PP/
├── config.py                 # Central configs, vegetable presets, state lists
├── dashboard.py              # Interactive Streamlit dashboard UI
├── data_processing.py        # Data cleaning and imputation logic
├── feature_engineering.py    # Lag & rolling window feature generation
├── main.py                   # CLI Orchestrator to run the pipeline
├── model.py                  # XGBoost training, forecasting, and evaluation
├── scraper.py                # Agmarknet data extraction (API + Selenium)
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── data/                     # Output directory for raw and cleaned CSVs
└── models/                   # Output directory for saved XGBoost models
```

## 🥦 Supported Vegetables

| Category       | Vegetables                                                         |
|----------------|---------------------------------------------------------------------|
| **Staples**    | Tomato, Onion, Potato, Green Chilli, Garlic, Ginger                |
| **Leafy**      | Spinach, Methi (Fenugreek), Coriander                             |
| **Gourds**     | Bottle Gourd, Bitter Gourd, Ridge Gourd, Snake Gourd, Ash Gourd   |
| **Cruciferous**| Cabbage, Cauliflower, Capsicum                                     |
| **Others**     | Brinjal, Lady Finger, Beans, Peas, Drumstick, Cucumber, Carrot    |

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ayush1860/Commodity-PP.git
   cd Commodity-PP
   ```

2. **Create a virtual environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Interactive Dashboard (Recommended)
Run the Streamlit dashboard for a complete GUI experience:
```bash
streamlit run dashboard.py
```
*This will open the dashboard in your browser, where you can select from 30+ vegetables and 19 major states, run the pipeline, and view interactive Plotly charts.*

### 2. Command Line Interface (CLI)
You can also run the complete pipeline end-to-end directly from your terminal.

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

**Predict Green Chilli with a 14-day forecast:**
```bash
python main.py --commodity "Green Chilli" --state "Madhya Pradesh" --forecast 14
```

**Check all available CLI options:**
```bash
python main.py --help
```

## 📈 ML Architecture

1. **Data Source**: Agmarknet 2.0 API (`Modal_Price`, `Min_Price`, `Max_Price`, `Arrivals_Tonnes`)
2. **Algorithm**: XGBoost Regressor
3. **Hyperparameter Tuning**: Optuna (Tuning learning rate, max depth, subsample, etc.)
4. **Time Series Handling**: Walk-forward validation across multiple splits to evaluate true sequential forecasting performance based on `RMSE` and `MAPE`.

## 📊 Feature Categories

| Category           | Features                                                            |
|--------------------|----------------------------------------------------------------------|
| **Temporal**       | Year, Month, Day, DayOfWeek, Quarter, WeekOfYear, IsWeekend        |
| **Cyclical**       | Month_Sin, Month_Cos, DayOfWeek_Sin, DayOfWeek_Cos                 |
| **Lags**           | Price_Lag_1, Price_Lag_7, Price_Lag_14, Price_Lag_30                |
| **Rolling Stats**  | Rolling_Mean_7/30, Rolling_Std_7/30, Rolling_Min/Max_7/30          |
| **Momentum**       | Price_Change_1d/7d, Price_Pct_Change_1d/7d, Price_Spread           |
| **Supply**         | Arrivals_Lag_1/7, Arrivals_Rolling_Mean_7/30                       |
