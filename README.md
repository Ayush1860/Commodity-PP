# 🌾 Commodity Price Prediction Engine

An end-to-end Machine Learning pipeline and interactive dashboard for forecasting agricultural commodity prices in India. This project scrapes daily market price data from the Govt. of India's [Agmarknet](https://agmarknet.gov.in) portal, engineers time-series features, and uses an XGBoost model optimized with Optuna to predict future commodity prices.

## 🚀 Features

- **Automated Data Extraction**: Scrapes live Agmarknet data via APIs (and fallback Selenium).
- **Automated Cleaning & Preprocessing**: Handles missing dates, outliers, and formats data for time-series forecasting.
- **Advanced Feature Engineering**: Automatically generates rolling windows, lags, and date-time features.
- **ML Training Pipeline**: Trains an XGBoost model, with optional Optuna hyperparameter optimization.
- **Interactive Dashboard**: A beautiful Streamlit dashboard to test predictions, visualize historical price trends, view feature importance, and generate forecasts.

## 📁 Project Structure

```text
CommodityPP/
├── config.py                 # Central configurations (APIs, limits, paths)
├── dashboard.py              # Interactive Streamlit dashboard UI
├── data_processing.py        # Data cleaning and imputation logic
├── feature_engineering.py    # Lag & rolling window feature generation
├── main.py                   # CLI Orchestrator to run the pipeline
├── model.py                  # XGBoost training, forecasting, and evaluation
├── scraper.py                # Agmarknet data extraction (API + Selenium)
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── data/                     # Output directory for raw and Cleaned CSVs
└── models/                   # Output directory for saved XGBoost models
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/CommodityPP.git
   cd CommodityPP
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
*This will open the dashboard in your browser, allowing you to configure parameters, run the pipeline, and view interactive Plotly charts.*

### 2. Command Line Interface (CLI)
You can also run the complete pipeline end-to-end directly from your terminal.

**Basic Run (Fetch 1 year of data, train, and forecast 7 days):**
```bash
python main.py --commodity Wheat --state "Madhya Pradesh"
```

**Advanced Run (Skip scraping if CSV exists, skip tuning):**
```bash
python main.py --commodity Tomato --state Karnataka --skip-scrape --csv data/tomato_karnataka_cleaned.csv --no-tune
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
