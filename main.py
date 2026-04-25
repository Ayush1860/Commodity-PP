"""
main.py — Pipeline Orchestrator (CLI Entry Point).

Runs the complete pipeline: Scrape → Clean → Feature Engineer → Train → Forecast.
Focused on Indian vegetable market price prediction.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

import pandas as pd

import config
from scraper import AgmarknetScraper
from data_processing import load_data, clean_pipeline
from feature_engineering import engineer_features, get_feature_columns
from model import train_model, save_model, load_model, forecast_next_n_days

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    commodity: str,
    state: str,
    days_back: int = 365,
    from_date: str = None,
    to_date: str = None,
    commodity_group: str = None,
    force_selenium: bool = False,
    tune: bool = True,
    n_trials: int = None,
    forecast_days: int = 7,
    skip_scrape: bool = False,
    existing_csv: str = None,
) -> dict:
    """
    Execute the full pipeline end-to-end.

    Args:
        commodity: Commodity name (e.g., "Wheat").
        state: State name (e.g., "Madhya Pradesh").
        days_back: Number of days of historical data to fetch.
        from_date: Explicit start date (overrides days_back).
        to_date: Explicit end date (defaults to today).
        commodity_group: Commodity group for API filter.
        force_selenium: Force Selenium-based scraping.
        tune: Run Optuna hyperparameter tuning.
        n_trials: Number of Optuna trials.
        forecast_days: Number of days to forecast.
        skip_scrape: Skip scraping, use existing CSV.
        existing_csv: Path to pre-existing CSV data.

    Returns:
        Dictionary with keys: data, forecast, metrics, model, importance
    """
    results = {}

    # ── 1. Resolve Dates ─────────────────────────────────────────────────
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    if to_date is None:
        to_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  🥬 VEGETABLE PRICE PREDICTION ENGINE")
    print(f"{'='*60}")
    print(f"  Vegetable  : {commodity}")
    print(f"  State      : {state}")
    print(f"  Date Range : {from_date} → {to_date}")
    print(f"  Forecast   : {forecast_days} days ahead")
    print(f"{'='*60}\n")

    # ── 2. Data Extraction ───────────────────────────────────────────────
    if skip_scrape and existing_csv:
        print(f"[1/5] Loading existing data from: {existing_csv}")
        raw_df = load_data(existing_csv)
    else:
        print("[1/5] Scraping data from Agmarknet...")
        scraper = AgmarknetScraper()
        raw_df = scraper.fetch_data(
            commodity_name=commodity,
            state=state,
            from_date=from_date,
            to_date=to_date,
            commodity_group=commodity_group,
            force_selenium=force_selenium,
        )

        if raw_df.empty:
            print("  ✗ No data retrieved. Check parameters and try again.")
            sys.exit(1)

        # Save raw data
        csv_path = scraper.save_to_csv(raw_df, commodity, state, from_date, to_date)
        print(f"  ✓ Raw data saved: {csv_path} ({len(raw_df)} records)")

    results["raw_data"] = raw_df

    # ── 3. Data Cleaning ─────────────────────────────────────────────────
    print("\n[2/5] Cleaning and preprocessing data...")
    clean_df = clean_pipeline(raw_df)
    print(f"  ✓ Cleaned shape: {clean_df.shape}")

    # Save cleaned data
    clean_csv = os.path.join(
        config.DATA_DIR,
        f"{commodity}_{state}_cleaned.csv".replace(" ", "_").lower(),
    )
    clean_df.to_csv(clean_csv, index=False)
    print(f"  ✓ Cleaned data saved: {clean_csv}")
    results["clean_data"] = clean_df

    # ── 4. Feature Engineering ───────────────────────────────────────────
    print("\n[3/5] Engineering features...")
    feature_df = engineer_features(
        clean_df,
        forecast_horizon=1,  # predict 1-day-ahead for iterative forecasting
    )
    feature_cols = get_feature_columns(feature_df)
    print(f"  ✓ Feature shape: {feature_df.shape}")
    print(f"  ✓ {len(feature_cols)} features created")
    results["feature_data"] = feature_df

    # ── 5. Model Training ───────────────────────────────────────────────
    tuning_label = "with Optuna tuning" if tune else "with default params"
    print(f"\n[4/5] Training XGBoost model ({tuning_label})...")
    model, metrics, importance = train_model(
        feature_df,
        tune=tune,
        n_trials=n_trials,
    )

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │     MODEL PERFORMANCE           │")
    print(f"  ├─────────────────────────────────┤")
    print(f"  │  Train RMSE : ₹{metrics['train_rmse']:>10.2f}    │")
    print(f"  │  Test  RMSE : ₹{metrics['test_rmse']:>10.2f}    │")
    print(f"  │  Train MAPE : {metrics['train_mape']:>9.2f}%     │")
    print(f"  │  Test  MAPE : {metrics['test_mape']:>9.2f}%     │")
    print(f"  └─────────────────────────────────┘")

    print(f"\n  Top 5 Features:")
    for _, row in importance.head(5).iterrows():
        bar = "█" * int(row["Importance"] * 50)
        print(f"    {row['Feature']:<25} {bar} ({row['Importance']:.3f})")

    # Save model
    model_path = save_model(model, commodity, state, metrics)
    print(f"\n  ✓ Model saved: {model_path}")

    results["model"] = model
    results["metrics"] = metrics
    results["importance"] = importance

    # ── 6. Forecast ──────────────────────────────────────────────────────
    print(f"\n[5/5] Generating {forecast_days}-day forecast...")
    forecast_df = forecast_next_n_days(
        model, clean_df, feature_cols, n_days=forecast_days
    )

    print(f"\n  ┌───────────────────────────────────────┐")
    print(f"  │    {forecast_days}-DAY PRICE FORECAST ({commodity})       │")
    print(f"  ├────────┬──────────────────────────────┤")
    print(f"  │  Day   │  Date       │  Predicted ₹   │")
    print(f"  ├────────┼─────────────┼────────────────┤")
    for _, row in forecast_df.iterrows():
        date_str = row["Date"].strftime("%Y-%m-%d")
        print(f"  │  t+{int(row['Day']):<3} │  {date_str}  │  ₹{row['Predicted_Price']:>10.2f}  │")
    print(f"  └────────┴─────────────┴────────────────┘")

    # Save forecast
    forecast_csv = os.path.join(
        config.DATA_DIR,
        f"{commodity}_{state}_forecast.csv".replace(" ", "_").lower(),
    )
    forecast_df.to_csv(forecast_csv, index=False)
    print(f"\n  ✓ Forecast saved: {forecast_csv}")

    results["forecast"] = forecast_df

    print(f"\n{'='*60}")
    print(f"  ✓ Pipeline complete!")
    print(f"  Run `streamlit run dashboard.py` for interactive visualization.")
    print(f"{'='*60}\n")

    return results


# ───────────────────────── CLI ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vegetable Price Prediction Engine — Indian Market",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --commodity Tomato --state Maharashtra
  python main.py --commodity Onion --state Karnataka --days 180 --no-tune
  python main.py --commodity Potato --state "Uttar Pradesh" --csv data/potato.csv --skip-scrape
  python main.py --commodity "Green Chilli" --state "Madhya Pradesh" --forecast 14
        """,
    )

    # Required
    parser.add_argument(
        "--commodity", required=True,
        help="Vegetable name (e.g., Tomato, Onion, Potato, Cabbage, Brinjal)"
    )
    parser.add_argument(
        "--state", required=True,
        help="State name (e.g., Maharashtra, Karnataka, 'Uttar Pradesh')"
    )

    # Date range
    parser.add_argument("--days", type=int, default=365,
                        help="Days of historical data (default: 365)")
    parser.add_argument("--from-date", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", default=None, help="End date (YYYY-MM-DD)")

    # Scraping options
    parser.add_argument("--group", default=None, help="Commodity group (e.g., Cereals)")
    parser.add_argument("--selenium", action="store_true",
                        help="Force Selenium scraping")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip scraping, use existing data")
    parser.add_argument("--csv", default=None,
                        help="Path to existing CSV data")

    # Model options
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=None,
                        help=f"Optuna trials (default: {config.N_OPTUNA_TRIALS})")
    parser.add_argument("--forecast", type=int, default=7,
                        help="Days to forecast (default: 7)")

    args = parser.parse_args()

    run_pipeline(
        commodity=args.commodity,
        state=args.state,
        days_back=args.days,
        from_date=args.from_date,
        to_date=args.to_date,
        commodity_group=args.group,
        force_selenium=args.selenium,
        tune=not args.no_tune,
        n_trials=args.trials,
        forecast_days=args.forecast,
        skip_scrape=args.skip_scrape,
        existing_csv=args.csv,
    )


if __name__ == "__main__":
    main()
