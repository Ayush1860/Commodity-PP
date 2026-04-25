"""
feature_engineering.py — Time-Series Feature Engineering Module.

Creates temporal, lag, rolling average, volume, and price-change
features to feed into the XGBoost forecasting model.
"""

import logging

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the Date column:
    Year, Month, Day, DayOfWeek, Quarter, WeekOfYear, IsWeekend,
    DayOfYear, MonthSin/MonthCos (cyclical encoding).
    """
    df = df.copy()

    if "Date" not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column.")

    dt = df["Date"].dt

    df["Year"] = dt.year
    df["Month"] = dt.month
    df["Day"] = dt.day
    df["DayOfWeek"] = dt.dayofweek  # Monday=0, Sunday=6
    df["Quarter"] = dt.quarter
    df["WeekOfYear"] = dt.isocalendar().week.astype(int)
    df["DayOfYear"] = dt.dayofyear
    df["IsWeekend"] = (dt.dayofweek >= 5).astype(int)

    # Cyclical encoding for month (captures Jan ↔ Dec continuity)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # Cyclical encoding for day of week
    df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    logger.info("Temporal features added.")
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "Modal_Price",
    lags: list[int] = None,
) -> pd.DataFrame:
    """
    Create lag (lookback) features for the target price column.

    Lags represent the value of the target at t-n, helping the model
    learn from historical trends.
    """
    df = df.copy()
    lags = lags or config.LAG_PERIODS  # [1, 7, 14, 30]

    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found. Skipping lags.")
        return df

    for lag in lags:
        df[f"Price_Lag_{lag}"] = df[target_col].shift(lag)
        logger.debug(f"  Added Price_Lag_{lag}")

    # Additional: Arrivals lag
    if "Arrivals_Tonnes" in df.columns:
        df["Arrivals_Lag_1"] = df["Arrivals_Tonnes"].shift(1)
        df["Arrivals_Lag_7"] = df["Arrivals_Tonnes"].shift(7)

    logger.info(f"Lag features added for periods: {lags}")
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Modal_Price",
    windows: list[int] = None,
) -> pd.DataFrame:
    """
    Calculate rolling window statistics to smooth out daily volatility.

    Features: rolling mean, std, min, max for each window size.
    """
    df = df.copy()
    windows = windows or config.ROLLING_WINDOWS  # [7, 30]

    if target_col not in df.columns:
        logger.warning(
            f"Target column '{target_col}' not found. Skipping rolling features."
        )
        return df

    for window in windows:
        # Rolling mean
        df[f"Rolling_Mean_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        # Rolling standard deviation (volatility measure)
        df[f"Rolling_Std_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )
        # Rolling min and max (price range indicators)
        df[f"Rolling_Min_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).min()
        )
        df[f"Rolling_Max_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).max()
        )

    # Rolling arrivals mean (supply trend)
    if "Arrivals_Tonnes" in df.columns:
        for window in windows:
            df[f"Arrivals_Rolling_Mean_{window}"] = (
                df["Arrivals_Tonnes"].rolling(window=window, min_periods=1).mean()
            )

    logger.info(f"Rolling features added for windows: {windows}")
    return df


def add_price_change_features(
    df: pd.DataFrame,
    target_col: str = "Modal_Price",
) -> pd.DataFrame:
    """
    Add percentage and absolute price change features.

    These capture momentum and trend direction.
    """
    df = df.copy()

    if target_col not in df.columns:
        return df

    # Absolute price change
    df["Price_Change_1d"] = df[target_col].diff(1)
    df["Price_Change_7d"] = df[target_col].diff(7)

    # Percentage price change
    df["Price_Pct_Change_1d"] = df[target_col].pct_change(1) * 100
    df["Price_Pct_Change_7d"] = df[target_col].pct_change(7) * 100

    # Price spread (Max - Min as a volatility proxy)
    if "Max_Price" in df.columns and "Min_Price" in df.columns:
        df["Price_Spread"] = df["Max_Price"] - df["Min_Price"]
        df["Price_Spread_Pct"] = (
            df["Price_Spread"] / df[target_col] * 100
        ).replace([np.inf, -np.inf], 0)

    logger.info("Price change features added.")
    return df


def add_target_column(
    df: pd.DataFrame,
    target_col: str = "Modal_Price",
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Create the prediction target: price at time t+horizon.

    This shifts the target backward so that each row's features (at time t)
    align with the future price (at time t+horizon).
    """
    df = df.copy()
    df[f"Target_Price_t+{horizon}"] = df[target_col].shift(-horizon)
    logger.info(f"Target column created: Target_Price_t+{horizon}")
    return df


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaN values produced by lag/rolling/target calculations.
    These are typically the first and last few rows.
    """
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} incomplete rows (NaN from lag/rolling/target).")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns (excluding Date, identifiers, and target).
    """
    exclude_patterns = [
        "Date", "Market", "Commodity", "Variety", "State", "District",
        "Target_", "Modal_Price", "Min_Price", "Max_Price",
    ]

    feature_cols = []
    for col in df.columns:
        if not any(pat in col for pat in exclude_patterns):
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32, float, int]:
                feature_cols.append(col)

    return feature_cols


def engineer_features(
    df: pd.DataFrame,
    target_col: str = "Modal_Price",
    forecast_horizon: int = 1,
    lags: list[int] = None,
    windows: list[int] = None,
    add_target: bool = True,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Steps:
    1. Auto-adapt lags/windows to dataset size
    2. Temporal features (year, month, day, cyclical encodings)
    3. Lag features (configurable periods)
    4. Rolling window features (mean, std, min, max)
    5. Price change features (momentum, spread)
    6. Target column (future price at t+horizon)
    7. Drop rows with NaN
    """
    logger.info(f"Starting feature engineering. Input shape: {df.shape}")

    n_rows = len(df)

    # Auto-adapt lags and windows for small datasets to avoid losing all rows
    # We need at least ~20 rows left after NaN drops for meaningful training
    if lags is None:
        lags = [l for l in config.LAG_PERIODS if l < n_rows * 0.6]
        if not lags:
            lags = [1]  # always keep at least lag-1
        logger.info(f"Auto-adapted lags for {n_rows} rows: {lags}")

    if windows is None:
        windows = [w for w in config.ROLLING_WINDOWS if w <= n_rows * 0.6]
        if not windows:
            windows = [min(7, max(2, n_rows // 3))]
        logger.info(f"Auto-adapted windows for {n_rows} rows: {windows}")

    df = add_temporal_features(df)
    df = add_lag_features(df, target_col=target_col, lags=lags)
    df = add_rolling_features(df, target_col=target_col, windows=windows)
    df = add_price_change_features(df, target_col=target_col)

    if add_target:
        df = add_target_column(df, target_col=target_col, horizon=forecast_horizon)

    if drop_na:
        df = drop_incomplete_rows(df)

    logger.info(f"Feature engineering complete. Output shape: {df.shape}")
    logger.info(f"Feature columns: {get_feature_columns(df)}")

    return df


# ───────────────────────── CLI Usage ─────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Engineer features from cleaned commodity data."
    )
    parser.add_argument("input_csv", help="Path to cleaned CSV")
    parser.add_argument("--output", "-o", help="Output CSV path")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Forecast horizon in days (default: 1)")
    parser.add_argument("--no-drop-na", action="store_true",
                        help="Keep rows with NaN values")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, parse_dates=["Date"])
    df = engineer_features(
        df,
        forecast_horizon=args.horizon,
        drop_na=not args.no_drop_na,
    )

    output = args.output or args.input_csv.replace(".csv", "_features.csv")
    df.to_csv(output, index=False)
    print(f"✓ Engineered features saved to: {output}")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {get_feature_columns(df)}")
    print(f"\nSample:\n{df.head()}")
