"""
data_processing.py — Data Cleaning & Preprocessing Module.

Handles missing values, outlier removal, name normalization,
and prepares raw scraped data for feature engineering.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data and ensure correct dtypes."""
    df = pd.read_csv(filepath)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Min_Price", "Max_Price", "Modal_Price", "Arrivals_Tonnes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize commodity, market, and state names.
    - Strip whitespace
    - Convert to title case
    - Map known aliases
    """
    df = df.copy()

    name_aliases = {
        "Paddy(Dhan)": "Paddy",
        "Paddy (Dhan)": "Paddy",
        "Potato(Red)": "Potato Red",
        "Potato (Red)": "Potato Red",
        "Onion(Red)": "Onion Red",
        "Tomato(Local)": "Tomato",
        "Green Chilli": "Green Chilli",
    }

    for col in ["Commodity", "Market", "State", "Variety", "District"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            if col == "Commodity":
                df[col] = df[col].replace(name_aliases)

    return df


def handle_missing_values(
    df: pd.DataFrame,
    price_cols: list[str] = None,
    max_gap: int = 3,
) -> pd.DataFrame:
    """
    Handle missing values in the time-series data.

    Strategy:
    - For price columns: forward-fill, then interpolate gaps up to `max_gap` days.
    - For Arrivals: fill with 0 (non-trading day = no arrivals).
    - Drop rows where key price columns are still NaN after filling.
    """
    df = df.copy()
    if price_cols is None:
        price_cols = ["Min_Price", "Max_Price", "Modal_Price"]

    # Sort by date first
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    # Forward-fill then interpolate for price columns
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
            df[col] = df[col].interpolate(
                method="linear", limit=max_gap, limit_direction="forward"
            )

    # Fill arrivals with 0
    if "Arrivals_Tonnes" in df.columns:
        df["Arrivals_Tonnes"] = df["Arrivals_Tonnes"].fillna(0)

    # Drop rows where Modal_Price is still missing
    if "Modal_Price" in df.columns:
        before = len(df)
        df = df.dropna(subset=["Modal_Price"])
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing Modal_Price.")

    return df


def fill_non_trading_days(
    df: pd.DataFrame,
    market_col: str = "Market",
) -> pd.DataFrame:
    """
    Fill gaps for non-trading days (weekends, holidays).
    Creates a continuous daily date range per market and forward-fills prices.
    """
    if "Date" not in df.columns:
        return df

    df = df.copy()
    filled_frames = []

    # If there's only one market or no market column, treat as single series
    groups = df.groupby(market_col) if market_col in df.columns else [(None, df)]

    for market_name, group in groups:
        # Aggregate duplicates per date (e.g. multiple varieties at same market)
        # before reindexing, otherwise pandas raises "duplicate labels" error
        agg_dict = {}
        for col in ["Min_Price", "Max_Price", "Modal_Price"]:
            if col in group.columns:
                agg_dict[col] = "mean"
        if "Arrivals_Tonnes" in group.columns:
            agg_dict["Arrivals_Tonnes"] = "sum"
        for col in ["Market", "Commodity", "Variety", "State", "District"]:
            if col in group.columns:
                agg_dict[col] = "first"

        if agg_dict:
            group = group.groupby("Date").agg(agg_dict).reset_index()

        date_range = pd.date_range(
            start=group["Date"].min(),
            end=group["Date"].max(),
            freq="D",
        )
        group = group.set_index("Date").reindex(date_range)
        group.index.name = "Date"

        # Forward-fill categorical columns
        for col in ["Market", "Commodity", "Variety", "State", "District"]:
            if col in group.columns:
                group[col] = group[col].ffill().bfill()

        # Forward-fill then interpolate price columns
        for col in ["Min_Price", "Max_Price", "Modal_Price"]:
            if col in group.columns:
                group[col] = group[col].ffill()

        if "Arrivals_Tonnes" in group.columns:
            group["Arrivals_Tonnes"] = group["Arrivals_Tonnes"].fillna(0)

        group = group.reset_index().rename(columns={"index": "Date"})
        filled_frames.append(group)

    if filled_frames:
        return pd.concat(filled_frames, ignore_index=True).sort_values("Date")
    return df


def remove_outliers(
    df: pd.DataFrame,
    column: str = "Modal_Price",
    method: str = "iqr",
    factor: float = 1.5,
    action: str = "clip",
) -> pd.DataFrame:
    """
    Detect and handle outliers in price data.

    Args:
        column: Column to check for outliers.
        method: Detection method ('iqr' or 'zscore').
        factor: Multiplier for IQR or Z-score threshold.
        action: 'clip' (cap to bounds), 'remove' (drop rows),
                or 'flag' (add outlier flag column).

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    if column not in df.columns:
        logger.warning(f"Column '{column}' not found. Skipping outlier removal.")
        return df

    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
    elif method == "zscore":
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - factor * std
        upper_bound = mean + factor * std
    else:
        raise ValueError(f"Unknown outlier method: {method}")

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    n_outliers = outlier_mask.sum()

    if n_outliers > 0:
        logger.info(
            f"Found {n_outliers} outliers in '{column}' "
            f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])."
        )

        if action == "clip":
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        elif action == "remove":
            df = df[~outlier_mask].reset_index(drop=True)
        elif action == "flag":
            df[f"{column}_outlier"] = outlier_mask.astype(int)
        else:
            raise ValueError(f"Unknown outlier action: {action}")

    return df


def aggregate_by_date(
    df: pd.DataFrame,
    agg_cols: dict = None,
) -> pd.DataFrame:
    """
    Aggregate data to daily level (if multiple markets/varieties exist).

    By default, takes weighted average of prices and sum of arrivals.
    """
    if "Date" not in df.columns:
        return df

    if agg_cols is None:
        agg_cols = {}
        for col in ["Min_Price", "Max_Price", "Modal_Price"]:
            if col in df.columns:
                agg_cols[col] = "mean"
        if "Arrivals_Tonnes" in df.columns:
            agg_cols["Arrivals_Tonnes"] = "sum"

    if not agg_cols:
        return df

    # Preserve commodity info
    info_cols = {}
    for col in ["Commodity", "State"]:
        if col in df.columns:
            info_cols[col] = "first"

    all_agg = {**agg_cols, **info_cols}
    df_agg = df.groupby("Date").agg(all_agg).reset_index()

    # Round price columns
    for col in ["Min_Price", "Max_Price", "Modal_Price"]:
        if col in df_agg.columns:
            df_agg[col] = df_agg[col].round(2)

    return df_agg.sort_values("Date").reset_index(drop=True)


def clean_pipeline(
    df: pd.DataFrame,
    aggregate: bool = True,
    fill_gaps: bool = True,
    remove_price_outliers: bool = True,
) -> pd.DataFrame:
    """
    Run the full cleaning pipeline.

    Steps:
    1. Normalize names
    2. Handle missing values
    3. Optionally fill non-trading day gaps
    4. Optionally remove outliers
    5. Optionally aggregate to daily level
    """
    logger.info(f"Starting cleaning pipeline. Input shape: {df.shape}")

    df = normalize_names(df)
    df = handle_missing_values(df)

    if fill_gaps:
        df = fill_non_trading_days(df)

    if remove_price_outliers:
        df = remove_outliers(df, column="Modal_Price", method="iqr", action="clip")

    if aggregate:
        df = aggregate_by_date(df)

    logger.info(f"Cleaning complete. Output shape: {df.shape}")
    return df


# ───────────────────────── CLI Usage ─────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean commodity price data.")
    parser.add_argument("input_csv", help="Path to raw CSV from scraper")
    parser.add_argument("--output", "-o", help="Output CSV path")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Skip daily aggregation")
    parser.add_argument("--no-fill-gaps", action="store_true",
                        help="Skip non-trading day filling")

    args = parser.parse_args()

    df = load_data(args.input_csv)
    df = clean_pipeline(
        df,
        aggregate=not args.no_aggregate,
        fill_gaps=not args.no_fill_gaps,
    )

    output = args.output or args.input_csv.replace(".csv", "_cleaned.csv")
    df.to_csv(output, index=False)
    print(f"✓ Cleaned data saved to: {output}")
    print(f"  Shape: {df.shape}")
    print(f"\nSample:\n{df.head(10)}")
