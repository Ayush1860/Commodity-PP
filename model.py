"""
model.py — XGBoost Model Training, Tuning, Evaluation & Forecasting.

Uses XGBRegressor with Optuna hyperparameter tuning
and TimeSeriesSplit walk-forward validation.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import config
from feature_engineering import (
    engineer_features,
    get_feature_columns,
    add_temporal_features,
    add_lag_features,
    add_rolling_features,
    add_price_change_features,
)

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


# ─────────────────────────── Metrics ─────────────────────────────────────

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (as a %)."""
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100)


# ─────────────────────────── Optuna Tuning ───────────────────────────────

def optimize_hyperparams(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = None,
    n_splits: int = None,
) -> dict:
    """
    Use Optuna to find optimal XGBoost hyperparameters
    with TimeSeriesSplit cross-validation.
    """
    import optuna
    from optuna.samplers import TPESampler

    n_trials = n_trials or config.N_OPTUNA_TRIALS
    n_splits = n_splits or config.TIME_SERIES_SPLITS

    logger.info(
        f"Starting Optuna hyperparameter optimization: "
        f"{n_trials} trials, {n_splits}-fold TimeSeriesSplit"
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity": 0,
            "n_jobs": -1,
            "random_state": 42,
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        }

        fold_rmses = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)
            rmse = calculate_rmse(y_val.values, y_pred)
            fold_rmses.append(rmse)

        return np.mean(fold_rmses)

    # Suppress Optuna logging noise
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name="xgboost_commodity_price",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": 42,
    })

    logger.info(f"Best RMSE: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    return best_params


# ─────────────────────────── Training ────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    target_col: str = "Target_Price_t+1",
    feature_cols: list[str] = None,
    params: dict = None,
    tune: bool = True,
    n_trials: int = None,
) -> tuple:
    """
    Train an XGBoost model on the prepared DataFrame.

    Args:
        df: DataFrame with features and target column.
        target_col: Name of the target column.
        feature_cols: List of feature column names. Auto-detected if None.
        params: Pre-defined hyperparameters. If None and tune=True, uses Optuna.
        tune: Whether to run hyperparameter optimization.
        n_trials: Number of Optuna trials.

    Returns:
        Tuple of (trained_model, metrics_dict, feature_importance_df)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # ── Train/Test Split (last 20% as hold-out) ─────────────────────────
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        f"Split: train={len(X_train)}, test={len(X_test)} "
        f"(test from {df['Date'].iloc[split_idx] if 'Date' in df.columns else split_idx})"
    )

    # ── Hyperparameter Tuning ────────────────────────────────────────────
    if params is None:
        if tune:
            params = optimize_hyperparams(
                X_train, y_train, n_trials=n_trials
            )
        else:
            # Sensible defaults
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "learning_rate": 0.1,
                "max_depth": 6,
                "n_estimators": 500,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "min_child_weight": 3,
                "verbosity": 0,
                "n_jobs": -1,
                "random_state": 42,
            }

    # ── Train Final Model ────────────────────────────────────────────────
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_rmse": calculate_rmse(y_train.values, y_pred_train),
        "test_rmse": calculate_rmse(y_test.values, y_pred_test),
        "train_mape": calculate_mape(y_train.values, y_pred_train),
        "test_mape": calculate_mape(y_test.values, y_pred_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "best_params": params,
    }

    logger.info(f"Train RMSE: {metrics['train_rmse']:.4f}")
    logger.info(f"Test  RMSE: {metrics['test_rmse']:.4f}")
    logger.info(f"Train MAPE: {metrics['train_mape']:.2f}%")
    logger.info(f"Test  MAPE: {metrics['test_mape']:.2f}%")

    # ── Feature Importance ───────────────────────────────────────────────
    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return model, metrics, importance


# ─────────────────────────── Forecasting ─────────────────────────────────

def forecast_next_n_days(
    model: xgb.XGBRegressor,
    historical_df: pd.DataFrame,
    feature_cols: list[str],
    n_days: int = 7,
    target_col: str = "Modal_Price",
) -> pd.DataFrame:
    """
    Generate an n-day price forecast by iteratively predicting one day
    at a time and updating lag/rolling features.

    Args:
        model: Trained XGBoost model.
        historical_df: DataFrame with Date and Modal_Price columns
                       (must have enough rows for lag calculations).
        feature_cols: Feature columns used during training.
        n_days: Number of days to forecast.
        target_col: Price column name.

    Returns:
        DataFrame with Date and Predicted_Price for n_days ahead.
    """
    logger.info(f"Forecasting next {n_days} days...")

    # Work with a copy that includes all data for lag/rolling calculations
    df = historical_df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    last_date = df["Date"].max()
    predictions = []

    for day in range(1, n_days + 1):
        forecast_date = last_date + pd.Timedelta(days=day)

        # ── Create a new row with the forecast date ──────────────────────
        new_row = pd.DataFrame({"Date": [forecast_date], target_col: [np.nan]})

        # Copy other non-feature columns
        for col in ["Commodity", "State", "Arrivals_Tonnes"]:
            if col in df.columns:
                new_row[col] = df[col].iloc[-1]

        df = pd.concat([df, new_row], ignore_index=True)

        # ── Re-engineer features for the last row ────────────────────────
        df = add_temporal_features(df)
        df = add_lag_features(df, target_col=target_col)
        df = add_rolling_features(df, target_col=target_col)
        df = add_price_change_features(df, target_col=target_col)

        # ── Predict using only the last row ──────────────────────────────
        last_row = df.iloc[[-1]]
        X_pred = last_row[feature_cols].copy()

        # Handle any missing features (e.g., from insufficient data)
        for col in feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0
            X_pred[col] = X_pred[col].fillna(0)

        predicted_price = float(model.predict(X_pred)[0])
        predictions.append({
            "Date": forecast_date,
            "Predicted_Price": round(predicted_price, 2),
            "Day": day,
        })

        # ── Update the target in the DataFrame for next iteration ────────
        df.loc[df.index[-1], target_col] = predicted_price

        logger.info(
            f"  Day {day} ({forecast_date.strftime('%Y-%m-%d')}): "
            f"₹{predicted_price:.2f}"
        )

    forecast_df = pd.DataFrame(predictions)
    logger.info(f"Forecast complete. {len(forecast_df)} days predicted.")
    return forecast_df


# ─────────────────────────── Model Persistence ───────────────────────────

def save_model(
    model: xgb.XGBRegressor,
    commodity: str,
    state: str,
    metrics: dict = None,
) -> str:
    """Save trained model and metadata to the models directory."""
    safe_name = f"{commodity}_{state}".replace(" ", "_").lower()
    model_path = os.path.join(config.MODEL_DIR, f"{safe_name}_model.joblib")
    meta_path = os.path.join(config.MODEL_DIR, f"{safe_name}_meta.joblib")

    joblib.dump(model, model_path)
    if metrics:
        joblib.dump(metrics, meta_path)

    logger.info(f"Model saved to: {model_path}")
    return model_path


def load_model(commodity: str, state: str) -> tuple:
    """Load a saved model and its metadata."""
    safe_name = f"{commodity}_{state}".replace(" ", "_").lower()
    model_path = os.path.join(config.MODEL_DIR, f"{safe_name}_model.joblib")
    meta_path = os.path.join(config.MODEL_DIR, f"{safe_name}_meta.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at: {model_path}")

    model = joblib.load(model_path)
    metrics = joblib.load(meta_path) if os.path.exists(meta_path) else None

    logger.info(f"Model loaded from: {model_path}")
    return model, metrics


# ─────────────────────────── Walk-Forward Evaluation ─────────────────────

def walk_forward_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Target_Price_t+1",
    n_splits: int = None,
    params: dict = None,
) -> dict:
    """
    Perform walk-forward (expanding window) validation.

    Returns detailed per-fold metrics and aggregated performance.
    """
    n_splits = n_splits or config.TIME_SERIES_SPLITS
    tscv = TimeSeriesSplit(n_splits=n_splits)

    X = df[feature_cols]
    y = df[target_col]

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**(params or {
            "objective": "reg:squarederror",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 300,
            "verbosity": 0,
            "n_jobs": -1,
        }))
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_val)
        rmse = calculate_rmse(y_val.values, y_pred)
        mape = calculate_mape(y_val.values, y_pred)

        fold_results.append({
            "fold": fold,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "rmse": rmse,
            "mape": mape,
        })

        logger.info(
            f"Fold {fold}/{n_splits}: "
            f"train={len(X_train)}, val={len(X_val)}, "
            f"RMSE={rmse:.4f}, MAPE={mape:.2f}%"
        )

    avg_rmse = np.mean([r["rmse"] for r in fold_results])
    avg_mape = np.mean([r["mape"] for r in fold_results])

    return {
        "fold_results": fold_results,
        "avg_rmse": avg_rmse,
        "avg_mape": avg_mape,
    }


# ───────────────────────── CLI Usage ─────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost commodity price model.")
    parser.add_argument("input_csv", help="Path to feature-engineered CSV")
    parser.add_argument("--commodity", default="commodity", help="Commodity name")
    parser.add_argument("--state", default="state", help="State name")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip Optuna tuning, use default params")
    parser.add_argument("--trials", type=int, default=None,
                        help="Number of Optuna trials")
    parser.add_argument("--forecast", type=int, default=7,
                        help="Days to forecast (default: 7)")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, parse_dates=["Date"])

    # Train
    model, metrics, importance = train_model(
        df, tune=not args.no_tune, n_trials=args.trials
    )

    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"Train RMSE : ₹{metrics['train_rmse']:.2f}")
    print(f"Test  RMSE : ₹{metrics['test_rmse']:.2f}")
    print(f"Train MAPE : {metrics['train_mape']:.2f}%")
    print(f"Test  MAPE : {metrics['test_mape']:.2f}%")
    print(f"\nTop 10 Features:")
    print(importance.head(10).to_string(index=False))

    # Save
    model_path = save_model(model, args.commodity, args.state, metrics)

    # Forecast
    feature_cols = get_feature_columns(df)
    forecast = forecast_next_n_days(
        model, df, feature_cols, n_days=args.forecast
    )
    print(f"\n{'='*50}")
    print(f"{args.forecast}-DAY PRICE FORECAST")
    print(f"{'='*50}")
    print(forecast.to_string(index=False))
