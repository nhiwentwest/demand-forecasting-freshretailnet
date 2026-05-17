"""
Stockout-aware preprocessing for FreshRetailNet-50K.

Implements effective sales (y_eff) computation and stockout flag derivation
from hourly stock status data.

References:
    - VN2 Winner (arXiv:2601.18919): stockout-aware y_eff definition
    - FreshRetailNet-50K (arXiv:2505.16319): hours_stock_status format
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_hours_stock_status(status_str: str) -> np.ndarray:
    """Parse hourly stock status string into binary array.

    FRN-50K format: comma-separated 0/1 for each operating hour (6AM-10PM).
    1 = in-stock, 0 = out-of-stock.

    Args:
        status_str: String like "1,1,1,0,0,1,..." (17 values for 17 hours).

    Returns:
        Binary numpy array of shape (17,).
    """
    if pd.isna(status_str) or status_str == "":
        return np.ones(17)  # Default: assume in-stock if missing
    try:
        arr = np.array([int(x) for x in str(status_str).split(",")])
        return arr
    except (ValueError, AttributeError):
        return np.ones(17)


def compute_stockout_features(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Derive stockout-related features from hours_stock_status.

    Computes:
        - stockout_ratio: fraction of operating hours that were out-of-stock
        - is_stockout: binary flag (1 if stockout_ratio > threshold)
        - stockout_hours: count of out-of-stock hours
        - first_stockout_hour: hour index of first stockout (-1 if none)

    Args:
        df: DataFrame with 'hours_stock_status' column.
        threshold: Ratio threshold to flag a day as stockout.

    Returns:
        DataFrame with added stockout feature columns.
    """
    df = df.copy()
    logger.info("Parsing hours_stock_status and computing stockout features...")

    # Parse stock status arrays
    stock_arrays = df["hours_stock_status"].apply(parse_hours_stock_status)

    # Stockout ratio: fraction of hours out-of-stock
    df["stockout_ratio"] = stock_arrays.apply(lambda x: 1.0 - np.mean(x))

    # Binary stockout flag
    df["is_stockout"] = (df["stockout_ratio"] > threshold).astype(int)

    # Count of stockout hours
    df["stockout_hours"] = stock_arrays.apply(lambda x: int(np.sum(x == 0)))

    # First stockout hour (-1 if no stockout)
    df["first_stockout_hour"] = stock_arrays.apply(
        lambda x: int(np.argmin(x)) if np.any(x == 0) else -1
    )

    stockout_pct = df["is_stockout"].mean() * 100
    logger.info(
        f"Stockout features computed: {stockout_pct:.1f}% of days flagged as stockout "
        f"(threshold={threshold})"
    )

    return df


def compute_effective_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Compute effective sales: observed sales when in-stock, NaN when stockout.

    This prevents censored demand values from leaking into feature calculations.
    All downstream lag/rolling features should be computed on y_eff.

    Reference: VN2 Winner (arXiv:2601.18919) — Definition of y_eff.

    Args:
        df: DataFrame with 'sale_amount' and 'is_stockout' columns.

    Returns:
        DataFrame with added 'y_eff' column.
    """
    df = df.copy()

    df["y_eff"] = df["sale_amount"].where(df["is_stockout"] == 0, np.nan)

    n_censored = df["y_eff"].isna().sum()
    logger.info(
        f"Effective sales computed: {n_censored:,} censored observations "
        f"({n_censored / len(df) * 100:.1f}%)"
    )

    return df


def compute_dynamic_scaling(df: pd.DataFrame, window_weeks: int = 53, min_periods_days: int = 7, floor: float = 1.0) -> pd.DataFrame:
    """Per-series dynamic scaling using rolling mean of effective sales.

    Normalizes targets so a global model learns patterns instead of
    absolute magnitudes. Critical for multi-SKU forecasting.

    Reference: VN2 Winner (arXiv:2601.18919) — Per-series dynamic scaling.

    Args:
        df: DataFrame with 'y_eff' and 'series_id' columns, sorted by date.
        window_weeks: Rolling window size in weeks.
        min_periods_days: Minimum observations required.
        floor: Minimum scale factor to avoid division by tiny numbers.

    Returns:
        DataFrame with 'scale_factor' and 'y_scaled' columns.
    """
    df = df.copy()
    df = df.sort_values(["series_id", "dt"])

    window_days = window_weeks * 7

    df["scale_factor"] = (
        df.groupby("series_id")["y_eff"]
        .transform(lambda x: x.rolling(window_days, min_periods=min_periods_days).mean())
    )
    df["scale_factor"] = df["scale_factor"].clip(lower=floor)

    # Fill initial NaN scale factors with series median
    df["scale_factor"] = df.groupby("series_id")["scale_factor"].transform(
        lambda x: x.fillna(x.median())
    )
    # Final fallback: global median
    df["scale_factor"] = df["scale_factor"].fillna(df["scale_factor"].median())

    df["y_scaled"] = df["y_eff"] / df["scale_factor"]

    logger.info(
        f"Dynamic scaling applied: scale_factor range "
        f"[{df['scale_factor'].min():.2f}, {df['scale_factor'].max():.2f}]"
    )

    return df


def parse_hours_sale(hours_sale_str: str) -> np.ndarray:
    """Parse hourly sales string into numeric array.

    Args:
        hours_sale_str: Comma-separated hourly sales values.

    Returns:
        Numpy array of hourly sales.
    """
    if pd.isna(hours_sale_str) or hours_sale_str == "":
        return np.zeros(17)
    try:
        return np.array([float(x) for x in str(hours_sale_str).split(",")])
    except (ValueError, AttributeError):
        return np.zeros(17)


def run_preprocessing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Pipeline:
        1. Compute stockout features from hours_stock_status
        2. Compute effective sales (y_eff)
        3. Compute per-series dynamic scaling

    Args:
        df: Raw dataframe with FRN-50K columns.
        config: Experiment configuration dict.

    Returns:
        Preprocessed DataFrame ready for feature engineering.
    """
    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 60)

    # Step 1: Stockout features
    threshold = config.get("data", {}).get("stockout_threshold", 0.5)
    df = compute_stockout_features(df, threshold=threshold)

    # Step 2: Effective sales
    df = compute_effective_sales(df)

    # Step 3: Dynamic scaling
    feat_cfg = config.get("features", {})
    df = compute_dynamic_scaling(
        df,
        window_weeks=feat_cfg.get("scaling_window_weeks", 53),
        min_periods_days=feat_cfg.get("scaling_min_periods_days", 7),
        floor=feat_cfg.get("scaling_floor", 1.0),
    )

    logger.info("Preprocessing pipeline complete")
    return df
