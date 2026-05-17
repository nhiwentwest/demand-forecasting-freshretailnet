"""
Stockout-aware feature engineering for demand forecasting.

All lag and rolling features are computed on y_eff (effective sales)
to prevent censored demand values from contaminating the feature space.

References:
    - VN2 Winner (arXiv:2601.18919): stockout-aware lags, Fourier, time-decay
    - IJF 2024 (FTGM): Fourier terms for seasonal demand
    - Wu 2026: lag + rolling features as top SHAP importance
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Stockout-Aware Lag & Rolling Features [VN2 Winner]
# ============================================================


def stockout_aware_lags(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Compute lag features on effective sales (y_eff), skipping stockout days.

    Lags are computed on y_eff which is NaN during stockouts. After computing,
    remaining NaNs are filled with per-series median to avoid information loss.

    Reference: VN2 Winner (arXiv:2601.18919) — All lags on y_eff.
    Verified by: Wu 2026 — SHAP shows lag features are most important.

    Args:
        df: DataFrame with 'y_eff' and 'series_id', sorted by date.
        lags: List of lag periods. Default: [1, 7, 14, 21, 28].

    Returns:
        DataFrame with lag feature columns.
    """
    if lags is None:
        lags = [1, 7, 14, 21, 28]

    df = df.copy()
    df = df.sort_values(["series_id", "dt"])

    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = df.groupby("series_id")["y_eff"].shift(lag)

    logger.info(f"Computed {len(lags)} stockout-aware lag features: {lags}")
    return df


def stockout_aware_rolling(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Compute rolling mean and std on effective sales, skipping stockout days.

    Rolling statistics use min_periods=1 to handle series with many stockouts.

    Reference: VN2 Winner (arXiv:2601.18919), Wu 2026 (SHAP importance).

    Args:
        df: DataFrame with 'y_eff' and 'series_id', sorted by date.
        windows: List of rolling window sizes. Default: [7, 14, 28].

    Returns:
        DataFrame with rolling mean and std columns.
    """
    if windows is None:
        windows = [7, 14, 28]

    df = df.copy()
    df = df.sort_values(["series_id", "dt"])

    for window in windows:
        df[f"rolling_mean_{window}"] = df.groupby("series_id")["y_eff"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = df.groupby("series_id")["y_eff"].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    logger.info(f"Computed rolling mean/std for windows: {windows}")
    return df


def stockout_ratio_features(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Compute rolling stockout ratio — captures chronic stockout patterns.

    Args:
        df: DataFrame with 'is_stockout' and 'series_id', sorted by date.
        windows: Rolling windows for stockout ratio. Default: [7, 14, 28].

    Returns:
        DataFrame with stockout ratio columns.
    """
    if windows is None:
        windows = [7, 14, 28]

    df = df.copy()
    df = df.sort_values(["series_id", "dt"])

    for window in windows:
        df[f"stockout_ratio_{window}d"] = df.groupby("series_id")[
            "is_stockout"
        ].transform(lambda x: x.rolling(window, min_periods=1).mean())

    logger.info(f"Computed stockout ratio for windows: {windows}")
    return df


# ============================================================
# Fourier Features [IJF 2024, VN2 Winner]
# ============================================================


def fourier_features(
    df: pd.DataFrame,
    periods: list[int] | None = None,
    n_harmonics: int = 3,
) -> pd.DataFrame:
    """Generate Fourier sin/cos features for smooth seasonality encoding.

    Captures multi-periodicity (weekly, monthly cycles) without abrupt
    transitions of one-hot encoding.

    References:
        - IJF 2024 (FTGM): Fourier time-varying Grey model
        - VN2 Winner: 3 harmonics for weekly + yearly cycles

    Args:
        df: DataFrame with 'dt' column.
        periods: Cycle periods in days. Default: [7, 30].
        n_harmonics: Number of Fourier harmonics per period.

    Returns:
        DataFrame with sin/cos feature columns.
    """
    if periods is None:
        periods = [7, 30]

    df = df.copy()

    # Day index from the earliest date
    df["day_idx"] = (df["dt"] - df["dt"].min()).dt.days

    for period in periods:
        for k in range(1, n_harmonics + 1):
            df[f"sin_{period}d_h{k}"] = np.sin(
                2 * np.pi * k * df["day_idx"] / period
            )
            df[f"cos_{period}d_h{k}"] = np.cos(
                2 * np.pi * k * df["day_idx"] / period
            )

    n_features = len(periods) * n_harmonics * 2
    logger.info(
        f"Generated {n_features} Fourier features: "
        f"periods={periods}, harmonics={n_harmonics}"
    )
    return df


# ============================================================
# Calendar Features
# ============================================================


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from the date column.

    These are required by the assignment schema (day_of_week, week_of_year,
    month, year) and capture abrupt weekly/monthly transitions.

    Args:
        df: DataFrame with 'dt' column.

    Returns:
        DataFrame with calendar feature columns.
    """
    df = df.copy()

    df["day_of_week"] = df["dt"].dt.dayofweek  # 0=Mon, 6=Sun
    df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
    df["month"] = df["dt"].dt.month
    df["year"] = df["dt"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_month"] = df["dt"].dt.day

    logger.info("Calendar features extracted")
    return df


# ============================================================
# Time-Decayed Sample Weights [VN2 Winner]
# ============================================================


def time_decay_weights(
    df: pd.DataFrame,
    recent_weight: float = 1.0,
    mid_weight: float = 0.5,
    old_weight: float = 0.25,
) -> pd.DataFrame:
    """Assign time-decayed sample weights — recent data weighted higher.

    Used as sample_weight in XGBoost/LightGBM training so the model
    focuses on recent demand patterns.

    Reference: VN2 Winner (arXiv:2601.18919) — Time-decayed weights.

    Args:
        df: DataFrame with 'dt' column.
        recent_weight: Weight for last 365 days.
        mid_weight: Weight for 365-730 days ago.
        old_weight: Weight for >730 days ago.

    Returns:
        DataFrame with 'sample_weight' column.
    """
    df = df.copy()

    max_date = df["dt"].max()
    days_ago = (max_date - df["dt"]).dt.days

    df["sample_weight"] = np.where(
        days_ago <= 365,
        recent_weight,
        np.where(days_ago <= 730, mid_weight, old_weight),
    )

    logger.info(
        f"Time-decay weights: recent({recent_weight}), "
        f"mid({mid_weight}), old({old_weight})"
    )
    return df


# ============================================================
# Target Encoding [Towards AI 2024]
# ============================================================


def seasonality_adjusted_target_encoding(
    df: pd.DataFrame,
    train_mask: pd.Series,
    cols: list[str] | None = None,
    smooth_factor: int = 100,
) -> pd.DataFrame:
    """Compute seasonality-adjusted target encoding for categorical features.

    Encoding is computed per (category, month) to capture seasonal demand
    levels without temporal data leakage. Smoothed with global mean.

    Reference: Towards AI 2024 — Dynamic categorical encoding for time series.

    Args:
        df: Full DataFrame (train + val + test).
        train_mask: Boolean mask indicating training rows.
        cols: Categorical columns to encode. Default: ['product_id', 'city_id'].
        smooth_factor: Regularization parameter for Bayesian smoothing.

    Returns:
        DataFrame with target-encoded columns.
    """
    if cols is None:
        cols = ["product_id", "city_id"]

    df = df.copy()
    train_data = df[train_mask]
    global_mean = train_data["y_eff"].mean()

    for col in cols:
        # Per (category, month) statistics
        group_stats = (
            train_data.groupby([col, "month"])["y_eff"]
            .agg(["mean", "count"])
            .reset_index()
        )
        group_stats.columns = [col, "month", "te_mean", "te_count"]

        # Bayesian smoothing: blend group mean with global mean
        group_stats["te_value"] = (
            group_stats["te_mean"] * group_stats["te_count"]
            + global_mean * smooth_factor
        ) / (group_stats["te_count"] + smooth_factor)

        # Merge back
        te_col = f"{col}_te"
        df = df.merge(
            group_stats[[col, "month", "te_value"]].rename(
                columns={"te_value": te_col}
            ),
            on=[col, "month"],
            how="left",
        )
        # Fill missing with global mean
        df[te_col] = df[te_col].fillna(global_mean)

    logger.info(f"Target encoding computed for: {cols}")
    return df


# ============================================================
# Cross-Store Features
# ============================================================


def cross_store_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute city-level aggregated demand as cross-store signals.

    Provides context about city-wide demand trends that individual
    store-product series cannot capture alone.

    Args:
        df: DataFrame with 'city_id', 'dt', 'y_eff' columns.

    Returns:
        DataFrame with cross-store feature columns.
    """
    df = df.copy()

    # City-level daily demand
    city_daily = (
        df.groupby(["city_id", "dt"])["y_eff"]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )
    city_daily.columns = [
        "city_id",
        "dt",
        "city_avg_demand",
        "city_total_demand",
        "city_active_series",
    ]

    df = df.merge(city_daily, on=["city_id", "dt"], how="left")

    # Category-level demand within city
    cat_daily = (
        df.groupby(["city_id", "first_category_id", "dt"])["y_eff"]
        .mean()
        .reset_index()
    )
    cat_daily.columns = ["city_id", "first_category_id", "dt", "category_city_avg"]

    df = df.merge(
        cat_daily, on=["city_id", "first_category_id", "dt"], how="left"
    )

    logger.info("Cross-store features computed (city-level and category-level)")
    return df


# ============================================================
# Feature Imputation
# ============================================================


def impute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in lag and rolling features with per-series median.

    Reference: VN2 Winner — Impute missing lags with per-series median.

    Args:
        df: DataFrame with lag/rolling columns that may have NaNs.

    Returns:
        DataFrame with imputed feature columns.
    """
    df = df.copy()

    lag_rolling_cols = [
        c
        for c in df.columns
        if c.startswith("lag_") or c.startswith("rolling_")
    ]

    for col in lag_rolling_cols:
        # Per-series median imputation
        df[col] = df.groupby("series_id")[col].transform(
            lambda x: x.fillna(x.median())
        )
        # Fallback: global median
        df[col] = df[col].fillna(df[col].median())

    n_imputed = sum(1 for c in lag_rolling_cols if df[c].isna().sum() == 0)
    logger.info(f"Imputed {len(lag_rolling_cols)} lag/rolling columns")

    return df


# ============================================================
# Full Feature Engineering Pipeline
# ============================================================


def run_feature_engineering(
    df: pd.DataFrame, config: dict, train_mask: pd.Series | None = None
) -> pd.DataFrame:
    """Run the complete feature engineering pipeline.

    Pipeline order:
        1. Stockout-aware lags
        2. Stockout-aware rolling stats
        3. Stockout ratio features
        4. Fourier features
        5. Calendar features
        6. Time-decay weights
        7. Target encoding (requires train_mask)
        8. Cross-store features
        9. Impute lag NaNs

    Args:
        df: Preprocessed DataFrame with y_eff, scale_factor, etc.
        config: Experiment configuration dict.
        train_mask: Boolean mask for training rows (needed for target encoding).

    Returns:
        DataFrame with all features ready for modeling.
    """
    feat_cfg = config.get("features", {})

    logger.info("=" * 60)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 60)

    # Ensure sorted
    df = df.sort_values(["series_id", "dt"]).reset_index(drop=True)

    # 1. Stockout-aware lags
    df = stockout_aware_lags(df, lags=feat_cfg.get("lags"))

    # 2. Rolling features
    df = stockout_aware_rolling(df, windows=feat_cfg.get("rolling_windows"))

    # 3. Stockout ratio
    df = stockout_ratio_features(df, windows=feat_cfg.get("stockout_ratio_windows"))

    # 4. Fourier features
    df = fourier_features(
        df,
        periods=feat_cfg.get("fourier_periods"),
        n_harmonics=feat_cfg.get("fourier_harmonics", 3),
    )

    # 5. Calendar features
    df = calendar_features(df)

    # 6. Time-decay weights
    decay_cfg = feat_cfg.get("time_decay", {})
    df = time_decay_weights(
        df,
        recent_weight=decay_cfg.get("recent_weight", 1.0),
        mid_weight=decay_cfg.get("mid_weight", 0.5),
        old_weight=decay_cfg.get("old_weight", 0.25),
    )

    # 7. Target encoding (only if train_mask provided)
    if train_mask is not None:
        df = seasonality_adjusted_target_encoding(df, train_mask)

    # 8. Cross-store features
    df = cross_store_features(df)

    # 9. Impute remaining NaNs in lag/rolling
    df = impute_lag_features(df)

    n_features = len([c for c in df.columns if c not in [
        "dt", "series_id", "y_eff", "y_scaled", "scale_factor",
        "sale_amount", "hours_sale", "hours_stock_status",
        "sample_weight", "is_stockout",
    ]])
    logger.info(f"Feature engineering complete: {n_features} features generated")

    return df
