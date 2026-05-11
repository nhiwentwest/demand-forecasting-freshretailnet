"""
Tree-based forecasting models: XGBoost + LightGBM.

Per-city models with SHAP explainability and checkpoint support.

References:
    - Wu 2026 (JID): XGBoost RMSE=0.714, R²=0.816 on FRN-50K
    - VN2 Winner (arXiv:2601.18919): XGBoost with stockout-aware features
    - Chalmers 2024: LightGBM improves 33.2% over exponential smoothing
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Feature columns for tree models
# ============================================================

# Lag features (top SHAP importance per Wu 2026)
LAG_FEATURES = [f"lag_{k}" for k in [1, 7, 14, 21, 28]]

# Rolling statistics
ROLLING_FEATURES = [
    f"rolling_{stat}_{w}"
    for stat in ["mean", "std"]
    for w in [7, 14, 28]
]

# Stockout pattern features
STOCKOUT_FEATURES = [
    f"stockout_ratio_{w}d" for w in [7, 14, 28]
] + ["stockout_ratio", "is_stockout"]

# Calendar features
CALENDAR_FEATURES = [
    "day_of_week", "week_of_year", "month", "year",
    "is_weekend", "day_of_month",
]

# Fourier features
FOURIER_FEATURES = [
    f"{func}_{p}d_h{k}"
    for p in [7, 30]
    for k in [1, 2, 3]
    for func in ["sin", "cos"]
]

# Categorical and encoded features
CATEGORICAL_FEATURES = [
    "city_id", "store_id", "first_category_id",
    "second_category_id", "third_category_id",
]

ENCODED_FEATURES = ["product_id_te", "city_id_te"]

# Promotion and weather
PROMO_WEATHER_FEATURES = [
    "discount", "activity_flag", "holiday_flag",
    "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
]

# Cross-store features
CROSS_STORE_FEATURES = [
    "city_avg_demand", "city_total_demand",
    "city_active_series", "category_city_avg",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get all available feature columns from the DataFrame.

    Returns only columns that exist in the DataFrame, handling cases
    where some features may not have been generated.
    """
    all_features = (
        LAG_FEATURES
        + ROLLING_FEATURES
        + STOCKOUT_FEATURES
        + CALENDAR_FEATURES
        + FOURIER_FEATURES
        + ENCODED_FEATURES
        + PROMO_WEATHER_FEATURES
        + CROSS_STORE_FEATURES
    )
    available = [c for c in all_features if c in df.columns]
    logger.info(f"Using {len(available)}/{len(all_features)} feature columns")
    return available


# ============================================================
# XGBoost Per-City Training [Wu 2026]
# ============================================================


def train_xgboost_per_city(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    target_col: str = "y_scaled",
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train per-city XGBoost models with time-decay sample weights.

    Wu 2026 confirms per-city XGBoost achieves RMSE=0.714, R²=0.816
    on FRN-50K when using lag + rolling features as primary signals.

    Args:
        train_df: Training data with features.
        val_df: Validation data for early stopping.
        config: XGBoost config from experiment.yaml.
        target_col: Target column name.
        checkpoint_dir: Directory to save model checkpoints.

    Returns:
        Dict mapping city_id → fitted XGBRegressor.
    """
    import xgboost as xgb

    xgb_cfg = config.get("stage2_forecasting", {}).get("xgboost", {})
    feature_cols = get_feature_columns(train_df)

    # Convert categoricals to codes for XGBoost
    cat_in_features = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    models = {}
    cities = train_df["city_id"].unique()

    for city in cities:
        city_train = train_df[train_df["city_id"] == city].copy()
        city_val = val_df[val_df["city_id"] == city].copy()

        if len(city_train) < 100:
            logger.warning(f"City {city}: only {len(city_train)} train samples, skipping")
            continue

        # Prepare features
        X_train = city_train[feature_cols].copy()
        X_val = city_val[feature_cols].copy()

        # Encode categoricals as int codes
        for col in cat_in_features:
            for frame in [X_train, X_val]:
                if frame[col].dtype.name == "category":
                    frame[col] = frame[col].cat.codes
                else:
                    frame[col] = frame[col].astype("category").cat.codes

        y_train = city_train[target_col].values
        y_val = city_val[target_col].values
        sample_weights = city_train["sample_weight"].values

        # Handle NaN targets (stockout days without recovery)
        valid_train = ~np.isnan(y_train)
        valid_val = ~np.isnan(y_val)

        model = xgb.XGBRegressor(
            n_estimators=xgb_cfg.get("n_estimators", 2000),
            max_depth=xgb_cfg.get("max_depth", 8),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            tree_method="hist",
            enable_categorical=False,
            random_state=42,
        )

        model.fit(
            X_train[valid_train],
            y_train[valid_train],
            sample_weight=sample_weights[valid_train],
            eval_set=[(X_val[valid_val], y_val[valid_val])],
            verbose=False,
        )

        models[city] = model
        best_iter = model.best_iteration if hasattr(model, "best_iteration") else -1
        logger.info(
            f"City {city}: XGBoost trained on {valid_train.sum():,} samples, "
            f"best_iteration={best_iter}"
        )

        # Save checkpoint
        if checkpoint_dir:
            ckpt_path = Path(checkpoint_dir) / f"xgboost_{city}.json"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(ckpt_path))

    logger.info(f"Trained {len(models)} per-city XGBoost models")
    return models


# ============================================================
# LightGBM Per-City Training [Chalmers 2024]
# ============================================================


def train_lightgbm_per_city(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    target_col: str = "y_scaled",
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train per-city LightGBM models.

    Complementary to XGBoost — typically faster with similar accuracy.
    Chalmers 2024 shows 33.2% improvement over exponential smoothing.

    Args:
        train_df: Training data with features.
        val_df: Validation data for early stopping.
        config: LightGBM config from experiment.yaml.
        target_col: Target column name.
        checkpoint_dir: Directory to save model checkpoints.

    Returns:
        Dict mapping city_id → fitted LGBMRegressor.
    """
    import lightgbm as lgb

    lgb_cfg = config.get("stage2_forecasting", {}).get("lightgbm", {})
    feature_cols = get_feature_columns(train_df)
    cat_in_features = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    models = {}
    cities = train_df["city_id"].unique()

    for city in cities:
        city_train = train_df[train_df["city_id"] == city].copy()
        city_val = val_df[val_df["city_id"] == city].copy()

        if len(city_train) < 100:
            continue

        X_train = city_train[feature_cols].copy()
        X_val = city_val[feature_cols].copy()

        for col in cat_in_features:
            for frame in [X_train, X_val]:
                if frame[col].dtype.name == "category":
                    frame[col] = frame[col].cat.codes
                else:
                    frame[col] = frame[col].astype("category").cat.codes

        y_train = city_train[target_col].values
        y_val = city_val[target_col].values
        sample_weights = city_train["sample_weight"].values

        valid_train = ~np.isnan(y_train)
        valid_val = ~np.isnan(y_val)

        model = lgb.LGBMRegressor(
            n_estimators=lgb_cfg.get("n_estimators", 2000),
            num_leaves=lgb_cfg.get("num_leaves", 63),
            learning_rate=lgb_cfg.get("learning_rate", 0.05),
            feature_fraction=lgb_cfg.get("feature_fraction", 0.8),
            bagging_fraction=lgb_cfg.get("bagging_fraction", 0.8),
            bagging_freq=1,
            random_state=42,
            verbose=-1,
        )

        model.fit(
            X_train[valid_train],
            y_train[valid_train],
            sample_weight=sample_weights[valid_train],
            eval_set=[(X_val[valid_val], y_val[valid_val])],
        )

        models[city] = model

        if checkpoint_dir:
            ckpt_path = Path(checkpoint_dir) / f"lightgbm_{city}.txt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            model.booster_.save_model(str(ckpt_path))

    logger.info(f"Trained {len(models)} per-city LightGBM models")
    return models


# ============================================================
# SHAP Explainability [Wu 2026]
# ============================================================


def compute_shap_importance(model, X: pd.DataFrame, max_samples: int = 5000) -> pd.DataFrame:
    """Compute SHAP feature importance for a tree model.

    Wu 2026 confirms: lag features and rolling averages dominate SHAP
    importance for XGBoost on FRN-50K.

    Args:
        model: Fitted XGBoost or LightGBM model.
        X: Feature DataFrame.
        max_samples: Max samples for SHAP computation (speed).

    Returns:
        DataFrame with feature names and mean |SHAP| values, sorted.
    """
    import shap

    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    importance = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    logger.info(f"Top 5 SHAP features: {importance.head()['feature'].tolist()}")
    return importance


# ============================================================
# Prediction
# ============================================================


def predict_per_city(
    models: dict[str, Any],
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    scale_back: bool = True,
) -> np.ndarray:
    """Generate predictions from per-city models.

    Args:
        models: Dict mapping city_id → model.
        df: DataFrame with features and city_id.
        feature_cols: Feature columns. Auto-detected if None.
        scale_back: If True, multiply predictions by scale_factor.

    Returns:
        Array of predictions aligned with df index.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    predictions = np.full(len(df), np.nan)

    for city, model in models.items():
        mask = df["city_id"] == city
        if mask.sum() == 0:
            continue

        X = df.loc[mask, feature_cols].copy()
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
        for col in cat_cols:
            if X[col].dtype.name == "category":
                X[col] = X[col].cat.codes
            else:
                X[col] = X[col].astype("category").cat.codes

        preds = model.predict(X)
        predictions[mask.values] = preds

    if scale_back and "scale_factor" in df.columns:
        predictions = predictions * df["scale_factor"].values

    n_predicted = (~np.isnan(predictions)).sum()
    logger.info(f"Predicted {n_predicted:,}/{len(df):,} samples")

    return predictions


def load_tree_models(
    checkpoint_dir: str | Path, model_type: str = "xgboost"
) -> dict[str, Any]:
    """Load saved per-city tree models from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing model files.
        model_type: 'xgboost' or 'lightgbm'.

    Returns:
        Dict mapping city_id → loaded model.
    """
    checkpoint_dir = Path(checkpoint_dir)
    models = {}

    if model_type == "xgboost":
        import xgboost as xgb

        for path in sorted(checkpoint_dir.glob("xgboost_*.json")):
            city = path.stem.replace("xgboost_", "")
            model = xgb.XGBRegressor()
            model.load_model(str(path))
            models[city] = model

    elif model_type == "lightgbm":
        import lightgbm as lgb

        for path in sorted(checkpoint_dir.glob("lightgbm_*.txt")):
            city = path.stem.replace("lightgbm_", "")
            model = lgb.Booster(model_file=str(path))
            models[city] = model

    logger.info(f"Loaded {len(models)} {model_type} models from {checkpoint_dir}")
    return models
