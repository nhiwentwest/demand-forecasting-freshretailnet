"""
Stacking ensemble: Ridge meta-learner on base model predictions.

Combines heterogeneous models (tree + deep) for best-of-both-worlds forecasting.

References:
    - arXiv:2010.08158: Ridge meta-learner optimal for <10 base models
    - arXiv:2025 (stock forecasting): Stacking R²=0.97–0.99
    - Chalmers 2024: Ensemble needed because TFT inconsistent alone
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def train_stacking_meta(
    base_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    alpha: float = 1.0,
    cv_splits: int = 3,
) -> tuple[Ridge, dict]:
    """Train Ridge meta-learner on out-of-fold base predictions.

    Uses TimeSeriesSplit CV to respect temporal ordering and avoid leakage.
    Ridge is preferred over Lasso when base models < 10 (arXiv:2010.08158).

    Args:
        base_predictions: Dict of {model_name: prediction_array}.
        y_true: True target values.
        alpha: Ridge regularization strength.
        cv_splits: Number of time-series CV splits.

    Returns:
        Tuple of (fitted Ridge meta-learner, CV metrics dict).
    """
    # Stack base predictions as features
    model_names = sorted(base_predictions.keys())
    X_meta = np.column_stack([base_predictions[name] for name in model_names])

    # Remove NaN rows
    valid = ~np.isnan(X_meta).any(axis=1) & ~np.isnan(y_true)
    X_meta_valid = X_meta[valid]
    y_valid = y_true[valid]

    logger.info(
        f"Stacking meta-learner: {len(model_names)} base models, "
        f"{len(y_valid):,} valid samples (from {len(y_true):,})"
    )

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    oof_preds = np.full(len(y_valid), np.nan)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_meta_valid)):
        meta_model = Ridge(alpha=alpha)
        meta_model.fit(X_meta_valid[train_idx], y_valid[train_idx])
        fold_preds = meta_model.predict(X_meta_valid[val_idx])
        oof_preds[val_idx] = fold_preds

        fold_rmse = np.sqrt(np.mean((y_valid[val_idx] - fold_preds) ** 2))
        cv_scores.append(fold_rmse)
        logger.info(f"  Fold {fold + 1}/{cv_splits}: RMSE = {fold_rmse:.4f}")

    # Final fit on all data
    final_model = Ridge(alpha=alpha)
    final_model.fit(X_meta_valid, y_valid)

    # Log model weights
    weights = dict(zip(model_names, final_model.coef_))
    logger.info(f"Meta-learner weights: {weights}")
    logger.info(f"Meta-learner intercept: {final_model.intercept_:.4f}")
    logger.info(f"CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    metrics = {
        "cv_rmse_mean": float(np.mean(cv_scores)),
        "cv_rmse_std": float(np.std(cv_scores)),
        "cv_rmse_per_fold": [float(s) for s in cv_scores],
        "weights": {k: float(v) for k, v in weights.items()},
        "intercept": float(final_model.intercept_),
        "base_models": model_names,
    }

    return final_model, metrics


def predict_stacking(
    meta_model: Ridge,
    base_predictions: dict[str, np.ndarray],
    model_names: list[str] | None = None,
) -> np.ndarray:
    """Generate stacking ensemble predictions.

    Args:
        meta_model: Fitted Ridge meta-learner.
        base_predictions: Dict of {model_name: prediction_array}.
        model_names: Ordered model names (must match training order).

    Returns:
        Stacked ensemble predictions.
    """
    if model_names is None:
        model_names = sorted(base_predictions.keys())

    X_meta = np.column_stack([base_predictions[name] for name in model_names])
    predictions = meta_model.predict(X_meta)

    return predictions


def generate_probabilistic_forecast(
    stacking_pred: np.ndarray,
    tft_quantiles: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Generate probabilistic forecast: point + confidence intervals.

    Uses stacking point forecast as P50, scales TFT quantile spread
    to match stacking magnitude for calibrated intervals.

    Args:
        stacking_pred: Point predictions from stacking ensemble.
        tft_quantiles: Dict with 'q10', 'q50', 'q90' from TFT.

    Returns:
        Dict with 'q10', 'q50', 'q90' probabilistic forecasts.
    """
    result = {"q50": stacking_pred}

    if tft_quantiles is not None and "q10" in tft_quantiles and "q90" in tft_quantiles:
        # Scale TFT spread to stacking point forecast
        tft_q50 = tft_quantiles.get("q50", stacking_pred)
        spread_low = np.where(
            tft_q50 > 0,
            tft_quantiles["q10"] / tft_q50,
            0.8,  # Default 20% spread if TFT q50 is zero
        )
        spread_high = np.where(
            tft_q50 > 0,
            tft_quantiles["q90"] / tft_q50,
            1.2,
        )

        result["q10"] = stacking_pred * spread_low
        result["q90"] = stacking_pred * spread_high
    else:
        # Fallback: simple symmetric intervals based on prediction magnitude
        result["q10"] = stacking_pred * 0.75
        result["q90"] = stacking_pred * 1.25

    return result


def save_stacking_model(
    meta_model: Ridge,
    metrics: dict,
    output_dir: str | Path,
):
    """Save stacking meta-learner and metrics.

    Args:
        meta_model: Fitted Ridge model.
        metrics: Training metrics.
        output_dir: Output directory.
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(meta_model, output_dir / "stacking_ridge.joblib")

    with open(output_dir / "stacking_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Stacking model saved to {output_dir}")


def load_stacking_model(model_dir: str | Path) -> Ridge:
    """Load saved stacking meta-learner.

    Args:
        model_dir: Directory containing stacking_ridge.joblib.

    Returns:
        Fitted Ridge model.
    """
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "stacking_ridge.joblib")
    logger.info(f"Loaded stacking model from {model_dir}")
    return model
