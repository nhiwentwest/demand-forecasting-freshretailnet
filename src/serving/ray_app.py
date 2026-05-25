"""
Ray Serve deployment for demand forecasting inference.

Loads trained XGBoost + LightGBM models and exposes REST API
for real-time demand forecasting.

Usage:
    # Start serving (from project root, after training)
    python -m src.serving.ray_app --models-dir ./outputs/checkpoints

    # Or with serve CLI
    serve run src.serving.ray_app:app
"""

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Google Drive checkpoint URLs ──
# Note: You will need to upload your newly trained WPE=0 checkpoints to Google Drive
# and update these IDs if you want the API to auto-download them.
GDRIVE_CHECKPOINTS = {
    "knn_global.joblib": "dummy_id_knn_replace_me_123",
    "lightgbm_global.txt": "dummy_id_lgb_replace_me_123",
    "random_forest_global.joblib": "dummy_id_rf_replace_me_123",
    "ridge_standalone.joblib": "dummy_id_ridge_replace_me_123",
    "stacking_ridge.joblib": "dummy_id_stacking_replace_me_123",
    "xgboost_global.json": "dummy_id_xgb_replace_me_123",
}


def download_checkpoints(models_dir: str):
    """Download missing checkpoints from Google Drive."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    missing = {name: fid for name, fid in GDRIVE_CHECKPOINTS.items()
               if not (models_path / name).exists()}

    if not missing:
        logger.info("All checkpoints found locally.")
        return

    logger.info(f"Missing {len(missing)} checkpoints — downloading from Google Drive...")

    try:
        import gdown
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    for name, file_id in missing.items():
        dest = str(models_path / name)
        url = f"https://drive.google.com/uc?id={file_id}"
        logger.info(f"  Downloading {name}...")
        try:
            gdown.download(url, dest, quiet=False)
            logger.info(f"  ✅ {name} downloaded")
        except Exception as e:
            logger.error(f"  ❌ Failed to download {name}: {e}")

    logger.info("Checkpoint download complete.")

# ── Feature engineering for inference ──

FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_14", "rolling_std_28",
    "stockout_ratio_7d", "stockout_ratio_14d", "stockout_ratio_28d",
    "stockout_ratio",
    "is_stockout", "stockout_hours",
    "series_scale",
    "day_of_week", "week_of_year", "month", "is_weekend", "day_of_month",
    "sin_7d_h1", "cos_7d_h1", "sin_7d_h2", "cos_7d_h2",
    "sin_7d_h3", "cos_7d_h3",
    "sin_30d_h1", "cos_30d_h1", "sin_30d_h2", "cos_30d_h2",
    "sin_30d_h3", "cos_30d_h3",
    "discount", "activity_flag", "holiday_flag",
    "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
    "first_category_id_te", "city_id_te",
    "city_id",
]

OPERATING_HOURS = 16


def compute_features_from_history(history: list[dict], forecast_date: dict) -> np.ndarray:
    """Compute the 45 features from recent sales history + forecast day context.

    Args:
        history: List of dicts with keys: sale_amount, stockout_ratio,
                 is_stockout, stockout_hours. Must be sorted by date ascending,
                 last entry = most recent day. Minimum 28 entries recommended.
        forecast_date: Dict with keys: day_of_week, week_of_year, month,
                       is_weekend, day_of_month, discount, activity_flag,
                       holiday_flag, precpt, avg_temperature, avg_humidity,
                       avg_wind_level, city_id, first_category_id_te,
                       city_id_te.

    Returns:
        Feature vector of shape (45,).
    """
    sales = [h.get("sale_amount", 0.0) for h in history]
    stockout_ratios = [h.get("stockout_ratio", 0.0) for h in history]

    # Pad if history is too short
    while len(sales) < 28:
        sales.insert(0, np.mean(sales) if sales else 0.0)
        stockout_ratios.insert(0, 0.0)

    s = np.array(sales, dtype=np.float64)

    # Lags (relative to last day in history)
    lag_1 = s[-1]
    lag_7 = s[-7] if len(s) >= 7 else s[0]
    lag_14 = s[-14] if len(s) >= 14 else s[0]
    lag_21 = s[-21] if len(s) >= 21 else s[0]
    lag_28 = s[-28] if len(s) >= 28 else s[0]

    # Rolling stats
    rolling_mean_7 = np.mean(s[-7:])
    rolling_mean_14 = np.mean(s[-14:])
    rolling_mean_28 = np.mean(s[-28:])
    rolling_std_7 = np.std(s[-7:])
    rolling_std_14 = np.std(s[-14:])
    rolling_std_28 = np.std(s[-28:])

    # Stockout rolling averages
    sr = np.array(stockout_ratios, dtype=np.float64)
    stockout_ratio_7d = np.mean(sr[-7:])
    stockout_ratio_14d = np.mean(sr[-14:])
    stockout_ratio_28d = np.mean(sr[-28:])

    # Latest stockout info
    last_h = history[-1] if history else {}
    stockout_ratio = last_h.get("stockout_ratio", 0.0)
    is_stockout = last_h.get("is_stockout", 0)
    stockout_hours = last_h.get("stockout_hours", 0)

    # Series scale
    series_scale = rolling_mean_28 if rolling_mean_28 > 0 else 1.0

    # Calendar features from forecast_date
    dow = forecast_date.get("day_of_week", 0)
    woy = forecast_date.get("week_of_year", 1)
    month = forecast_date.get("month", 1)
    is_weekend = forecast_date.get("is_weekend", 0)
    dom = forecast_date.get("day_of_month", 1)

    # Fourier features
    fourier = []
    for period in [7, 30]:
        for k in [1, 2, 3]:
            fourier.append(np.sin(2 * np.pi * k * dow / period))  # sin
            fourier.append(np.cos(2 * np.pi * k * dow / period))  # cos

    # Context features from forecast_date
    discount = forecast_date.get("discount", 0.0)
    activity_flag = forecast_date.get("activity_flag", 0)
    holiday_flag = forecast_date.get("holiday_flag", 0)
    precpt = forecast_date.get("precpt", 0.0)
    avg_temp = forecast_date.get("avg_temperature", 25.0)
    avg_hum = forecast_date.get("avg_humidity", 70.0)
    avg_wind = forecast_date.get("avg_wind_level", 2.0)

    # Target encoding (pre-computed, passed in)
    first_cat_te = forecast_date.get("first_category_id_te", 1.0)
    city_id_te = forecast_date.get("city_id_te", 1.0)
    city_id = forecast_date.get("city_id", 0)

    features = [
        lag_1, lag_7, lag_14, lag_21, lag_28,
        rolling_mean_7, rolling_mean_14, rolling_mean_28,
        rolling_std_7, rolling_std_14, rolling_std_28,
        stockout_ratio_7d, stockout_ratio_14d, stockout_ratio_28d,
        stockout_ratio,
        is_stockout, stockout_hours,
        series_scale,
        dow, woy, month, is_weekend, dom,
        *fourier,
        discount, activity_flag, holiday_flag,
        precpt, avg_temp, avg_hum, avg_wind,
        first_cat_te, city_id_te,
        city_id,
    ]

    return np.array(features, dtype=np.float64)


def create_app(models_dir: str = "./outputs/checkpoints"):
    """Create Ray Serve FastAPI application with real model inference.

    Args:
        models_dir: Directory containing saved model checkpoints
                    (xgboost_global.json, lightgbm_global.txt, etc).
                    If not found locally, auto-downloads from Google Drive.
    """
    # Auto-download checkpoints if missing
    download_checkpoints(models_dir)

    try:
        from ray import serve
    except ImportError:
        raise ImportError("ray[serve] is required. Install with: pip install 'ray[serve]'")
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import starlette.requests

    app = FastAPI(
        title="Demand Forecasting API",
        description=(
            "FreshRetailNet-50K demand forecasting service.\n\n"
            "Pipeline: TimesNet+PatchTST recovery → XGBoost+LightGBM → Stacking Ensemble"
        ),
        version="1.0.0",
    )

    # ── Request/Response schemas ──

    class SalesDay(BaseModel):
        """One day of historical sales data."""
        sale_amount: float = Field(..., description="Units sold on this day")
        stockout_ratio: float = Field(0.0, description="Fraction of operating hours out of stock (0-1)")
        is_stockout: int = Field(0, description="1 if stockout occurred, 0 otherwise")
        stockout_hours: int = Field(0, description="Number of hours out of stock")

    class ForecastDay(BaseModel):
        """Context for the day to forecast."""
        day_of_week: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
        week_of_year: int = Field(..., ge=1, le=53)
        month: int = Field(..., ge=1, le=12)
        is_weekend: int = Field(0, ge=0, le=1)
        day_of_month: int = Field(..., ge=1, le=31)
        discount: float = Field(0.0, description="Discount amount")
        activity_flag: int = Field(0, description="1 if promotion active")
        holiday_flag: int = Field(0, description="1 if holiday")
        precpt: float = Field(0.0, description="Precipitation")
        avg_temperature: float = Field(25.0)
        avg_humidity: float = Field(70.0)
        avg_wind_level: float = Field(2.0)
        city_id: int = Field(0)
        first_category_id_te: float = Field(1.0, description="Target-encoded category")
        city_id_te: float = Field(1.0, description="Target-encoded city")

    class ForecastRequest(BaseModel):
        """Request body for demand forecast."""
        history: list[SalesDay] = Field(
            ...,
            min_length=7,
            description="Recent sales history (min 7 days, recommend 28). Oldest first.",
        )
        forecast_days: list[ForecastDay] = Field(
            ...,
            min_length=1,
            description="One or more days to forecast.",
        )

    class PredictionResult(BaseModel):
        day_of_week: int
        month: int
        demand_xgboost: float
        demand_lightgbm: float
        demand_ensemble: float

    class ForecastResponse(BaseModel):
        predictions: list[PredictionResult]
        model_info: dict
        n_history_days: int

    class HealthResponse(BaseModel):
        status: str
        models_loaded: list[str]

    # ── Serve deployment ──

    @serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
    @serve.ingress(app)
    class ForecastService:
        def __init__(self):
            self.models_dir = Path(models_dir)
            self.models = {}
            self._load_models()

        def _load_models(self):
            """Load trained model checkpoints."""
            logger.info(f"Loading models from {self.models_dir}...")

            # XGBoost global
            xgb_path = self.models_dir / "xgboost_global.json"
            if xgb_path.exists():
                import xgboost as xgb
                model = xgb.XGBRegressor()
                model.load_model(str(xgb_path))
                self.models["xgboost"] = model
                logger.info(f"  Loaded XGBoost from {xgb_path}")

            # LightGBM global
            lgb_path = self.models_dir / "lightgbm_global.txt"
            if lgb_path.exists():
                import lightgbm as lgb
                model = lgb.Booster(model_file=str(lgb_path))
                self.models["lightgbm"] = model
                logger.info(f"  Loaded LightGBM from {lgb_path}")

            # Random Forest
            rf_path = self.models_dir / "random_forest_global.joblib"
            if rf_path.exists():
                import joblib
                self.models["random_forest"] = joblib.load(rf_path)
                logger.info(f"  Loaded Random Forest from {rf_path}")

            # Ridge Regression (standalone)
            ridge_sa_path = self.models_dir / "ridge_standalone.joblib"
            if ridge_sa_path.exists():
                import joblib
                ridge_data = joblib.load(ridge_sa_path)
                self.models["ridge"] = ridge_data  # {"model": ..., "scaler": ...}
                logger.info(f"  Loaded Ridge standalone from {ridge_sa_path}")

            # kNN
            knn_path = self.models_dir / "knn_global.joblib"
            if knn_path.exists():
                import joblib
                self.models["knn"] = joblib.load(knn_path)
                logger.info(f"  Loaded kNN from {knn_path}")

            # Stacking Ridge
            ridge_path = self.models_dir / "stacking_ridge.joblib"
            if ridge_path.exists():
                import joblib
                self.models["stacking"] = joblib.load(ridge_path)
                logger.info(f"  Loaded Stacking Ridge from {ridge_path}")

            if not self.models:
                logger.warning("No models found! Inference will fail.")
            else:
                logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

        @app.get("/health", response_model=HealthResponse)
        async def health(self):
            """Health check — returns loaded model names."""
            return HealthResponse(
                status="healthy" if self.models else "no_models",
                models_loaded=list(self.models.keys()),
            )

        @app.post("/forecast")
        async def forecast(self, raw_request: starlette.requests.Request):
            """Generate demand forecast from recent sales history."""
            if not self.models:
                raise HTTPException(status_code=503, detail="No models loaded")

            # Parse raw JSON to avoid Pydantic+Ray Serve serialization issues
            import json as _json
            body = await raw_request.body()
            data = _json.loads(body)
            history_list = data.get("history", [])
            forecast_list = data.get("forecast_days", [])

            history_dicts = []
            for h in history_list:
                if hasattr(h, 'model_dump'):
                    history_dicts.append(h.model_dump())
                elif hasattr(h, 'dict'):
                    history_dicts.append(h.dict())
                elif isinstance(h, dict):
                    history_dicts.append(h)
                else:
                    history_dicts.append({"sale_amount": float(h.sale_amount),
                                          "stockout_ratio": float(getattr(h, 'stockout_ratio', 0)),
                                          "is_stockout": int(getattr(h, 'is_stockout', 0)),
                                          "stockout_hours": int(getattr(h, 'stockout_hours', 0))})

            predictions = []

            for fday in forecast_list:
                if hasattr(fday, 'model_dump'):
                    fday_dict = fday.model_dump()
                elif hasattr(fday, 'dict'):
                    fday_dict = fday.dict()
                elif isinstance(fday, dict):
                    fday_dict = fday
                else:
                    fday_dict = {k: getattr(fday, k, 0) for k in [
                        "day_of_week", "week_of_year", "month", "is_weekend",
                        "day_of_month", "discount", "activity_flag", "holiday_flag",
                        "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
                        "city_id", "first_category_id_te", "city_id_te"]}

                features = compute_features_from_history(history_dicts, fday_dict)
                X = features.reshape(1, -1)

                # XGBoost prediction
                pred_xgb = 0.0
                if "xgboost" in self.models:
                    pred_xgb = float(self.models["xgboost"].predict(X)[0])
                    pred_xgb = max(pred_xgb, 0.0)

                # LightGBM prediction
                pred_lgb = 0.0
                if "lightgbm" in self.models:
                    pred_lgb = float(self.models["lightgbm"].predict(X)[0])
                    pred_lgb = max(pred_lgb, 0.0)

                # Random Forest prediction
                pred_rf = 0.0
                if "random_forest" in self.models:
                    pred_rf = float(self.models["random_forest"].predict(X)[0])
                    pred_rf = max(pred_rf, 0.0)

                # Stacking ensemble (meta-learner expects [lgb, rf, xgb] order)
                if "stacking" in self.models and all(k in self.models for k in ["xgboost", "lightgbm", "random_forest"]):
                    meta_X = np.array([[pred_lgb, pred_rf, pred_xgb]])
                    pred_ens = float(self.models["stacking"].predict(meta_X)[0])
                    pred_ens = max(pred_ens, 0.0)
                else:
                    available = [p for p in [pred_xgb, pred_lgb, pred_rf] if p > 0]
                    pred_ens = float(np.mean(available)) if available else 0.0

                predictions.append({
                    "day_of_week": fday_dict.get("day_of_week", 0),
                    "month": fday_dict.get("month", 1),
                    "demand_xgboost": round(pred_xgb, 4),
                    "demand_lightgbm": round(pred_lgb, 4),
                    "demand_ensemble": round(pred_ens, 4),
                })

            return {
                "predictions": predictions,
                "model_info": {
                    "models": list(self.models.keys()),
                    "pipeline": "TimesNet+PatchTST recovery → XGBoost+LightGBM → Ridge Stacking",
                    "features_used": len(FEATURE_COLS),
                },
                "n_history_days": len(history_dicts),
            }

    deployment = ForecastService.bind()
    return deployment


# ── Direct execution ──
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start demand forecasting API server")
    parser.add_argument("--models-dir", default="./outputs/checkpoints",
                        help="Path to model checkpoints directory")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import ray
    from ray import serve

    ray.init(ignore_reinit_error=True)
    serve.start(http_options={"host": "0.0.0.0", "port": args.port})
    deployment = create_app(models_dir=args.models_dir)
    serve.run(deployment)
    print(f"✅ Serving at http://0.0.0.0:{args.port}")
    print(f"📖 API docs at http://0.0.0.0:{args.port}/docs")
    print(f"❤️  Health check: http://0.0.0.0:{args.port}/health")

    import signal
    signal.sigwait([signal.SIGINT, signal.SIGTERM])
