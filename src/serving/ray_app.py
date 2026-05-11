"""
Ray Serve deployment for demand forecasting inference.

Exposes REST API for consumer consumption per assignment requirement.

Usage:
    # Start serving
    serve run src.serving.ray_app:deployment

    # Or programmatically
    python -m src.serving.ray_app
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def create_app(models_dir: str = "./models"):
    """Create Ray Serve FastAPI application.

    Args:
        models_dir: Directory containing saved model checkpoints.

    Returns:
        Ray Serve deployment handle.
    """
    from ray import serve
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(
        title="Demand Forecasting API",
        description="FreshRetailNet-50K demand forecasting with ensemble recovery + stacking",
        version="1.0.0",
    )

    # ── Request/Response schemas ──

    class ForecastRequest(BaseModel):
        city_id: str
        store_id: Optional[str] = None
        product_ids: list[str] = []
        horizon_days: int = 7

    class DayForecast(BaseModel):
        date: str
        demand: float
        ci_low: float
        ci_high: float

    class ProductForecast(BaseModel):
        product_id: str
        predictions: list[DayForecast]
        model_used: str

    class ForecastResponse(BaseModel):
        forecasts: list[ProductForecast]
        pipeline: str = "TimesNet+iTransformer → XGBoost+LightGBM+TFT+N-HiTS → Ridge"

    class ModelInfo(BaseModel):
        name: str
        type: str
        cities: list[str]

    class HealthResponse(BaseModel):
        status: str
        models_loaded: int

    # ── Serve deployment ──

    @serve.deployment(num_replicas=1)
    @serve.ingress(app)
    class ForecastService:
        def __init__(self):
            self.models_dir = Path(models_dir)
            self.models = {}
            self._load_models()

        def _load_models(self):
            """Load all available models from checkpoint directory."""
            logger.info(f"Loading models from {self.models_dir}...")

            # Load XGBoost models
            xgb_dir = self.models_dir / "xgboost"
            if xgb_dir.exists():
                import xgboost as xgb

                for path in sorted(xgb_dir.glob("*.json")):
                    city = path.stem.replace("xgboost_", "")
                    model = xgb.XGBRegressor()
                    model.load_model(str(path))
                    self.models[f"xgboost_{city}"] = model

            # Load stacking meta-learner
            stacking_path = self.models_dir / "stacking" / "stacking_ridge.joblib"
            if stacking_path.exists():
                import joblib

                self.models["stacking"] = joblib.load(stacking_path)

            logger.info(f"Loaded {len(self.models)} models")

        @app.get("/health", response_model=HealthResponse)
        async def health(self):
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                models_loaded=len(self.models),
            )

        @app.get("/models")
        async def list_models(self):
            """List all loaded models and their capabilities."""
            info = []
            for name in self.models:
                model_type = name.split("_")[0] if "_" in name else name
                info.append(
                    {
                        "name": name,
                        "type": model_type,
                    }
                )
            return {"models": info}

        @app.post("/forecast", response_model=ForecastResponse)
        async def forecast(self, request: ForecastRequest):
            """Generate demand forecast for given products.

            Pipeline: Stage 1 recovery → Stage 2 stacking forecast
            """
            forecasts = []

            for product_id in request.product_ids:
                # Use best available model for the city
                model_key = f"xgboost_{request.city_id}"
                model_used = model_key if model_key in self.models else "fallback"

                # Generate predictions (placeholder — actual implementation
                # would load recent data and run through the pipeline)
                predictions = []
                import datetime

                base_date = datetime.date.today()
                for d in range(request.horizon_days):
                    pred_date = base_date + datetime.timedelta(days=d + 1)
                    # In production, this would be actual model inference
                    demand = 0.0
                    predictions.append(
                        DayForecast(
                            date=pred_date.isoformat(),
                            demand=round(demand, 2),
                            ci_low=round(demand * 0.75, 2),
                            ci_high=round(demand * 1.25, 2),
                        )
                    )

                forecasts.append(
                    ProductForecast(
                        product_id=product_id,
                        predictions=predictions,
                        model_used=model_used,
                    )
                )

            return ForecastResponse(forecasts=forecasts)

    deployment = ForecastService.bind()
    return deployment


# Allow running directly
if __name__ == "__main__":
    import ray
    from ray import serve

    ray.init()
    deployment = create_app()
    serve.run(deployment, host="0.0.0.0", port=8000)
    print("Serving at http://0.0.0.0:8000")
    print("Docs at http://0.0.0.0:8000/docs")
