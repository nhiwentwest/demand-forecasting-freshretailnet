"""
Ensemble Latent Demand Recovery using TimesNet + iTransformer.

Recovers true demand from censored sales data during stockout periods.
Uses inverse-error weighted ensemble of two imputation models.

References:
    - FreshRetailNet-50K (arXiv:2505.16319): Two-stage framework, TimesNet #1
    - MIE-TS (ACM 2022): Multiple imputation ensemble for time series
    - arXiv:2010.08158: Optimal inverse-error weighting for ensemble
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prepare_hourly_recovery_data(df: pd.DataFrame) -> dict:
    """Prepare hourly-level data for demand recovery models.

    Extracts hourly sales and stock status arrays from the daily records,
    creating the masked input required by TimesNet/iTransformer.

    Formulation from FRN-50K paper:
        ŷ = f_θ(y ⊙ s, p, w, c, s)
    where y = observed hourly sales, s = stock mask, p/w/c = covariates.

    Args:
        df: DataFrame with 'hours_sale', 'hours_stock_status', and covariates.

    Returns:
        Dict with keys:
            - 'hourly_sales': array of shape (n_samples, 17)
            - 'stock_mask': binary array of shape (n_samples, 17)
            - 'covariates': dict of covariate arrays
            - 'metadata': series_id, dt for each sample
    """
    from src.data.preprocessing import parse_hours_sale, parse_hours_stock_status

    logger.info("Preparing hourly data for demand recovery...")

    hourly_sales = np.stack(df["hours_sale"].apply(parse_hours_sale).values)
    stock_mask = np.stack(
        df["hours_stock_status"].apply(parse_hours_stock_status).values
    )

    # Masked sales: observed where in-stock, 0 where stockout
    masked_sales = hourly_sales * stock_mask

    covariates = {
        "discount": df["discount"].values,
        "activity_flag": df["activity_flag"].values,
        "holiday_flag": df["holiday_flag"].values,
        "precpt": df["precpt"].values,
        "avg_temperature": df["avg_temperature"].values,
        "avg_humidity": df["avg_humidity"].values,
        "day_of_week": df["dt"].dt.dayofweek.values,
    }

    metadata = df[["series_id", "dt"]].reset_index(drop=True)

    stockout_hours = (stock_mask == 0).sum()
    total_hours = stock_mask.size
    logger.info(
        f"Hourly data prepared: {len(df):,} days × 17 hours, "
        f"{stockout_hours:,}/{total_hours:,} stockout hours "
        f"({stockout_hours / total_hours * 100:.1f}%)"
    )

    return {
        "hourly_sales": hourly_sales,
        "masked_sales": masked_sales,
        "stock_mask": stock_mask,
        "covariates": covariates,
        "metadata": metadata,
    }


def build_neuralforecast_df(hourly_data: dict) -> pd.DataFrame:
    """Convert hourly data into NeuralForecast-compatible long format.

    NeuralForecast expects: unique_id, ds, y columns.

    Args:
        hourly_data: Output from prepare_hourly_recovery_data().

    Returns:
        DataFrame in NeuralForecast format.
    """
    records = []
    metadata = hourly_data["metadata"]
    hourly_sales = hourly_data["masked_sales"]  # Use masked (0 where stockout)
    stock_mask = hourly_data["stock_mask"]

    for idx in range(len(metadata)):
        series_id = metadata.iloc[idx]["series_id"]
        date = metadata.iloc[idx]["dt"]

        for hour_idx in range(17):
            actual_hour = hour_idx + 6  # 6AM to 10PM
            records.append(
                {
                    "unique_id": series_id,
                    "ds": pd.Timestamp(date) + pd.Timedelta(hours=actual_hour),
                    "y": float(hourly_sales[idx, hour_idx]),
                    "stock_mask": int(stock_mask[idx, hour_idx]),
                }
            )

    nf_df = pd.DataFrame(records)
    logger.info(f"NeuralForecast DataFrame: {len(nf_df):,} hourly records")
    return nf_df


def train_timesnet(nf_df: pd.DataFrame, config: dict, checkpoint_dir: str | Path | None = None):
    """Train TimesNet for demand recovery (imputation).

    TimesNet captures multi-periodicity (intraday 17h + weekly 7d)
    via 2D temporal variation modeling.

    Reference: FRN-50K paper — TimesNet WAPE = 27.62% (Task 1 #1).

    Args:
        nf_df: NeuralForecast-format DataFrame.
        config: TimesNet config from experiment.yaml.
        checkpoint_dir: Directory for PyTorch Lightning checkpoints.

    Returns:
        Fitted NeuralForecast object with TimesNet.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TimesNet
    from neuralforecast.losses.pytorch import MAE
    import pytorch_lightning as pl

    callbacks = []
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                save_top_k=2,
                monitor="valid_loss",
                every_n_epochs=1,
                save_last=True,
                filename="timesnet-{epoch}-{valid_loss:.4f}",
            )
        )

    tc = config.get("stage1_recovery", {}).get("timesnet", {})

    model = TimesNet(
        h=tc.get("horizon", 119),
        input_size=tc.get("input_size", 238),
        hidden_size=tc.get("hidden_size", 64),
        conv_hidden_size=tc.get("conv_hidden_size", 32),
        num_kernels=tc.get("num_kernels", 6),
        encoder_layers=tc.get("encoder_layers", 2),
        loss=MAE(),
        max_steps=tc.get("max_steps", 500),
        batch_size=tc.get("batch_size", 32),
        learning_rate=tc.get("learning_rate", 1e-3),
        scaler_type=tc.get("scaler_type", "robust"),
        trainer_kwargs={"callbacks": callbacks} if callbacks else {},
    )

    nf = NeuralForecast(models=[model], freq="h")

    logger.info("Training TimesNet for demand recovery...")
    nf.fit(df=nf_df)
    logger.info("TimesNet training complete")

    return nf


def train_itransformer(nf_df: pd.DataFrame, config: dict, checkpoint_dir: str | Path | None = None):
    """Train iTransformer for demand recovery.

    iTransformer uses channel-first attention, good for fusing
    promotion + weather covariates with temporal patterns.

    Reference: FRN-50K paper — iTransformer WAPE = 33.30% (Task 1 #2).

    Args:
        nf_df: NeuralForecast-format DataFrame.
        config: iTransformer config from experiment.yaml.
        checkpoint_dir: Directory for checkpoints.

    Returns:
        Fitted NeuralForecast object with iTransformer.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import iTransformer
    from neuralforecast.losses.pytorch import MAE
    import pytorch_lightning as pl

    callbacks = []
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                save_top_k=2,
                monitor="valid_loss",
                every_n_epochs=1,
                save_last=True,
                filename="itransformer-{epoch}-{valid_loss:.4f}",
            )
        )

    ic = config.get("stage1_recovery", {}).get("itransformer", {})

    model = iTransformer(
        h=ic.get("horizon", 119),
        input_size=ic.get("input_size", 238),
        hidden_size=ic.get("hidden_size", 64),
        n_heads=ic.get("n_heads", 4),
        e_layers=ic.get("e_layers", 2),
        loss=MAE(),
        max_steps=ic.get("max_steps", 500),
        batch_size=ic.get("batch_size", 32),
        learning_rate=ic.get("learning_rate", 1e-3),
        trainer_kwargs={"callbacks": callbacks} if callbacks else {},
    )

    nf = NeuralForecast(models=[model], freq="h")

    logger.info("Training iTransformer for demand recovery...")
    nf.fit(df=nf_df)
    logger.info("iTransformer training complete")

    return nf


def ensemble_recovery(
    pred_timesnet: np.ndarray,
    pred_itransformer: np.ndarray,
    wape_timesnet: float | None = None,
    wape_itransformer: float | None = None,
) -> np.ndarray:
    """Combine recovery predictions via inverse-error weighted average.

    If WAPE values are provided, uses inverse-WAPE weighting (optimal per
    arXiv:2010.08158). Otherwise, uses equal weights.

    References:
        - MIE-TS (ACM 2022): ensemble imputation reduces variance
        - arXiv:2010.08158: inverse-error weighting is optimal

    Args:
        pred_timesnet: TimesNet predictions.
        pred_itransformer: iTransformer predictions.
        wape_timesnet: Validation WAPE for TimesNet (lower = better).
        wape_itransformer: Validation WAPE for iTransformer.

    Returns:
        Ensemble-recovered demand array.
    """
    if wape_timesnet is not None and wape_itransformer is not None:
        # Inverse-error weighting
        inv_t = 1.0 / max(wape_timesnet, 1e-8)
        inv_i = 1.0 / max(wape_itransformer, 1e-8)
        w_t = inv_t / (inv_t + inv_i)
        w_i = 1.0 - w_t
        logger.info(
            f"Ensemble weights (inverse-WAPE): TimesNet={w_t:.3f}, "
            f"iTransformer={w_i:.3f}"
        )
    else:
        w_t, w_i = 0.5, 0.5
        logger.info("Ensemble weights: equal (0.5, 0.5)")

    recovered = w_t * pred_timesnet + w_i * pred_itransformer
    return recovered


def merge_observed_recovered(
    observed_hourly: np.ndarray,
    recovered_hourly: np.ndarray,
    stock_mask: np.ndarray,
) -> np.ndarray:
    """Merge observed sales (in-stock hours) with recovered demand (stockout hours).

    Keep observed values where stock was available; use model predictions
    where stockout occurred.

    Args:
        observed_hourly: Raw hourly sales array.
        recovered_hourly: Model-predicted hourly demand.
        stock_mask: Binary mask (1 = in-stock, 0 = stockout).

    Returns:
        Final hourly demand array (observed + recovered).
    """
    final = np.where(stock_mask == 1, observed_hourly, recovered_hourly)

    n_recovered = int((stock_mask == 0).sum())
    n_total = stock_mask.size
    logger.info(
        f"Merged: {n_recovered:,}/{n_total:,} hours recovered "
        f"({n_recovered / n_total * 100:.1f}%)"
    )

    return final


def aggregate_hourly_to_daily(
    hourly_demand: np.ndarray, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate hourly recovered demand to daily totals.

    Args:
        hourly_demand: Array of shape (n_days, 17) — final hourly demand.
        metadata: DataFrame with series_id, dt for each day.

    Returns:
        DataFrame with daily recovered demand.
    """
    daily_demand = hourly_demand.reshape(-1, 17).sum(axis=1)

    result = metadata.copy()
    result["recovered_demand"] = daily_demand

    logger.info(
        f"Aggregated to daily: {len(result):,} records, "
        f"mean demand={daily_demand.mean():.2f}"
    )

    return result


def save_recovery_results(
    recovered_df: pd.DataFrame,
    metrics: dict,
    output_dir: str | Path,
):
    """Save recovery results and metrics for Stage 2 consumption.

    Args:
        recovered_df: DataFrame with recovered_demand column.
        metrics: Task 1 evaluation metrics.
        output_dir: Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save recovered demand
    parquet_path = output_dir / "recovered_demand.parquet"
    recovered_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved recovered demand to {parquet_path}")

    # Save metrics
    metrics_path = output_dir / "task1_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved Task 1 metrics to {metrics_path}")
