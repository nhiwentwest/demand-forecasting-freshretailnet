"""
Deep learning forecasting models: TFT + N-HiTS.

Global models with per-series group IDs.
Designed for Kaggle GPU training with checkpointing.

References:
    - TFT (arXiv:2511.00552): R²=0.9875 on Walmart, probabilistic P10/P50/P90
    - N-HiTS (arXiv:2201.12886): 50x faster than Transformers, multi-periodicity
    - FRN-50K paper: TFT #1 for Task 2 forecasting
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# TFT — Temporal Fusion Transformer [arXiv:2511.00552]
# ============================================================


def prepare_tft_dataset(
    df: pd.DataFrame,
    config: dict,
    target_col: str = "recovered_demand",
):
    """Prepare pytorch-forecasting TimeSeriesDataSet for TFT.

    TFT requires explicit classification of covariates:
        - static_categoricals: fixed per series (city, category)
        - time_varying_known_*: known in advance (calendar, fourier, holidays)
        - time_varying_unknown_*: only known in past (demand, rolling stats)

    Args:
        df: Feature-engineered DataFrame.
        config: Experiment config.
        target_col: Target column name.

    Returns:
        Tuple of (training_dataset, validation_dataset, dataloader_kwargs).
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    tft_cfg = config.get("stage2_forecasting", {}).get("tft", {})

    # Create integer time index per series
    df = df.sort_values(["series_id", "dt"]).copy()
    df["time_idx"] = df.groupby("series_id").cumcount()

    # Split by time_idx for TFT
    max_time = df["time_idx"].max()
    test_days = config.get("data", {}).get("test_days", 7)
    val_days = config.get("data", {}).get("val_days", 7)

    train_cutoff = max_time - test_days - val_days
    val_cutoff = max_time - test_days

    max_encoder_length = tft_cfg.get("max_encoder_length", 30)
    max_prediction_length = config.get("stage2_forecasting", {}).get(
        "forecast_horizon", 7
    )

    # Ensure string types for categoricals
    for col in ["city_id", "store_id", "first_category_id",
                 "second_category_id", "third_category_id", "series_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Fill NaN target with 0 (will be masked during training)
    df[target_col] = df[target_col].fillna(0)

    # Time-varying known reals (known in advance for future)
    time_varying_known_reals = ["discount"]
    known_fourier = [c for c in df.columns if c.startswith("sin_") or c.startswith("cos_")]
    time_varying_known_reals.extend(known_fourier)

    # Time-varying known categoricals
    time_varying_known_cats = ["day_of_week", "month", "holiday_flag"]
    for col in time_varying_known_cats:
        df[col] = df[col].astype(str)

    # Time-varying unknown reals (only known in the past)
    time_varying_unknown_reals = [target_col]
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    time_varying_unknown_reals.extend(rolling_cols)

    # Weather (assume unknown for future — conservative)
    weather_cols = ["precpt", "avg_temperature", "avg_humidity", "avg_wind_level"]
    for wc in weather_cols:
        if wc in df.columns:
            time_varying_unknown_reals.append(wc)

    # Filter to existing columns
    time_varying_known_reals = [c for c in time_varying_known_reals if c in df.columns]
    time_varying_unknown_reals = [c for c in time_varying_unknown_reals if c in df.columns]

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= train_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["city_id", "first_category_id"],
        time_varying_known_categoricals=time_varying_known_cats,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[(df["time_idx"] > train_cutoff) & (df["time_idx"] <= val_cutoff)],
        stop_randomization=True,
    )

    batch_size = tft_cfg.get("batch_size", 64)

    logger.info(
        f"TFT datasets: train={len(training):,} samples, "
        f"val={len(validation):,} samples, "
        f"encoder={max_encoder_length}, horizon={max_prediction_length}"
    )

    return training, validation, {"batch_size": batch_size}


def train_tft(
    training_dataset,
    validation_dataset,
    config: dict,
    checkpoint_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
):
    """Train Temporal Fusion Transformer.

    Features:
        - Variable selection networks (auto feature importance)
        - Static enrichment (city/category context)
        - Temporal self-attention
        - Quantile output (P10/P50/P90) via QuantileLoss

    Args:
        training_dataset: pytorch-forecasting TimeSeriesDataSet.
        validation_dataset: Validation TimeSeriesDataSet.
        config: TFT config from experiment.yaml.
        checkpoint_dir: Directory for Lightning checkpoints.
        resume_from: Path to checkpoint to resume from.

    Returns:
        Fitted TemporalFusionTransformer model.
    """
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    tft_cfg = config.get("stage2_forecasting", {}).get("tft", {})
    batch_size = tft_cfg.get("batch_size", 64)

    train_dl = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=2
    )
    val_dl = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=2
    )

    # Quantile loss for probabilistic output
    quantiles = tft_cfg.get("quantiles", [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=tft_cfg.get("hidden_size", 64),
        attention_head_size=tft_cfg.get("attention_head_size", 4),
        dropout=tft_cfg.get("dropout", 0.1),
        hidden_continuous_size=tft_cfg.get("hidden_continuous_size", 16),
        output_size=len(quantiles),
        loss=QuantileLoss(quantiles=quantiles),
        learning_rate=tft_cfg.get("learning_rate", 1e-3),
        reduce_on_plateau_patience=tft_cfg.get("reduce_on_plateau_patience", 4),
        log_interval=10,
    )

    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        pl.callbacks.LearningRateMonitor(),
    ]

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                save_top_k=2,
                monitor="val_loss",
                every_n_epochs=1,
                save_last=True,
                filename="tft-{epoch}-{val_loss:.4f}",
            )
        )

    trainer = pl.Trainer(
        max_epochs=tft_cfg.get("max_epochs", 50),
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    logger.info("Training TFT...")
    trainer.fit(
        tft,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=str(resume_from) if resume_from else None,
    )
    logger.info("TFT training complete")

    # Load best checkpoint
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        tft = TemporalFusionTransformer.load_from_checkpoint(best_path)
        logger.info(f"Loaded best TFT from {best_path}")

    return tft


# ============================================================
# N-HiTS [arXiv:2201.12886]
# ============================================================


def train_nhits(
    nf_df: pd.DataFrame,
    config: dict,
    checkpoint_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
):
    """Train N-HiTS for multi-periodicity demand forecasting.

    N-HiTS uses hierarchical interpolation + multi-rate data sampling.
    Pure MLP-based → 50x faster than Transformer architectures.

    Reference: arXiv:2201.12886 — 25% accuracy improvement over Transformers.

    Args:
        nf_df: NeuralForecast-format DataFrame (unique_id, ds, y).
        config: N-HiTS config.
        checkpoint_dir: Directory for checkpoints.
        resume_from: Path to resume checkpoint.

    Returns:
        Fitted NeuralForecast object with N-HiTS.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MAE
    import pytorch_lightning as pl

    nhits_cfg = config.get("stage2_forecasting", {}).get("nhits", {})
    horizon = config.get("stage2_forecasting", {}).get("forecast_horizon", 7)

    callbacks = []
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                save_last=True,
                every_n_epochs=1,
                filename="nhits-{epoch}-{valid_loss:.4f}",
            )
        )

    model = NHITS(
        h=horizon,
        input_size=nhits_cfg.get("input_size", 30),
        n_blocks=nhits_cfg.get("n_blocks", [1, 1, 1]),
        mlp_units=nhits_cfg.get("mlp_units", [[512, 512]] * 3),
        n_pool_kernel_size=nhits_cfg.get("n_pool_kernel_size", [4, 2, 1]),
        loss=MAE(),
        max_steps=nhits_cfg.get("max_steps", 500),
        batch_size=nhits_cfg.get("batch_size", 32),
        trainer_kwargs={"callbacks": callbacks} if callbacks else {},
    )

    nf = NeuralForecast(models=[model], freq="D")

    logger.info("Training N-HiTS...")
    nf.fit(df=nf_df)
    logger.info("N-HiTS training complete")

    return nf


def prepare_neuralforecast_daily(
    df: pd.DataFrame,
    target_col: str = "recovered_demand",
) -> pd.DataFrame:
    """Convert daily DataFrame to NeuralForecast format.

    Args:
        df: DataFrame with series_id, dt, and target column.
        target_col: Target column name.

    Returns:
        NeuralForecast-format DataFrame (unique_id, ds, y).
    """
    nf_df = df[["series_id", "dt", target_col]].copy()
    nf_df.columns = ["unique_id", "ds", "y"]
    nf_df["ds"] = pd.to_datetime(nf_df["ds"])
    nf_df = nf_df.dropna(subset=["y"])

    logger.info(
        f"NeuralForecast daily DataFrame: {len(nf_df):,} records, "
        f"{nf_df['unique_id'].nunique():,} series"
    )
    return nf_df
