"""
Kaggle Session Guard — auto-save before session timeout.
Handles 12-hour Kaggle GPU session limits with graceful checkpoint saving.
"""

import time
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class KaggleSessionGuard:
    """Monitor Kaggle session time and trigger saves before timeout.

    Usage:
        guard = KaggleSessionGuard(session_hours=12, save_buffer_minutes=15)

        for epoch in range(max_epochs):
            train_one_epoch(model)

            if guard.should_save_and_stop():
                guard.save_checkpoint(model, metrics, epoch)
                break
    """

    def __init__(self, session_hours: float = 12, save_buffer_minutes: float = 15):
        self.start_time = time.time()
        self.limit_seconds = (session_hours * 60 - save_buffer_minutes) * 60
        self.session_hours = session_hours
        self.save_buffer_minutes = save_buffer_minutes

        logger.info(
            f"Session guard initialized: {session_hours}h session, "
            f"{save_buffer_minutes}min buffer → hard stop at {self.limit_seconds/3600:.1f}h"
        )

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def elapsed_hours(self) -> float:
        return self.elapsed_seconds / 3600

    @property
    def remaining_seconds(self) -> float:
        return max(0, self.limit_seconds - self.elapsed_seconds)

    @property
    def remaining_hours(self) -> float:
        return self.remaining_seconds / 3600

    def should_save_and_stop(self) -> bool:
        """Check if we should save and stop training."""
        should_stop = self.elapsed_seconds >= self.limit_seconds
        if should_stop:
            logger.warning(
                f"⚠️ Session guard triggered at {self.elapsed_hours:.2f}h. "
                f"Saving checkpoint and stopping."
            )
        return should_stop

    def log_status(self, extra_info: str = ""):
        """Log current session status."""
        msg = (
            f"⏱ Session: {self.elapsed_hours:.2f}h elapsed, "
            f"{self.remaining_hours:.2f}h remaining"
        )
        if extra_info:
            msg += f" | {extra_info}"
        logger.info(msg)

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        models: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Save all artifacts to checkpoint directory.

        Args:
            checkpoint_dir: Directory to save checkpoints.
            models: Dict of {name: model_object}. Each must have a save method
                    or be serializable.
            metrics: Dict of evaluation metrics.
            metadata: Extra metadata (epoch, step, etc.).
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        if metrics:
            metrics_path = checkpoint_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Saved metrics to {metrics_path}")

        # Save metadata
        meta = {
            "elapsed_hours": self.elapsed_hours,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(metadata or {}),
        }
        meta_path = checkpoint_dir / "session_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Save models (framework-specific saving handled by caller)
        if models:
            for name, model in models.items():
                model_dir = checkpoint_dir / name
                model_dir.mkdir(exist_ok=True)
                # Try common save patterns
                if hasattr(model, "save_model"):
                    model.save_model(str(model_dir / "model.json"))
                elif hasattr(model, "save"):
                    model.save(str(model_dir / "model.pt"))
                else:
                    import joblib
                    joblib.dump(model, model_dir / "model.joblib")
                logger.info(f"Saved model '{name}' to {model_dir}")

        logger.info(
            f"✅ Checkpoint saved to {checkpoint_dir} at {self.elapsed_hours:.2f}h"
        )
