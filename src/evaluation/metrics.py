"""
Benchmark evaluation metrics aligned with FreshRetailNet-50K paper.

Implements all metrics from the official paper + custom business metrics.
Generates comparison tables and ablation reports.

References:
    - FreshRetailNet-50K (arXiv:2505.16319): WAPE, WPE, ρ_DS definitions
    - Wu 2026: RMSE, R² benchmarks
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Core Metrics [FreshRetailNet-50K Paper]
# ============================================================


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error.

    WAPE = Σ|y_true - y_pred| / Σy_true × 100

    Lower is better. Primary metric in FRN-50K paper.
    """
    total = np.sum(np.abs(y_true))
    if total == 0:
        return float("inf")
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100)


def wpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Percentage Error (bias indicator).

    WPE = (Σy_pred - Σy_true) / Σy_true × 100

    Positive = over-prediction, Negative = under-prediction.
    Near zero is ideal. Raw sales typically show WPE ≈ -7% (systematic underestimate).
    """
    total = np.sum(np.abs(y_true))
    if total == 0:
        return float("inf")
    return float((np.sum(y_pred) - np.sum(y_true)) / total * 100)


def decoupling_score(
    forecast_errors: np.ndarray, stockout_ratios: np.ndarray
) -> float:
    """Decoupling Score ρ_DS — correlation between forecast error and stockout.

    ρ_DS = corr(|y_true - y_pred|, stockout_ratio)

    Target: ρ_DS ≈ 0 (forecast decorrelated from stockout).
    Raw sales: ρ_DS ≈ -0.57 (heavily correlated).
    After recovery: ρ_DS ≈ 0.07 (decorrelated).
    """
    if len(forecast_errors) < 3:
        return float("nan")
    corr = np.corrcoef(forecast_errors, stockout_ratios)
    return float(corr[0, 1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² Score (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Mean Absolute Percentage Error (with epsilon to avoid division by zero)."""
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, epsilon))) * 100)


# ============================================================
# Quantile / Probabilistic Metrics
# ============================================================


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball (quantile) loss.

    Asymmetric loss for evaluating quantile predictions.
    """
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return float(np.mean(loss))


def coverage_rate(
    y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray
) -> float:
    """Coverage rate: fraction of actuals within prediction interval.

    Target: should match the intended coverage (e.g., 80% for q10-q90).
    """
    covered = (y_true >= q_low) & (y_true <= q_high)
    return float(np.mean(covered) * 100)


# ============================================================
# Composite Evaluation
# ============================================================


class BenchmarkEvaluator:
    """Full evaluation framework aligned with FRN-50K paper metrics."""

    def task1_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        stockout_mask: np.ndarray,
    ) -> dict:
        """Evaluate Stage 1 demand recovery quality.

        Args:
            y_true: True demand (from eval set ground truth).
            y_pred: Recovered demand predictions.
            stockout_mask: Binary mask (1 = stockout occurred).

        Returns:
            Dict of Task 1 metrics.
        """
        abs_errors = np.abs(y_true - y_pred)

        return {
            "WAPE": wape(y_true, y_pred),
            "WPE": wpe(y_true, y_pred),
            "rho_DS": decoupling_score(abs_errors, stockout_mask),
            "RMSE": rmse(y_true, y_pred),
            "MAE": mae(y_true, y_pred),
        }

    def task2_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """Evaluate Stage 2 forecasting quality.

        Args:
            y_true: True daily demand.
            y_pred: Forecasted demand.

        Returns:
            Dict of Task 2 metrics.
        """
        return {
            "RMSE": rmse(y_true, y_pred),
            "MAE": mae(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
            "WAPE": wape(y_true, y_pred),
            "WPE": wpe(y_true, y_pred),
            "MAPE": mape(y_true, y_pred),
        }

    def quantile_metrics(
        self,
        y_true: np.ndarray,
        quantile_preds: dict[str, np.ndarray],
    ) -> dict:
        """Evaluate probabilistic forecast quality.

        Args:
            y_true: True values.
            quantile_preds: Dict with keys like 'q10', 'q50', 'q90'.

        Returns:
            Dict of quantile metrics.
        """
        result = {}

        for q_name, q_pred in quantile_preds.items():
            q_value = float(q_name.replace("q", "")) / 100
            result[f"pinball_{q_name}"] = pinball_loss(y_true, q_pred, q_value)

        if "q10" in quantile_preds and "q90" in quantile_preds:
            result["coverage_80"] = coverage_rate(
                y_true, quantile_preds["q10"], quantile_preds["q90"]
            )

        if "q02" in quantile_preds and "q98" in quantile_preds:
            result["coverage_96"] = coverage_rate(
                y_true, quantile_preds["q02"], quantile_preds["q98"]
            )

        return result

    def per_city_breakdown(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        city_ids: np.ndarray,
    ) -> dict:
        """Compute metrics per city for granular analysis.

        Args:
            y_true: True values.
            y_pred: Predictions.
            city_ids: City ID for each sample.

        Returns:
            Dict mapping city_id → metrics dict.
        """
        result = {}
        for city in np.unique(city_ids):
            mask = city_ids == city
            if mask.sum() < 10:
                continue
            result[str(city)] = {
                "n_samples": int(mask.sum()),
                "RMSE": rmse(y_true[mask], y_pred[mask]),
                "R2": r2_score(y_true[mask], y_pred[mask]),
                "WAPE": wape(y_true[mask], y_pred[mask]),
            }
        return result

    def full_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        stockout_mask: np.ndarray | None = None,
        city_ids: np.ndarray | None = None,
        quantile_preds: dict | None = None,
        model_name: str = "model",
    ) -> dict:
        """Generate complete evaluation report.

        Args:
            y_true: True values.
            y_pred: Point predictions.
            stockout_mask: Stockout indicators (for ρ_DS).
            city_ids: City IDs (for per-city breakdown).
            quantile_preds: Quantile predictions.
            model_name: Name for logging.

        Returns:
            Complete metrics dictionary.
        """
        report = {
            "model": model_name,
            "n_samples": len(y_true),
            "task2": self.task2_metrics(y_true, y_pred),
        }

        if stockout_mask is not None:
            abs_errors = np.abs(y_true - y_pred)
            report["rho_DS"] = decoupling_score(abs_errors, stockout_mask)

        if city_ids is not None:
            report["per_city"] = self.per_city_breakdown(y_true, y_pred, city_ids)

        if quantile_preds is not None:
            report["quantile"] = self.quantile_metrics(y_true, quantile_preds)

        logger.info(f"[{model_name}] {report['task2']}")
        return report


def generate_comparison_table(results: list[dict]) -> str:
    """Generate markdown comparison table from multiple model results.

    Args:
        results: List of report dicts from BenchmarkEvaluator.full_report().

    Returns:
        Markdown-formatted comparison table.
    """
    if not results:
        return "No results to compare."

    header = "| Model | RMSE | MAE | R² | WAPE (%) | WPE (%) |"
    separator = "|-------|------|-----|----|---------|---------| "
    rows = [header, separator]

    for r in results:
        t2 = r.get("task2", {})
        row = (
            f"| {r.get('model', '?')} "
            f"| {t2.get('RMSE', 0):.4f} "
            f"| {t2.get('MAE', 0):.4f} "
            f"| {t2.get('R2', 0):.4f} "
            f"| {t2.get('WAPE', 0):.2f} "
            f"| {t2.get('WPE', 0):.2f} |"
        )
        rows.append(row)

    return "\n".join(rows)


def save_results(
    results: list[dict],
    output_dir: str | Path,
    filename: str = "benchmark_results.json",
):
    """Save evaluation results to JSON.

    Args:
        results: List of report dicts.
        output_dir: Output directory.
        filename: Output filename.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also save markdown table
    table_path = output_dir / "benchmark_table.md"
    with open(table_path, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(generate_comparison_table(results))
        f.write("\n")

    logger.info(f"Results saved to {path} and {table_path}")
