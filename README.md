# Demand Forecasting: Censored Demand Recovery & Multi-Model Ensemble

> kSynerX Internship Assessment — Option 2: Demand Forecasting  
> A benchmark-focused pipeline that aims to **beat published results** on FreshRetailNet-50K.

## Architecture

```
Stage 0: Stockout-Aware Feature Engineering     [VN2 Winner, arXiv:2601.18919]
    ↓
Stage 1: Ensemble Demand Recovery               [FRN-50K paper, arXiv:2505.16319]
    TimesNet (#1) + iTransformer (#2) → inverse-error weighted average
    ↓
Stage 2: Stacking Forecast                      [arXiv:2010.08158]
    XGBoost + LightGBM + TFT + N-HiTS → Ridge meta-learner
    ↓
Serving: Ray Serve REST API
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run data preparation (Phase 1)
python -m src.pipeline.data_preparation

# Run training (Phase 2) — designed for Kaggle GPU
# See notebooks/ for Kaggle-ready training scripts

# Start serving
python -m src.serving.ray_app
```

## Benchmark Results

> Results will be populated after training.

| Model | RMSE | MAE | R² | WAPE (%) | WPE (%) |
|-------|------|-----|----|---------|---------| 
| XGBoost (per-city) | — | — | — | — | — |
| LightGBM (per-city) | — | — | — | — | — |
| TFT (global) | — | — | — | — | — |
| N-HiTS (global) | — | — | — | — | — |
| **Stacking Ensemble** | — | — | — | — | — |

### Benchmark Targets

| Metric | Paper Ceiling (single model) | Our Target (ensemble) |
|--------|-----------------------------|-----------------------|
| Task 1 WAPE | 27.62% (TimesNet) | < 25% |
| Task 2 R² | 0.816 (XGBoost, Wu 2026) | > 0.95 |
| Task 2 WPE | 2.58% | < 1.0% |

## Key References

1. **FreshRetailNet-50K** (arXiv:2505.16319) — Dataset & two-stage framework
2. **VN2 Winner** (arXiv:2601.18919) — Stockout-aware feature engineering
3. **Wu 2026** (JID 15) — XGBoost baseline on FRN-50K
4. **TFT for Retail** (arXiv:2511.00552) — Temporal Fusion Transformer
5. **N-HiTS** (AAAI 2023) — Hierarchical interpolation forecasting

## Project Structure

```
├── configs/experiment.yaml          # All hyperparameters
├── src/
│   ├── data/                        # Loading, preprocessing, feature engineering
│   ├── stage1_recovery/             # TimesNet + iTransformer ensemble
│   ├── stage2_forecasting/          # XGBoost, LightGBM, TFT, N-HiTS, stacking
│   ├── evaluation/                  # Metrics framework (WAPE, WPE, ρ_DS)
│   └── serving/                     # Ray Serve REST API
├── notebooks/                       # Kaggle training notebooks
├── models/                          # Saved checkpoints
├── results/                         # Benchmark tables, figures
└── docs/
    ├── lessons_learned.md
    └── ai_disclosure.md
```

## AI Disclosure

See [docs/ai_disclosure.md](docs/ai_disclosure.md) for details on AI-assisted vs self-written code.
