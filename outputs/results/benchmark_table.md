# FreshRetailNet-50K Benchmark Results

**Pipeline completed in 4.48h**

## Task 2: Demand Forecasting

| Model | N | RMSE | MAE | R² | WAPE (%) | WPE (%) | ρ_DS |
|-------|---|------|-----|----|----------|---------|------|
| XGBoost (global) | 350,000 | 0.6650 | 0.3349 | 0.8706 | 28.00 | -3.87 | 0.121 |
| LGB-MSE (global) | 350,000 | 0.5853 | 0.3234 | 0.8997 | 27.03 | -4.91 | 0.151 |
| LGB-Recovered (global) | 350,000 | 0.6871 | 0.3537 | 0.8618 | 29.57 | -4.95 | 0.182 |
| LGB Blend 70/30 | 350,000 | 0.6057 | 0.3284 | 0.8926 | 27.45 | -4.92 | 0.160 |
| Random Forest (global) | 350,000 | 0.6263 | 0.3506 | 0.8852 | 29.30 | -6.73 | 0.145 |
| Ridge Regression | 350,000 | 0.7084 | 0.4045 | 0.8531 | 33.81 | -7.85 | 0.081 |
| kNN (k=15) | 350,000 | 0.7757 | 0.4472 | 0.8239 | 37.38 | -17.07 | 0.077 |
| LSTM (global) | 350,000 | 0.7386 | 0.4341 | 0.8403 | 36.29 | -12.51 | -0.058 |
| N-HiTS (global) | 350,000 | 0.7162 | 0.4237 | 0.8498 | 35.42 | -10.66 | -0.062 |
| Stacking Ensemble | 350,000 | 0.6048 | 0.3287 | 0.8929 | 27.48 | -6.58 | 0.154 |

**Paper baseline:** WAPE=27.62%, R²=0.816, ρ_DS=0.07
**Best model:** LGB-MSE (global) (WAPE=27.03%)

## Ablation Study

| Ablation | Condition | WAPE (%) | Δ WAPE |
|----------|-----------|----------|--------|
| Recovery impact | raw vs recovered | 28.00 vs 30.09 | +2.09 |
| Stacking value | single → ensemble | 28.00 → 27.03 | +0.96 |
| Global vs per-city | global vs per-city | 28.00 vs 29.30 | +1.30 |
| Fourier features | without → with | 27.66 → 28.00 | -0.34 |

## Task 1: Latent Demand Recovery

| Metric | Value |
|--------|-------|
| Ensemble WAPE | 38.47% |
| Ensemble RMSE | 0.7899 |
| TimesNet weight | 0.505 |
| PatchTST weight | 0.495 |
| Records recovered | 1,444,695 (32.1%) |