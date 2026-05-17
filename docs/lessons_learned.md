# Lessons Learned

> Key insights, challenges, and failed experiments during the FreshRetailNet-50K demand forecasting project.

## What Worked

### 1. Global Modeling > Per-City Modeling
Ablation study showed global XGBoost (all 18 cities in one model) achieved WAPE 28.00% vs per-city models at 29.30%. With 50,000 series across 18 cities, the global model benefits from shared patterns (seasonality, weather response) while `city_id` as a feature allows city-specific adjustments. Per-city models suffered from smaller training sets in low-volume cities (e.g., City 8 with only 455 test rows).

### 2. Stockout-Aware Feature Engineering
The dataset has 55.7% stockout rate — more than half of daily records show truncated sales. Key features that helped:
- `stockout_ratio`: fraction of operating hours out of stock (SHAP rank #2)
- `stockout_ratio_7d/14d/28d`: rolling stockout averages capture persistent supply issues
- `is_stockout`, `stockout_hours`: direct stockout context

These features let the model learn: "when stockout is high, observed sales understate true demand."

### 3. Ensemble Recovery (TimesNet + PatchTST)
Using two neural forecasting models (TimesNet and PatchTST) with cross-validation to recover latent demand on stockout days. The ensemble (WAPE 35.67%) outperformed individual models (35.38% and 36.79%) by combining their complementary strengths — TimesNet captures multi-period patterns while PatchTST excels at patch-level temporal dependencies.

### 4. LightGBM as Best Single Model
LightGBM consistently outperformed XGBoost (WAPE 27.03% vs 28.00%). The leaf-wise growth strategy of LightGBM proved more efficient for this high-dimensional tabular dataset. Tuning with lower learning rate (0.01 vs 0.02) and deeper trees (255 leaves vs 127) improved generalization further.

## What Didn't Work

### 1. Recovery as Training Target
Training LightGBM with `recovered_demand` (from Stage 1) as target instead of `sale_amount` increased WAPE from 28.00% to 30.09%. The recovery model's own error (WAPE ~35%) propagated as noise into the forecasting model. The recovered target helped WPE slightly but the WAPE cost was too high.

### 2. N-HiTS for This Dataset
N-HiTS (global neural model) achieved only 35.42% WAPE — significantly worse than tree-based models. With 50,000 short time series (90 days each), there isn't enough temporal depth for deep models to learn effectively. Tree models leverage cross-sectional features (weather, promotions, stockout context) much better than sequence models in this setting.

### 3. Stacking Ensemble
Ridge stacking of XGBoost + LightGBM + Random Forest (WAPE 27.48%) slightly underperformed the best single model (LightGBM at 27.03%). Meta-weights showed LightGBM dominates at 75%, suggesting the base models are too correlated for ensemble gains.

### 4. Fourier Features
Ablation showed Fourier harmonics (sin/cos for 7d and 30d periods) actually *harmed* performance: WAPE 27.66% without Fourier vs 28.00% with Fourier (+0.34%). The calendar features (day_of_week, month) already capture the same seasonal patterns more directly.

## Other Experiments

### Increasing Recovery Model Capacity
Upgrading TimesNet/PatchTST from `hidden_size=64` to `128` and `max_steps=500` to `2000` improved recovery WAPE from 38.36% to 35.67% — a meaningful but insufficient improvement. The 90-day series length fundamentally limits what neural sequence models can learn. Further capacity increases hit `input_size` constraints (series too short for training with large context windows — we encountered this error when setting `input_size=60`).

## Algorithm Families Implemented

The assignment listed 6 algorithm families. We implemented 5 of them:

| Algorithm Family | Implementation | Performance |
|-----------------|----------------|-------------|
| Gradient Boosting | XGBoost, LightGBM | Best performers (WAPE ~27%) |
| Random Forest | RandomForestRegressor (500 trees) | Competitive, used in stacking |
| Linear Regression / Ridge | Ridge Regression (standalone, α=10) | Baseline comparison |
| k-Nearest Neighbors | KNeighborsRegressor (k=15, distance-weighted) | Tabular data baseline |
| LSTM / GRU | NeuralForecast LSTM (2-layer, 128 hidden) | Short-series neural baseline |
| N-HiTS | Hierarchical interpolation (modern neural) | Neural architecture comparison |

### Algorithms Considered but Excluded

| Algorithm | Reason |
|-----------|--------|
| ARIMA/SARIMA | Single-series method; impractical for 50,000 series. Global approach is fundamentally different paradigm. |
| ETS/Holt-Winters | Same limitation as ARIMA — per-series only. Covered by the global neural models instead. |

## Key Numbers

| Metric | Value |
|--------|-------|
| Best Model | LightGBM (global, MSE objective) |
| WAPE | 27.03% (paper baseline: 27.62%) |
| R² | 0.8997 (paper baseline: 0.816) |
| Training time | ~4.5 hours (Lightning AI, T4 GPU) |
| Features used | 45 |
| Stockout rate | 55.7% |
| Recovery impact | 32.1% of rows modified |

## Features/Functionalities That Don't Work

| Feature | What Happened | Why |
|---------|---------------|-----|
| **Recovery as training target** | WAPE increased from 28.00% → 30.09% | Recovery model's own 35% error propagates as noise |
| **Fourier harmonics** | WAPE increased from 27.66% → 28.00% | Redundant with calendar features (day_of_week, month) |
| **Per-city models** | WAPE increased from 28.00% → 29.30% | Low-volume cities have too little data to train separate models |
| **N-HiTS / LSTM** | WAPE 35-37% (vs tree models at 27%) | 90-day series too short for deep sequence models |
| **Stacking ensemble** | WAPE 27.48% (vs single LGB at 27.03%) | Base models too correlated for ensemble benefit |
| **ARIMA/SARIMA** | Not implemented | Impractical: per-series method for 50,000 series |
| **ETS/Holt-Winters** | Not implemented | Same limitation as ARIMA |
