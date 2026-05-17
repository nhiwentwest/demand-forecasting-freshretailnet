# Demand Forecasting: FreshRetailNet-50K

> kSynerX Internship Assessment — Demand Forecasting

## Benchmark Results

> Trained on FreshRetailNet-50K (4.5M rows, 50K series, 18 cities, 90 days).
> Evaluated on held-out test set (350,000 rows, last 7 days).

| Model | Algorithm Family | WAPE (%) | WPE (%) | R² | RMSE |
|-------|-----------------|----------|---------|-----|------|
| **LGB-MSE (global)** | Gradient Boosting | **27.03** | -4.91 | **0.8997** | 0.5853 |
| Stacking Ensemble | Ensemble (Ridge meta) | 27.48 | -6.58 | 0.8929 | 0.6048 |
| XGBoost (global) | Gradient Boosting | 28.00 | -3.87 | 0.8706 | 0.6650 |
| LGB Blend 70/30 | Gradient Boosting | 27.45 | -4.92 | 0.8926 | 0.6057 |
| Random Forest (global) | Random Forest | 29.30 | -6.73 | 0.8852 | 0.6263 |
| Ridge Regression | Linear Regression | 33.81 | -7.85 | 0.8531 | 0.7084 |
| kNN (k=15) | k-Nearest Neighbors | 37.38 | -17.07 | 0.8239 | 0.7757 |
| LSTM (global) | LSTM / GRU | 36.29 | -12.51 | 0.8403 | 0.7386 |
| N-HiTS (global) | Neural (hierarchical) | 35.42 | -10.66 | 0.8498 | 0.7162 |

## Prototype: Ray Serve Inference API

The trained models are deployed as a REST API using **Ray Serve**, enabling real-time demand prediction.

![Ray Serve Dashboard — ForecastService running with prediction logs](docs/screenshots/ray_serve_predictions.png)

The screenshot shows:
- **Ray Serve Controller**: HEALTHY, Proxy: HEALTHY x1, Application: RUNNING
- **ForecastService** deployment: HEALTHY, 1 replica, deployed 2026/05/17
- **Prediction logs**: Multiple `POST /forecast` requests returning `200 OK` (response times 34–207ms)
- 6 models loaded: XGBoost, LightGBM, Random Forest, Ridge, kNN, Stacking Ensemble

```bash
# Start the API (auto-downloads checkpoints from Google Drive if missing)
python -m src.serving.ray_app --models-dir ./outputs/checkpoints

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"history":[...], "forecast_days":[...]}'
```

## Architecture

```
Stage 1: Ensemble Demand Recovery
    TimesNet + PatchTST → weighted ensemble (CV-validated)
    Recovers latent demand on stockout days (32.1% of records)
    ↓
Stage 2: Forecasting Models
    XGBoost (global) + LightGBM (global) → Ridge stacking
    45 features: lags, rolling stats, stockout context, calendar, weather, Fourier
    ↓
Serving: Ray Serve REST API
    Loads trained checkpoints, real-time inference via /forecast endpoint
```

### Ablation Study

| Experiment | WAPE | Δ vs baseline |
|-----------|------|--------------| 
| Global XGBoost (baseline) | 28.00% | — |
| Global vs per-city | 28.00% vs 29.30% | Global wins by 1.30% |
| With vs without Fourier | 27.66% vs 28.00% | Fourier adds +0.34% (slightly harmful) |
| Sale_amount vs recovered target | 28.00% vs 30.09% | Raw target is better |

## Features Considered

### Implemented (45 features)

| Category | Features | Count |
|----------|----------|-------|
| Lag features | lag_1, lag_7, lag_14, lag_21, lag_28 | 5 |
| Rolling statistics | mean/std for windows 7, 14, 28 | 6 |
| Stockout context | stockout_ratio, rolling 7d/14d/28d, is_stockout, stockout_hours | 6 |
| Scale normalization | series_scale (per-series mean) | 1 |
| Calendar | day_of_week, week_of_year, month, is_weekend, day_of_month | 5 |
| Fourier harmonics | sin/cos for periods 7d and 30d, harmonics 1-3 | 12 |
| External factors | discount, activity_flag, holiday_flag, weather (4 vars) | 7 |
| Target encoding | first_category_id_te, city_id_te | 2 |
| Identifier | city_id (for global model) | 1 |

### Not Implemented (with justification)

| Feature | Reason |
|---------|--------|
| `product_price` / `selling_price` | Not available in FreshRetailNet-50K dataset |
| `promotion_type` | Dataset provides binary `activity_flag` only |
| `is_bulk_order` | Not available in dataset (optional per assignment) |
| `year` | Dataset spans only 90 days within 2024; no inter-year signal |

## Algorithms

### Implemented (5/6 algorithm families from assignment)

| Algorithm Family | Implementation | Role |
|-----------------|----------------|------|
| **Gradient Boosting** | XGBoost, LightGBM | Best forecasters (WAPE ~27%) |
| **Random Forest** | RandomForestRegressor (500 trees) | Base forecaster + stacking component |
| **Linear Regression / Ridge** | Ridge Regression (standalone α=10) | Linear baseline + stacking meta-learner |
| **k-Nearest Neighbors** | KNeighborsRegressor (k=15, distance-weighted) | Instance-based baseline |
| **LSTM / GRU** | NeuralForecast LSTM (2-layer, 128 hidden) | Recurrent neural baseline |

Additionally implemented (not in assignment list):
| Model | Role |
|-------|------|
| **N-HiTS** | Hierarchical interpolation neural model |
| **TimesNet, PatchTST** | Used in Stage 1 demand recovery |

### Not Implemented (with justification)

| Algorithm | Reason |
|-----------|--------|
| ARIMA/SARIMA | Single-series method; impractical for 50,000 series. Would require fitting 50K individual models. |
| ETS/Holt-Winters | Same limitation as ARIMA — per-series only, does not scale to this dataset. |

## Project Structure

```
├── configs/experiment.yaml          # Hyperparameters
├── notebooks/
│   └── full_pipeline.ipynb          # Complete training pipeline (with outputs)
├── src/
│   ├── data/                        # Data loading, preprocessing, feature engineering
│   ├── stage1_recovery/             # TimesNet + PatchTST ensemble recovery
│   ├── stage2_forecasting/          # XGBoost, LightGBM, N-HiTS, stacking
│   ├── evaluation/                  # Metrics (WAPE, WPE, R², ρ_DS)
│   └── serving/                     # Ray Serve REST API with real inference
├── docs/
│   ├── lessons_learned.md           # Experiments, failures, insights
│   ├── ai_disclosure.md             # AI-assisted vs self-written code
│   └── screenshots/                 # Ray Serve demo screenshots
├── outputs/                         # Generated after training
│   ├── checkpoints/                 # Saved model files (auto-downloaded from Google Drive)
│   └── results/                     # Benchmark tables, SHAP plots
└── requirements.txt
```

## Key References

1. **FreshRetailNet-50K** — [Dingdong-Inc/FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (dataset)
2. **Wu et al. 2026** — [arXiv:2505.16319](https://arxiv.org/abs/2505.16319) (two-stage framework, paper baseline)
3. **NeuralForecast** — [Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast) (TimesNet, PatchTST, N-HiTS)

## AI Disclosure

See [docs/ai_disclosure.md](docs/ai_disclosure.md) for details.

## Lessons Learned

See [docs/lessons_learned.md](docs/lessons_learned.md) for full write-up.
