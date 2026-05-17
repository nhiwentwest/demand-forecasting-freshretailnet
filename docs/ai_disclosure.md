# AI Tools Disclosure

> Per assignment requirement: "Clearly explain which code is written by you and by using AI tools."

## AI Tools Used

| Tool | Purpose |
|------|---------|
| Claude (Anthropic) | Code generation, debugging, documentation drafting |

## What I (Human) Did

- **Research & problem definition**: Read the FreshRetailNet-50K paper (Wu et al. 2026), understood the two-stage recovery → forecasting framework, selected which algorithm families to implement
- **Experimental design**: Designed ablation studies (global vs per-city, Fourier features, recovered vs raw target), decided evaluation metrics (WAPE, WPE, R²)
- **Data analysis**: Identified the 55.7% stockout rate in the dataset, interpreted SHAP feature importance results, discovered the data leakage bug (recovery features accidentally in FEATURE_COLS)
- **Hyperparameter tuning decisions**: Chose LightGBM learning rate (0.01), num_leaves (255), n_estimators (8000) based on validation results
- **Infrastructure**: Set up training on Lightning AI (GPU), managed Kaggle/Lightning environments, handled dependency conflicts (NumPy 2.x, scikit-learn version mismatches)
- **All training runs**: Executed 3 full pipeline runs, monitored training, analyzed outputs
- **Final reporting**: Selected which results to include, verified numbers are accurate

## What AI (Claude) Did

- **Code generation**: Wrote the bulk of the pipeline code (`full_pipeline.py` ~1100 lines), the Ray Serve API (`ray_app.py` ~470 lines), and module scaffolding (`src/`)
- **Feature engineering code**: Implemented the 45-feature computation (lags, rolling stats, Fourier harmonics, stockout ratios)
- **Model training code**: Implemented XGBoost, LightGBM, Random Forest, Ridge, kNN, LSTM, N-HiTS training loops using scikit-learn and NeuralForecast APIs
- **Recovery pipeline**: Coded the TimesNet + PatchTST ensemble recovery stage using NeuralForecast
- **Stacking ensemble**: Implemented Ridge meta-learner stacking of XGBoost + LightGBM + Random Forest
- **Documentation**: Drafted README, lessons_learned.md, this disclosure file
- **Debugging**: Fixed import scoping issues, recovery cache logic, notebook conversion

## Key Code Sections (from paper methodology)

These are the core algorithmic pieces that implement the paper's framework:

### 1. Stockout-aware demand recovery (Stage 1)
The paper's key insight: use neural models to recover latent demand on stockout days.
```python
# Recovery: replace stockout-day sales with model predictions
mask = df["is_stockout"] == 1
df.loc[mask, "recovered_demand"] = ensemble_predictions[mask]
# Non-stockout days keep original sales
df.loc[~mask, "recovered_demand"] = df.loc[~mask, "sale_amount"]
```

### 2. Global model with per-series normalization
Instead of fitting 50K separate models, one global model with series_scale feature:
```python
series_scale = df.groupby("series_id")["sale_amount"].transform("mean")
df["series_scale"] = series_scale
# All series share one model, scale feature lets it adapt
```

### 3. Stockout context features (paper contribution)
```python
df["stockout_ratio_7d"] = df.groupby("series_id")["is_stockout"] \
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
# Tells the model: "this product was out of stock 40% of the last week"
```

### 4. LightGBM with tuned hyperparameters (best model)
```python
lgb.LGBMRegressor(
    n_estimators=8000, learning_rate=0.01, num_leaves=255,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    objective="mse", early_stopping_rounds=100
)
```

### 5. Stacking meta-learner
```python
# Ridge regression combines 3 base model predictions
meta_X = np.column_stack([pred_lgb, pred_rf, pred_xgb])
stacking_model = Ridge(alpha=1.0).fit(meta_X_train, y_train)
```
