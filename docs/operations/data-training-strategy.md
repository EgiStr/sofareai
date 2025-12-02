# Data Training Strategy - Best Practices

## Overview

SOFARE-AI menggunakan strategi training yang optimal untuk time-series ML, menggantikan rolling window dengan pendekatan full dataset yang lebih robust.

## Perubahan dari Rolling Window ke Full Dataset

### Sebelum (Rolling Window)
```python
ROLLING_WINDOW_SIZE = 2000  # Train on last 2000 candles
```

**Masalah:**
- ❌ Kehilangan pola historis jangka panjang
- ❌ Model tidak belajar dari event langka (black swan)
- ❌ Bias towards recent patterns
- ❌ Kurang generalisasi

### Sesudah (Full Dataset)
```python
USE_FULL_DATASET = True   # Use all available data
MIN_DATA_POINTS = 500     # Minimum data required
MAX_DATA_POINTS = None    # No limit (or set a max)

TRAIN_RATIO = 0.7   # 70% for training
VAL_RATIO = 0.15    # 15% for validation  
TEST_RATIO = 0.15   # 15% for final testing
```

**Keuntungan:**
- ✅ Model belajar dari semua pola historis
- ✅ Lebih robust terhadap berbagai kondisi market
- ✅ Proper temporal split (no data leakage)
- ✅ Held-out test set untuk evaluasi final yang objektif

## Best Practices Time-Series ML

### 1. Temporal Split (WAJIB!)

```
|-------- TRAIN (70%) --------|-- VAL (15%) --|-- TEST (15%) --|
    Jan 2023 - Sep 2024           Oct 2024       Nov-Dec 2024
         ↓                           ↓               ↓
    Model learns              Tune hyperparams   Final evaluation
```

**JANGAN PERNAH:**
- ❌ Random shuffle data time-series
- ❌ Gunakan future data untuk prediksi past
- ❌ Campurkan train/val/test secara random

### 2. Data Leakage Prevention

```python
# SALAH - Data leakage!
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# BENAR - Temporal split
train_indices = list(range(0, train_end))
val_indices = list(range(train_end, val_end))
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
```

### 3. Shuffle Strategy

```python
# Training: Shuffle OK (helps generalization within train period)
train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True)

# Validation/Test: NEVER shuffle (maintain temporal order)
val_loader = DataLoader(val_dataset, shuffle=False)
test_loader = DataLoader(test_dataset, shuffle=False)
```

### 4. Held-out Test Set

Test set harus:
- Tidak pernah digunakan selama training/tuning
- Hanya untuk evaluasi final
- Mencerminkan data "masa depan" yang belum pernah dilihat model

```python
# Final evaluation on test set
model.eval()
with torch.no_grad():
    for batch in test_loader:
        # Evaluate...
```

## Konfigurasi

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `USE_FULL_DATASET` | `True` | Gunakan seluruh data (recommended) |
| `MIN_DATA_POINTS` | `500` | Minimum data untuk training |
| `MAX_DATA_POINTS` | `None` | Limit maksimum (None = no limit) |
| `TRAIN_RATIO` | `0.7` | Proporsi data training |
| `VAL_RATIO` | `0.15` | Proporsi data validasi |
| `TEST_RATIO` | `0.15` | Proporsi data testing |

## Data Quality Checks

Training pipeline otomatis melakukan:

1. **Null Value Detection**
   ```python
   null_counts = df.isnull().sum()
   if null_counts.any():
       logger.warning(f"Found null values: {null_counts}")
   ```

2. **Duplicate Removal**
   ```python
   df = df.drop_duplicates(subset=['timestamp'], keep='last')
   ```

3. **Temporal Ordering**
   ```python
   df = df.sort_values('timestamp').reset_index(drop=True)
   ```

4. **Data Statistics Logging**
   ```python
   stats = get_data_statistics(df, feature_cols)
   mlflow.log_param("total_data_points", len(df))
   ```

## MLflow Tracking

Semua informasi data strategy di-log ke MLflow:

```python
mlflow.set_tag("data_strategy", "full_dataset")
mlflow.log_param("total_data_points", len(df))
mlflow.log_param("train_samples", len(train_dataset))
mlflow.log_param("val_samples", len(val_dataset))
mlflow.log_param("test_samples", len(test_dataset))

# Test metrics (final evaluation)
mlflow.log_metric("test_mae", test_mae)
mlflow.log_metric("test_rmse", test_rmse)
mlflow.log_metric("test_r2_score", test_r2)
mlflow.log_metric("test_cls_accuracy", test_accuracy)
```

## Fallback ke Rolling Window

Jika diperlukan (e.g., memory constraint), set:

```python
USE_FULL_DATASET = False
ROLLING_WINDOW_SIZE = 10000  # Atau sesuai kebutuhan
```

## Kapan Gunakan Masing-masing Strategy?

### Full Dataset (Recommended)
- ✅ Production training
- ✅ Research/experimentation
- ✅ Model evaluation
- ✅ Hyperparameter tuning

### Rolling Window
- ⚠️ Memory constraint
- ⚠️ Real-time adaptation (online learning)
- ⚠️ Testing/debugging cepat

## Auto-Promotion Based on Test Metrics

Model otomatis di-promote ke production jika:

```python
if test_accuracy > 0.55:
    version_manager.deploy_to_production(version)
    logger.info(f"Model deployed (test_accuracy: {test_accuracy:.4f})")
else:
    logger.info(f"Model not promoted (test_accuracy: {test_accuracy:.4f} < 0.55)")
```

## Contoh Output

```
INFO - Loaded OHLCV data: 50000 records, date range: 2023-01-01 to 2024-12-01
INFO - Using FULL DATASET: 50000 data points for training
INFO - Temporal Split - Train: 0:35000 (35000 samples)
INFO - Temporal Split - Val: 35000:42500 (7500 samples)
INFO - Temporal Split - Test: 42500:50000 (7500 samples)
INFO - TEST SET RESULTS - MAE: 0.0234, RMSE: 0.0456, Accuracy: 0.5823
INFO - Model v1.2.3 deployed to production (test_accuracy: 0.5823)
```
