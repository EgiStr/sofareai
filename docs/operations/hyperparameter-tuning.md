# Hyperparameter Tuning

Panduan lengkap untuk hyperparameter tuning di SOFARE-AI menggunakan Optuna dengan integrasi MLflow.

## Overview

SOFARE-AI menggunakan **Optuna** untuk automated hyperparameter optimization dengan fitur:

- **TPE Sampler**: Tree-structured Parzen Estimator untuk efficient search
- **Hyperband Pruner**: Early stopping untuk trial yang tidak promising
- **MLflow Integration**: Tracking semua experiments
- **Multi-objective Support**: Optimize accuracy + inference speed

## Quick Start

### 1. Run Hyperparameter Tuning

```bash
# Di dalam container
docker exec -it sofare_training python src/run_tuning.py tune --n-trials 50 --timeout 3600

# Atau dengan docker-compose
docker-compose exec training python src/run_tuning.py tune --n-trials 50
```

### 2. View Best Parameters

```bash
docker exec -it sofare_training python src/run_tuning.py show
```

### 3. Training Menggunakan Best Params

Training otomatis menggunakan best parameters jika file `best_params.json` tersedia:

```bash
docker-compose up training
```

## Search Space

### Model Architecture

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `hidden_size` | 64-192 | 64 | Hidden layer dimension |
| `embed_dim` | [64, 128] | 128 | Embedding dimension |
| `num_heads` | [4, 8] | 4 | Attention heads |
| `num_encoder_layers` | 1-3 | 2 | Transformer layers |
| `dropout` | 0.1-0.4 | 0.1 | Dropout rate |

### Training Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `learning_rate` | 1e-4 - 1e-2 (log) | 0.001 | Learning rate |
| `batch_size` | [32, 64] | 32 | Batch size |
| `epochs` | 5-10 | 5 | Training epochs |
| `weight_decay` | 1e-5 - 1e-3 (log) | 1e-5 | L2 regularization |
| `optimizer` | [adam, adamw] | adam | Optimizer type |
| `cls_weight` | 0.3-0.7 | 0.5 | Classification loss weight |

### Data Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `sequence_length` | 30-90 | 60 | Input sequence length |

## Configuration

### TuningConfig

```python
from hyperparameter_tuning import TuningConfig

config = TuningConfig(
    # Study settings
    study_name="sofare-tuning",
    n_trials=50,
    timeout=3600,  # seconds
    n_jobs=1,  # parallel trials
    
    # Optimization
    direction="maximize",  # maximize accuracy
    
    # Search space
    sequence_length_range=(30, 90),
    batch_size_options=[32, 64],
    hidden_size_range=(64, 192),
    embed_dim_options=[64, 128],
    learning_rate_range=(1e-4, 1e-2),
    
    # Pruning
    use_pruning=True,
    pruner_type="hyperband",
    
    # MLflow
    mlflow_tracking_uri="http://mlflow:5000",
    experiment_name="SOFARE-AI-Tuning"
)
```

## Best Practices

### 1. Start with Coarse Search

```python
# Initial broad search
config = TuningConfig(
    n_trials=20,
    hidden_size_range=(32, 256),
    learning_rate_range=(1e-5, 1e-1),
)
```

### 2. Refine with Focused Search

```python
# Based on initial results, narrow the search
config = TuningConfig(
    n_trials=50,
    hidden_size_range=(96, 160),  # Narrowed
    learning_rate_range=(5e-4, 5e-3),  # Narrowed
)
```

### 3. Use Pruning Effectively

Hyperband pruner saves compute by stopping unpromising trials early:

```python
# Hyperband configuration
config = TuningConfig(
    use_pruning=True,
    pruner_type="hyperband",
    epochs_range=(5, 15),  # More epochs for pruning to work
)
```

### 4. Log Scale for Learning Rate

Learning rate search should always use log scale:

```python
# In hyperparameter_tuning.py
"learning_rate": trial.suggest_float(
    "learning_rate",
    1e-4, 1e-2,
    log=True  # Important!
)
```

### 5. Conditional Hyperparameters

Ensure `embed_dim` is divisible by `num_heads`:

```python
# embed_dim must be divisible by num_heads
if params["embed_dim"] % params["num_heads"] != 0:
    params["embed_dim"] = (params["embed_dim"] // params["num_heads"]) * params["num_heads"]
```

## Integration dengan Pipeline

### Automatic Parameter Loading

`train.py` secara otomatis load best parameters:

```python
# train.py
BEST_PARAMS_PATH = "/app/shared_model/best_params.json"

def load_tuned_params() -> dict:
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, "r") as f:
            config = json.load(f)
            return config.get("best_params", {})
    return DEFAULT_PARAMS
```

### CI/CD Integration

```yaml
# .github/workflows/ci-cd.yml
tuning:
  runs-on: ubuntu-latest
  steps:
    - name: Run Hyperparameter Tuning
      run: |
        docker-compose exec training python src/run_tuning.py tune \
          --n-trials 30 \
          --timeout 1800
    
    - name: Upload Best Params
      uses: actions/upload-artifact@v4
      with:
        name: best-params
        path: shared_model/best_params.json
```

## MLflow Tracking

### View Experiments

1. Buka MLflow UI: http://localhost:5000
2. Pilih experiment "SOFARE-AI-Tuning"
3. Compare trials di dashboard

### Metrics yang Di-track

- `objective_0`: Primary metric (accuracy)
- All hyperparameters
- Trial state (COMPLETE, PRUNED, FAILED)
- Duration

## Parameter Importance

Setelah tuning, analisis parameter importance:

```python
from hyperparameter_tuning import HyperparameterTuner

# After running tuning
importance = tuner.get_param_importance()
for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{param}: {score:.4f}")
```

Output example:
```
learning_rate: 0.2845
hidden_size: 0.1923
dropout: 0.1456
num_encoder_layers: 0.1234
...
```

## Troubleshooting

### Out of Memory

```python
# Reduce search space
config = TuningConfig(
    batch_size_options=[16, 32],  # Smaller batches
    hidden_size_range=(32, 128),  # Smaller models
)
```

### Slow Convergence

```python
# Increase trials and use better sampler
config = TuningConfig(
    n_trials=100,
    use_pruning=True,
    pruner_type="hyperband",
)
```

### Too Many Pruned Trials

```python
# Use median pruner instead
config = TuningConfig(
    pruner_type="median",  # Less aggressive
)
```

## File Structure

```
training/src/
├── hyperparameter_tuning.py   # Main tuning module
├── run_tuning.py              # CLI runner
└── train.py                   # Uses best params

shared_model/
├── best_params.json           # Saved best parameters
└── best_params_history.csv    # Full optimization history
```

## API Reference

### HyperparameterTuner

```python
class HyperparameterTuner:
    def __init__(
        config: TuningConfig,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        feature_cols: List[str],
        macro_cols: List[str],
        safe_cols: List[str]
    )
    
    def run() -> Dict[str, Any]:
        """Run optimization study."""
    
    def get_optimization_history() -> pd.DataFrame:
        """Get trials history."""
    
    def save_best_config(path: str) -> None:
        """Save best parameters."""
    
    def get_param_importance() -> Dict[str, float]:
        """Get parameter importance scores."""
```

### TuningConfig

```python
@dataclass
class TuningConfig:
    study_name: str = "sofare-hyperparameter-tuning"
    n_trials: int = 50
    timeout: Optional[int] = 3600
    n_jobs: int = 1
    direction: str = "maximize"
    # ... more fields
```
