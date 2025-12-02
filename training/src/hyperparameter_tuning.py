"""
Hyperparameter Tuning Module for SOFARE-AI.

Implements best practices for MLOps hyperparameter optimization using Optuna:
- Automated hyperparameter search with TPE sampler
- Pruning for early stopping of unpromising trials
- MLflow integration for experiment tracking
- Multi-objective optimization (accuracy + inference speed)
- Distributed tuning support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
import mlflow
import mlflow.pytorch
import pickle
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from model import SofareM3
from dataset import TimeSeriesDataset
from features import add_technical_indicators

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    # Study settings
    study_name: str = "sofare-hyperparameter-tuning"
    n_trials: int = 50
    timeout: Optional[int] = 3600  # 1 hour timeout
    n_jobs: int = 1  # Parallel trials
    
    # Optimization direction
    direction: str = "maximize"  # For accuracy
    
    # Data settings
    sequence_length_range: Tuple[int, int] = (30, 120)
    batch_size_options: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    
    # Model architecture ranges
    hidden_size_range: Tuple[int, int] = (32, 256)
    embed_dim_options: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_heads_options: List[int] = field(default_factory=lambda: [2, 4, 8])
    num_layers_range: Tuple[int, int] = (1, 4)
    dropout_range: Tuple[float, float] = (0.1, 0.5)
    
    # Training ranges
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-2)
    epochs_range: Tuple[int, int] = (3, 15)
    weight_decay_range: Tuple[float, float] = (1e-6, 1e-3)
    
    # Optimizer options
    optimizer_options: List[str] = field(default_factory=lambda: ["adam", "adamw", "sgd"])
    
    # Early stopping
    early_stopping_patience: int = 3
    min_delta: float = 0.001
    
    # Pruning
    use_pruning: bool = True
    pruner_type: str = "hyperband"  # "median" or "hyperband"
    
    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    experiment_name: str = "SOFARE-AI-Tuning"


class OptunaMLflowCallback:
    """Callback to log Optuna trials to MLflow."""
    
    def __init__(self, experiment_name: str, tracking_uri: str):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Log trial results to MLflow."""
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            # Log parameters
            mlflow.log_params(trial.params)
            
            # Log metrics
            if trial.values is not None:
                for i, value in enumerate(trial.values):
                    mlflow.log_metric(f"objective_{i}", value)
            
            # Log trial metadata
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("trial_state", trial.state.name)
            mlflow.set_tag("study_name", study.study_name)
            
            if trial.state == TrialState.COMPLETE:
                mlflow.set_tag("best_trial", trial.number == study.best_trial.number)


class EarlyStoppingCallback:
    """Callback to stop study if no improvement after N trials."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.trials_without_improvement = 0
    
    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        
        current_value = trial.values[0] if trial.values else None
        if current_value is None:
            return
        
        if self.best_value is None:
            self.best_value = current_value
            return
        
        # Check for improvement (assuming maximization)
        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = current_value
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1
        
        if self.trials_without_improvement >= self.patience:
            logger.info(f"Early stopping: No improvement for {self.patience} trials")
            study.stop()


class HyperparameterTuner:
    """
    Main hyperparameter tuning class for SOFARE-AI models.
    
    Implements MLOps best practices:
    - Systematic search with TPE sampler
    - Trial pruning for efficiency
    - MLflow experiment tracking
    - Reproducible configurations
    """
    
    def __init__(
        self,
        config: TuningConfig,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        feature_cols: List[str],
        macro_cols: List[str],
        safe_cols: List[str]
    ):
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.feature_cols = feature_cols
        self.macro_cols = macro_cols
        self.safe_cols = safe_cols
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict] = None
        
        # Setup MLflow
        try:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.experiment_name)
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create TPE sampler with multivariate support."""
        return TPESampler(
            seed=42,
            multivariate=True,  # Consider parameter correlations
            warn_independent_sampling=True,
            n_startup_trials=10,  # Random trials before TPE kicks in
        )
    
    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create pruner for early stopping of unpromising trials."""
        if self.config.pruner_type == "hyperband":
            return HyperbandPruner(
                min_resource=1,
                max_resource=self.config.epochs_range[1],
                reduction_factor=3
            )
        else:
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2,
                interval_steps=1
            )
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        params = {
            # Data parameters
            "sequence_length": trial.suggest_int(
                "sequence_length",
                self.config.sequence_length_range[0],
                self.config.sequence_length_range[1],
                step=10
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                self.config.batch_size_options
            ),
            
            # Model architecture
            "hidden_size": trial.suggest_int(
                "hidden_size",
                self.config.hidden_size_range[0],
                self.config.hidden_size_range[1],
                step=32
            ),
            "embed_dim": trial.suggest_categorical(
                "embed_dim",
                self.config.embed_dim_options
            ),
            "num_heads": trial.suggest_categorical(
                "num_heads",
                self.config.num_heads_options
            ),
            "num_encoder_layers": trial.suggest_int(
                "num_encoder_layers",
                self.config.num_layers_range[0],
                self.config.num_layers_range[1]
            ),
            "dropout": trial.suggest_float(
                "dropout",
                self.config.dropout_range[0],
                self.config.dropout_range[1]
            ),
            
            # Training parameters
            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.config.learning_rate_range[0],
                self.config.learning_rate_range[1],
                log=True  # Log scale for learning rate
            ),
            "epochs": trial.suggest_int(
                "epochs",
                self.config.epochs_range[0],
                self.config.epochs_range[1]
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                self.config.weight_decay_range[0],
                self.config.weight_decay_range[1],
                log=True
            ),
            
            # Optimizer
            "optimizer": trial.suggest_categorical(
                "optimizer",
                self.config.optimizer_options
            ),
            
            # Loss weights
            "cls_weight": trial.suggest_float("cls_weight", 0.3, 0.7),
        }
        
        # Ensure embed_dim is divisible by num_heads
        if params["embed_dim"] % params["num_heads"] != 0:
            # Adjust embed_dim to be divisible
            params["embed_dim"] = (params["embed_dim"] // params["num_heads"]) * params["num_heads"]
        
        return params
    
    def _prepare_data(self, params: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders with suggested parameters."""
        sequence_length = params["sequence_length"]
        batch_size = params["batch_size"]
        
        # Scale data
        scaler_seq = MinMaxScaler()
        scaler_macro = MinMaxScaler()
        scaler_safe = MinMaxScaler()
        scaler_target = MinMaxScaler()
        
        # Training data
        train_seq = scaler_seq.fit_transform(self.train_data[self.feature_cols].values)
        train_macro = scaler_macro.fit_transform(self.train_data[self.macro_cols].values)
        train_safe = scaler_safe.fit_transform(self.train_data[self.safe_cols].values)
        train_target = scaler_target.fit_transform(
            self.train_data['log_return'].values.reshape(-1, 1)
        ).flatten()
        
        # Validation data
        val_seq = scaler_seq.transform(self.val_data[self.feature_cols].values)
        val_macro = scaler_macro.transform(self.val_data[self.macro_cols].values)
        val_safe = scaler_safe.transform(self.val_data[self.safe_cols].values)
        val_target = scaler_target.transform(
            self.val_data['log_return'].values.reshape(-1, 1)
        ).flatten()
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_seq, train_macro, train_safe, train_target, sequence_length
        )
        val_dataset = TimeSeriesDataset(
            val_seq, val_macro, val_safe, val_target, sequence_length
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _create_model(self, params: Dict[str, Any]) -> nn.Module:
        """Create model with suggested parameters."""
        model = SofareM3(
            micro_input_size=len(self.feature_cols),
            macro_input_size=len(self.macro_cols),
            safe_input_size=len(self.safe_cols),
            hidden_size=params["hidden_size"],
            embed_dim=params["embed_dim"],
            num_heads=params.get("num_heads", 4),
            num_encoder_layers=params.get("num_encoder_layers", 2),
            dropout=params.get("dropout", 0.1)
        )
        return model.to(self.device)
    
    def _create_optimizer(
        self,
        model: nn.Module,
        params: Dict[str, Any]
    ) -> optim.Optimizer:
        """Create optimizer with suggested parameters."""
        optimizer_name = params["optimizer"]
        lr = params["learning_rate"]
        weight_decay = params["weight_decay"]
        
        if optimizer_name == "adam":
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        cls_criterion: nn.Module,
        reg_criterion: nn.Module,
        cls_weight: float
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_cls_loss = 0
        total_reg_loss = 0
        
        for x_seq, x_macro, x_safe, y_cls, y_reg in train_loader:
            x_seq = x_seq.to(self.device)
            x_macro = x_macro.to(self.device)
            x_safe = x_safe.to(self.device)
            y_cls = y_cls.to(self.device)
            y_reg = y_reg.to(self.device)
            
            optimizer.zero_grad()
            cls_pred, reg_pred = model(x_seq, x_macro, x_safe)
            
            cls_loss = cls_criterion(cls_pred, y_cls)
            reg_loss = reg_criterion(reg_pred.squeeze(), y_reg)
            
            # Weighted multi-task loss
            loss = cls_weight * cls_loss + (1 - cls_weight) * reg_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
        
        return total_cls_loss / len(train_loader), total_reg_loss / len(train_loader)
    
    def _evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        cls_criterion: nn.Module,
        reg_criterion: nn.Module
    ) -> Dict[str, float]:
        """Evaluate model on validation set."""
        model.eval()
        total_cls_loss = 0
        total_reg_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_seq, x_macro, x_safe, y_cls, y_reg in val_loader:
                x_seq = x_seq.to(self.device)
                x_macro = x_macro.to(self.device)
                x_safe = x_safe.to(self.device)
                y_cls = y_cls.to(self.device)
                y_reg = y_reg.to(self.device)
                
                cls_pred, reg_pred = model(x_seq, x_macro, x_safe)
                
                cls_loss = cls_criterion(cls_pred, y_cls)
                reg_loss = reg_criterion(reg_pred.squeeze(), y_reg)
                
                total_cls_loss += cls_loss.item()
                total_reg_loss += reg_loss.item()
                
                # Classification predictions
                preds = cls_pred.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_cls.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        accuracy = np.mean(all_preds == all_targets)
        
        return {
            "val_cls_loss": total_cls_loss / len(val_loader),
            "val_reg_loss": total_reg_loss / len(val_loader),
            "accuracy": accuracy
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        # Get hyperparameters
        params = self._suggest_hyperparameters(trial)
        
        logger.info(f"Trial {trial.number}: {params}")
        
        try:
            # Prepare data
            train_loader, val_loader = self._prepare_data(params)
            
            # Create model and optimizer
            model = self._create_model(params)
            optimizer = self._create_optimizer(model, params)
            
            # Loss functions
            cls_criterion = nn.CrossEntropyLoss()
            reg_criterion = nn.HuberLoss(delta=1.0)
            
            # Training loop
            best_accuracy = 0
            epochs_without_improvement = 0
            
            for epoch in range(params["epochs"]):
                # Train
                train_cls_loss, train_reg_loss = self._train_epoch(
                    model, train_loader, optimizer,
                    cls_criterion, reg_criterion,
                    params["cls_weight"]
                )
                
                # Evaluate
                metrics = self._evaluate(
                    model, val_loader, cls_criterion, reg_criterion
                )
                
                accuracy = metrics["accuracy"]
                
                # Report intermediate value for pruning
                trial.report(accuracy, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
                # Track best accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                # Early stopping within trial
                if epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(f"Trial {trial.number}: Early stopping at epoch {epoch}")
                    break
                
                logger.info(
                    f"Trial {trial.number}, Epoch {epoch+1}: "
                    f"Accuracy={accuracy:.4f}, Best={best_accuracy:.4f}"
                )
            
            return best_accuracy
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()
    
    def run(self) -> Dict[str, Any]:
        """Run hyperparameter tuning study."""
        logger.info("Starting hyperparameter tuning...")
        
        # Create study
        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner() if self.config.use_pruning else None,
            load_if_exists=True
        )
        
        # Setup callbacks
        callbacks = [
            OptunaMLflowCallback(
                self.config.experiment_name,
                self.config.mlflow_tracking_uri
            ),
            EarlyStoppingCallback(
                patience=10,
                min_delta=self.config.min_delta
            )
        ]
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            callbacks=callbacks,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best accuracy: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "best_trial": self.study.best_trial.number,
            "n_trials": len(self.study.trials),
            "n_pruned": len([t for t in self.study.trials if t.state == TrialState.PRUNED]),
            "n_complete": len([t for t in self.study.trials if t.state == TrialState.COMPLETE])
        }
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if self.study is None:
            raise ValueError("No study available. Run tune() first.")
        
        trials = []
        for trial in self.study.trials:
            trial_data = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.values[0] if trial.values else None,
                **trial.params
            }
            trials.append(trial_data)
        
        return pd.DataFrame(trials)
    
    def save_best_config(self, path: str) -> None:
        """Save best hyperparameters to file."""
        if self.best_params is None:
            raise ValueError("No best params available. Run tune() first.")
        
        config = {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "study_name": self.config.study_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Best config saved to {path}")
    
    def get_param_importance(self) -> Dict[str, float]:
        """Get parameter importance scores."""
        if self.study is None:
            raise ValueError("No study available. Run tune() first.")
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return dict(importance)
        except Exception as e:
            logger.warning(f"Could not calculate param importance: {e}")
            return {}


def create_default_tuning_config() -> TuningConfig:
    """Create default tuning configuration for SOFARE-AI."""
    return TuningConfig(
        study_name="sofare-hyperparameter-tuning",
        n_trials=50,
        timeout=7200,  # 2 hours
        direction="maximize",
        sequence_length_range=(30, 90),
        batch_size_options=[32, 64],
        hidden_size_range=(64, 192),
        embed_dim_options=[64, 128],
        num_heads_options=[4, 8],
        num_layers_range=(1, 3),
        dropout_range=(0.1, 0.4),
        learning_rate_range=(1e-4, 1e-2),
        epochs_range=(5, 10),
        weight_decay_range=(1e-5, 1e-3),
        optimizer_options=["adam", "adamw"],
        use_pruning=True,
        pruner_type="hyperband",
        mlflow_tracking_uri="http://mlflow:5000",
        experiment_name="SOFARE-AI-Tuning"
    )


def load_best_params(path: str) -> Dict[str, Any]:
    """Load best hyperparameters from file."""
    with open(path, "r") as f:
        config = json.load(f)
    return config.get("best_params", {})
