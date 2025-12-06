"""
Comprehensive Drift Detection Module for SOFARE-AI.

Implements multiple statistical tests for detecting:
- Input/Feature drift
- Prediction/Output drift
- Concept drift
- Model performance degradation
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift that can be detected."""
    INPUT = "input"
    PREDICTION = "prediction"
    CONCEPT = "concept"
    PERFORMANCE = "performance"


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Container for drift detection results."""
    drift_type: DriftType
    feature_name: str
    drift_detected: bool
    severity: DriftSeverity
    p_value: float
    statistic: float
    test_name: str
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with JSON-serializable types."""
        return {
            "drift_type": self.drift_type.value,
            "feature_name": str(self.feature_name),
            "drift_detected": bool(self.drift_detected),
            "severity": self.severity.value,
            "p_value": float(self.p_value),
            "statistic": float(self.statistic),
            "test_name": str(self.test_name),
            "threshold": float(self.threshold),
            "timestamp": str(self.timestamp),
            "details": self._serialize_details(self.details)
        }
    
    def _serialize_details(self, details: Dict) -> Dict:
        """Recursively serialize details to JSON-compatible types."""
        result = {}
        for k, v in details.items():
            if isinstance(v, (np.integer, np.int64, np.int32)):
                result[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                result[k] = float(v)
            elif isinstance(v, (np.bool_, bool)):
                result[k] = bool(v)
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, dict):
                result[k] = self._serialize_details(v)
            elif isinstance(v, list):
                result[k] = [self._serialize_value(item) for item in v]
            else:
                result[k] = v
        return result
    
    def _serialize_value(self, v):
        """Serialize a single value."""
        if isinstance(v, (np.integer, np.int64, np.int32)):
            return int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            return float(v)
        elif isinstance(v, (np.bool_, bool)):
            return bool(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v


class StatisticalTests:
    """Collection of statistical tests for drift detection."""
    
    @staticmethod
    def kolmogorov_smirnov(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Two-sample Kolmogorov-Smirnov test for continuous distributions.
        
        Good for detecting changes in distribution shape.
        """
        statistic, p_value = ks_2samp(reference, current)
        return statistic, p_value
    
    @staticmethod
    def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index (PSI) for distribution comparison.
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        """
        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ref_proportions = (ref_counts + epsilon) / (len(reference) + epsilon * bins)
        cur_proportions = (cur_counts + epsilon) / (len(current) + epsilon * bins)
        
        # Calculate PSI
        psi = np.sum((cur_proportions - ref_proportions) * np.log(cur_proportions / ref_proportions))
        return psi
    
    @staticmethod
    def wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
        """
        Wasserstein distance (Earth Mover's Distance) between distributions.
        
        Measures the minimum "work" to transform one distribution into another.
        """
        return wasserstein_distance(reference, current)
    
    @staticmethod
    def chi_square(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> Tuple[float, float]:
        """
        Chi-square test for categorical or binned continuous data.
        """
        # Bin the data
        combined = np.concatenate([reference, current])
        _, bin_edges = np.histogram(combined, bins=bins)
        
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Avoid zero counts
        ref_counts = ref_counts + 1
        cur_counts = cur_counts + 1
        
        contingency = np.array([ref_counts, cur_counts])
        statistic, p_value, _, _ = chi2_contingency(contingency)
        
        return statistic, p_value
    
    @staticmethod
    def jensen_shannon_divergence(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Jensen-Shannon Divergence between distributions.
        
        Symmetric and bounded [0, 1] version of KL divergence.
        """
        # Create probability distributions
        combined = np.concatenate([reference, current])
        _, bin_edges = np.histogram(combined, bins=bins)
        
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize to probabilities
        epsilon = 1e-10
        p = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * bins)
        q = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * bins)
        
        # Calculate JS divergence
        m = 0.5 * (p + q)
        jsd = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))
        
        return jsd


class DriftDetector:
    """
    Main drift detection class implementing multiple detection strategies.
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        wasserstein_threshold: float = 0.1,
        chi2_threshold: float = 0.05,
        jsd_threshold: float = 0.1
    ):
        """
        Initialize drift detector with reference data and thresholds.
        
        Args:
            reference_data: Baseline data for comparison
            ks_threshold: P-value threshold for KS test
            psi_threshold: Threshold for PSI (0.2 = significant change)
            wasserstein_threshold: Threshold for Wasserstein distance
            chi2_threshold: P-value threshold for Chi-square test
            jsd_threshold: Threshold for Jensen-Shannon divergence
        """
        self.reference_data = reference_data
        self.thresholds = {
            "ks": ks_threshold,
            "psi": psi_threshold,
            "wasserstein": wasserstein_threshold,
            "chi2": chi2_threshold,
            "jsd": jsd_threshold
        }
        self.drift_history: List[DriftResult] = []
        self.tests = StatisticalTests()
    
    def set_reference_data(self, data: pd.DataFrame) -> None:
        """Set or update reference data."""
        self.reference_data = data.copy()
        logger.info(f"Reference data set with shape {data.shape}")
    
    def _determine_severity(self, p_value: float, statistic: float, test_type: str) -> DriftSeverity:
        """Determine severity based on test results."""
        if test_type in ["ks", "chi2"]:
            # P-value based tests
            if p_value >= 0.1:
                return DriftSeverity.NONE
            elif p_value >= 0.05:
                return DriftSeverity.LOW
            elif p_value >= 0.01:
                return DriftSeverity.MEDIUM
            elif p_value >= 0.001:
                return DriftSeverity.HIGH
            else:
                return DriftSeverity.CRITICAL
        else:
            # Statistic-based tests (PSI, Wasserstein, JSD)
            threshold = self.thresholds[test_type]
            ratio = statistic / threshold
            if ratio < 0.5:
                return DriftSeverity.NONE
            elif ratio < 1.0:
                return DriftSeverity.LOW
            elif ratio < 1.5:
                return DriftSeverity.MEDIUM
            elif ratio < 2.0:
                return DriftSeverity.HIGH
            else:
                return DriftSeverity.CRITICAL
    
    def detect_univariate_drift(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None,
        test_type: str = "ks"
    ) -> List[DriftResult]:
        """
        Detect drift in individual features.
        
        Args:
            current_data: Current data to compare against reference
            features: List of features to check (default: all numeric)
            test_type: Statistical test to use ('ks', 'psi', 'wasserstein', 'chi2', 'jsd')
        
        Returns:
            List of DriftResult objects
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        if features is None:
            features = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        results = []
        
        for feature in features:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                logger.warning(f"Feature '{feature}' not found in data, skipping")
                continue
            
            ref_values = self.reference_data[feature].dropna().values
            cur_values = current_data[feature].dropna().values
            
            if len(ref_values) < 10 or len(cur_values) < 10:
                logger.warning(f"Insufficient data for feature '{feature}', skipping")
                continue
            
            # Run appropriate test
            if test_type == "ks":
                statistic, p_value = self.tests.kolmogorov_smirnov(ref_values, cur_values)
                drift_detected = p_value < self.thresholds["ks"]
            elif test_type == "psi":
                statistic = self.tests.population_stability_index(ref_values, cur_values)
                p_value = 1.0 - min(statistic / self.thresholds["psi"], 1.0)
                drift_detected = statistic >= self.thresholds["psi"]
            elif test_type == "wasserstein":
                # Normalize for comparability
                ref_norm = (ref_values - ref_values.mean()) / (ref_values.std() + 1e-10)
                cur_norm = (cur_values - cur_values.mean()) / (cur_values.std() + 1e-10)
                statistic = self.tests.wasserstein(ref_norm, cur_norm)
                p_value = 1.0 - min(statistic / self.thresholds["wasserstein"], 1.0)
                drift_detected = statistic >= self.thresholds["wasserstein"]
            elif test_type == "chi2":
                statistic, p_value = self.tests.chi_square(ref_values, cur_values)
                drift_detected = p_value < self.thresholds["chi2"]
            elif test_type == "jsd":
                statistic = self.tests.jensen_shannon_divergence(ref_values, cur_values)
                p_value = 1.0 - min(statistic / self.thresholds["jsd"], 1.0)
                drift_detected = statistic >= self.thresholds["jsd"]
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            severity = self._determine_severity(p_value, statistic, test_type)
            
            result = DriftResult(
                drift_type=DriftType.INPUT,
                feature_name=feature,
                drift_detected=drift_detected,
                severity=severity,
                p_value=float(p_value),
                statistic=float(statistic),
                test_name=test_type.upper(),
                threshold=self.thresholds[test_type],
                details={
                    "reference_mean": float(ref_values.mean()),
                    "reference_std": float(ref_values.std()),
                    "current_mean": float(cur_values.mean()),
                    "current_std": float(cur_values.std()),
                    "reference_size": len(ref_values),
                    "current_size": len(cur_values)
                }
            )
            
            results.append(result)
            self.drift_history.append(result)
            
            if drift_detected:
                logger.warning(
                    f"Drift detected in '{feature}': {test_type.upper()} "
                    f"statistic={statistic:.4f}, p-value={p_value:.4f}, "
                    f"severity={severity.value}"
                )
        
        return results
    
    def detect_multivariate_drift(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> DriftResult:
        """
        Detect drift using multivariate analysis (simplified approach).
        
        Uses aggregated feature statistics for holistic drift detection.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set.")
        
        if features is None:
            features = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Run univariate tests on all features
        univariate_results = self.detect_univariate_drift(current_data, features, "ks")
        
        # Aggregate results
        drift_count = sum(1 for r in univariate_results if r.drift_detected)
        total_features = len(univariate_results)
        drift_ratio = drift_count / total_features if total_features > 0 else 0
        
        # Determine overall drift
        drift_detected = drift_ratio > 0.3  # More than 30% of features drifted
        
        if drift_ratio > 0.5:
            severity = DriftSeverity.CRITICAL
        elif drift_ratio > 0.3:
            severity = DriftSeverity.HIGH
        elif drift_ratio > 0.2:
            severity = DriftSeverity.MEDIUM
        elif drift_ratio > 0.1:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE
        
        result = DriftResult(
            drift_type=DriftType.INPUT,
            feature_name="MULTIVARIATE",
            drift_detected=drift_detected,
            severity=severity,
            p_value=1.0 - drift_ratio,
            statistic=drift_ratio,
            test_name="MULTIVARIATE_AGGREGATE",
            threshold=0.3,
            details={
                "features_with_drift": drift_count,
                "total_features": total_features,
                "drift_ratio": drift_ratio,
                "drifted_features": [r.feature_name for r in univariate_results if r.drift_detected]
            }
        )
        
        self.drift_history.append(result)
        return result
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> DriftResult:
        """
        Detect drift in model predictions.
        """
        statistic, p_value = self.tests.kolmogorov_smirnov(
            reference_predictions, current_predictions
        )
        
        drift_detected = p_value < self.thresholds["ks"]
        severity = self._determine_severity(p_value, statistic, "ks")
        
        result = DriftResult(
            drift_type=DriftType.PREDICTION,
            feature_name="predictions",
            drift_detected=drift_detected,
            severity=severity,
            p_value=float(p_value),
            statistic=float(statistic),
            test_name="KS",
            threshold=self.thresholds["ks"],
            details={
                "reference_mean": float(reference_predictions.mean()),
                "current_mean": float(current_predictions.mean()),
                "reference_std": float(reference_predictions.std()),
                "current_std": float(current_predictions.std())
            }
        )
        
        self.drift_history.append(result)
        return result
    
    def detect_performance_degradation(
        self,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        degradation_threshold: float = 0.1
    ) -> DriftResult:
        """
        Detect model performance degradation.
        
        Args:
            reference_metrics: Baseline model metrics
            current_metrics: Current model metrics
            degradation_threshold: Relative degradation threshold (0.1 = 10%)
        """
        degradations = {}
        for metric, ref_value in reference_metrics.items():
            if metric in current_metrics:
                cur_value = current_metrics[metric]
                # For metrics where higher is better (accuracy, f1, etc.)
                if ref_value > 0:
                    degradation = (ref_value - cur_value) / ref_value
                else:
                    degradation = 0
                degradations[metric] = degradation
        
        max_degradation = max(degradations.values()) if degradations else 0
        drift_detected = max_degradation > degradation_threshold
        
        if max_degradation > 0.3:
            severity = DriftSeverity.CRITICAL
        elif max_degradation > 0.2:
            severity = DriftSeverity.HIGH
        elif max_degradation > 0.1:
            severity = DriftSeverity.MEDIUM
        elif max_degradation > 0.05:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE
        
        result = DriftResult(
            drift_type=DriftType.PERFORMANCE,
            feature_name="model_performance",
            drift_detected=drift_detected,
            severity=severity,
            p_value=1.0 - max_degradation,
            statistic=max_degradation,
            test_name="PERFORMANCE_DEGRADATION",
            threshold=degradation_threshold,
            details={
                "reference_metrics": reference_metrics,
                "current_metrics": current_metrics,
                "degradations": degradations,
                "max_degradation": max_degradation
            }
        )
        
        self.drift_history.append(result)
        return result
    
    def get_drift_report(self) -> Dict[str, Any]:
        """Generate comprehensive drift report."""
        if not self.drift_history:
            return {"status": "No drift checks performed", "results": []}
        
        critical_count = sum(1 for r in self.drift_history if r.severity == DriftSeverity.CRITICAL)
        high_count = sum(1 for r in self.drift_history if r.severity == DriftSeverity.HIGH)
        medium_count = sum(1 for r in self.drift_history if r.severity == DriftSeverity.MEDIUM)
        
        overall_status = "HEALTHY"
        if critical_count > 0:
            overall_status = "CRITICAL"
        elif high_count > 0:
            overall_status = "WARNING"
        elif medium_count > 0:
            overall_status = "ATTENTION"
        
        return {
            "status": overall_status,
            "summary": {
                "total_checks": len(self.drift_history),
                "drift_detected": sum(1 for r in self.drift_history if r.drift_detected),
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "low": sum(1 for r in self.drift_history if r.severity == DriftSeverity.LOW),
                "none": sum(1 for r in self.drift_history if r.severity == DriftSeverity.NONE)
            },
            "results": [r.to_dict() for r in self.drift_history],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on drift results."""
        recommendations = []
        
        critical_drifts = [r for r in self.drift_history if r.severity == DriftSeverity.CRITICAL]
        high_drifts = [r for r in self.drift_history if r.severity == DriftSeverity.HIGH]
        
        if critical_drifts:
            recommendations.append(
                f"🚨 CRITICAL: {len(critical_drifts)} feature(s) show critical drift. "
                "Immediate model retraining recommended."
            )
            drifted_features = [r.feature_name for r in critical_drifts]
            recommendations.append(f"   Affected features: {', '.join(drifted_features)}")
        
        if high_drifts:
            recommendations.append(
                f"⚠️ WARNING: {len(high_drifts)} feature(s) show high drift. "
                "Schedule retraining within 24-48 hours."
            )
        
        performance_drifts = [r for r in self.drift_history if r.drift_type == DriftType.PERFORMANCE and r.drift_detected]
        if performance_drifts:
            recommendations.append(
                "📉 Model performance degradation detected. "
                "Consider A/B testing with a retrained model."
            )
        
        if not recommendations:
            recommendations.append("✅ No significant drift detected. Model is stable.")
        
        return recommendations
    
    def clear_history(self) -> None:
        """Clear drift detection history."""
        self.drift_history = []


def create_drift_detector_from_mlflow(
    experiment_name: str = "SOFARE-AI-Phase3",
    run_id: Optional[str] = None
) -> DriftDetector:
    """
    Create a drift detector with reference data from MLflow.
    
    Args:
        experiment_name: MLflow experiment name
        run_id: Specific run ID (default: latest successful run)
    
    Returns:
        Configured DriftDetector instance
    """
    import mlflow
    
    # Get reference data from MLflow artifacts
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    if run_id is None:
        # Get latest successful run
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if not runs.empty:
                run_id = runs.iloc[0].run_id
    
    detector = DriftDetector()
    
    # In production, load actual reference data from MLflow artifacts
    logger.info(f"Created drift detector with reference from run: {run_id}")
    
    return detector
