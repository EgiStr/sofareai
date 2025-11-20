import numpy as np
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger(__name__)

def check_drift(reference_data, current_data, threshold=0.05):
    """
    Checks for data drift using Kolmogorov-Smirnov test.
    
    Args:
        reference_data (np.array): Baseline data (e.g., training data from previous model).
        current_data (np.array): New data (e.g., recent window).
        threshold (float): P-value threshold. If p_value < threshold, drift is detected.
        
    Returns:
        bool: True if drift detected, False otherwise.
        dict: Details of the test (p_value, statistic).
    """
    # KS Test
    statistic, p_value = ks_2samp(reference_data, current_data)
    
    drift_detected = p_value < threshold
    
    if drift_detected:
        logger.warning(f"Drift detected! p-value: {p_value:.4f} < {threshold}")
    else:
        logger.info(f"No drift detected. p-value: {p_value:.4f}")
        
    return drift_detected, {"p_value": p_value, "statistic": statistic}
