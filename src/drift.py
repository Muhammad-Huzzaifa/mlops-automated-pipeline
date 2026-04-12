"""
Drift detection module.
"""

from scipy.stats import ks_2samp

from src.config import Config


def detect_drift_per_column(old_data, new_data, column):
    """
    Detect drift using KS test.
    """
    stat, p_value = ks_2samp(old_data[column], new_data[column])
    return p_value < 0.05


def detect_drift(old_data, new_data):
    """
    Detect overall drift.

    Returns:
        bool: Drift detected or not
    """

    drift_count = 0
    results = {}

    for col in Config.DRIFT_DETECTING_COLUMNS:
        if col not in old_data.columns:
            continue

        drift = detect_drift_per_column(old_data, new_data, col)

        results[col] = drift

        if drift:
            drift_count += 1

    # critical columns check
    critical_drift = any(results.get(c, False) for c in Config.CRITICAL_COLUMNS)

    overall = critical_drift or drift_count >= Config.DRIFT_THRESHOLD

    print("Drift results:", results)

    return overall
