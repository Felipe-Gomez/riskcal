"""Privacy parameter calibration module.

This module provides tools for calibrating privacy parameters (such as noise
multipliers) to achieve target privacy guarantees specified in terms of
operational risk metrics (advantage, FPR/FNR) or traditional (epsilon, delta)-DP.
"""

# Export core calibration interface
from riskcal.calibration.core import (
    PrivacyEvaluator,
    PrivacyMetrics,
    CalibrationTarget,
    CalibrationConfig,
    CalibrationResult,
    calibrate_parameter,
)

# Export DP-SGD specific calibration
from riskcal.calibration.dpsgd import (
    create_dpsgd_evaluator,
    create_dpsgd_epsilon_evaluator,
    find_noise_multiplier_for_advantage as find_noise_multiplier_for_advantage_dpsgd,
    find_noise_multiplier_for_err_rates as find_noise_multiplier_for_err_rates_dpsgd,
    get_advantage_for_dpsgd,
    get_beta_for_dpsgd,
)

# Export blackbox/accountant-based calibration
from riskcal.calibration.blackbox import (
    create_accountant_evaluator,
    find_noise_multiplier_for_epsilon_delta,
    find_noise_multiplier_for_advantage as find_noise_multiplier_for_advantage_blackbox,
    find_noise_multiplier_for_err_rates as find_noise_multiplier_for_err_rates_blackbox,
    CalibrationResult as BlackboxCalibrationResult,  # Legacy format
)

__all__ = [
    # Core calibration interface
    "PrivacyEvaluator",
    "PrivacyMetrics",
    "CalibrationTarget",
    "CalibrationConfig",
    "CalibrationResult",
    "calibrate_parameter",
    # DP-SGD
    "create_dpsgd_evaluator",
    "create_dpsgd_epsilon_evaluator",
    "find_noise_multiplier_for_advantage_dpsgd",
    "find_noise_multiplier_for_err_rates_dpsgd",
    "get_advantage_for_dpsgd",
    "get_beta_for_dpsgd",
    # Blackbox/Accountant
    "create_accountant_evaluator",
    "find_noise_multiplier_for_epsilon_delta",
    "find_noise_multiplier_for_advantage_blackbox",
    "find_noise_multiplier_for_err_rates_blackbox",
    "BlackboxCalibrationResult",
]
