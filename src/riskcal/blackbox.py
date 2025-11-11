"""DEPRECATED: Use riskcal.calibration.blackbox instead.

This module is deprecated and will be removed in v2.0.0.

For accountant-based calibration functions, use:
    from riskcal.calibration.blackbox import (
        find_noise_multiplier_for_epsilon_delta,
        find_noise_multiplier_for_advantage,
        find_noise_multiplier_for_err_rates,
    )
"""

import warnings

warnings.warn(
    "The 'riskcal.blackbox' module is deprecated and will be removed in v2.0.0. "
    "Use 'riskcal.calibration.blackbox' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new locations for backward compatibility
from riskcal.calibration.blackbox import (
    find_noise_multiplier_for_epsilon_delta,
    find_noise_multiplier_for_advantage,
    find_noise_multiplier_for_err_rates,
    CalibrationResult,
)

__all__ = [
    "find_noise_multiplier_for_epsilon_delta",
    "find_noise_multiplier_for_advantage",
    "find_noise_multiplier_for_err_rates",
    "CalibrationResult",
]
