"""DEPRECATED: Use riskcal.calibration.dpsgd and riskcal.accountants instead.

This module is deprecated and will be removed in v2.0.0.

For calibration functions, use:
    from riskcal.calibration.dpsgd import find_noise_multiplier_for_advantage, ...

For the CTDAccountant, use:
    from riskcal.accountants import CTDAccountant

For utility functions, use:
    from riskcal.utils import inverse_monotone_function
"""

import warnings

warnings.warn(
    "The 'riskcal.dpsgd' module is deprecated and will be removed in v2.0.0. "
    "Use 'riskcal.calibration.dpsgd' for calibration functions and "
    "'riskcal.accountants' for CTDAccountant instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new locations for backward compatibility
from riskcal.calibration.dpsgd import (
    find_noise_multiplier_for_advantage,
    find_noise_multiplier_for_err_rates,
    get_advantage_for_dpsgd,
    get_beta_for_dpsgd,
)
from riskcal.accountants.ctd import CTDAccountant
from riskcal.utils import inverse_monotone_function

__all__ = [
    "find_noise_multiplier_for_advantage",
    "find_noise_multiplier_for_err_rates",
    "get_advantage_for_dpsgd",
    "get_beta_for_dpsgd",
    "CTDAccountant",
    "inverse_monotone_function",
]
