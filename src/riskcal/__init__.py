"""riskcal: Privacy risk analysis and calibration for differential privacy.

This library provides tools for computing f-DP trade-off curves for differential
privacy mechanisms, and calibrating their noise to operational privacy risk
(attack accuracy/advantage, or attack TPR and FPR) instead of (epsilon, delta).

Recommended usage:
    from riskcal import analysis  # For computing privacy metrics
    from riskcal import calibration  # For noise calibration
    from riskcal import accountants  # For privacy accountants
"""

# New recommended API (no deprecation warnings)
from riskcal import analysis
from riskcal import calibration
from riskcal import accountants

# Legacy module imports (will emit deprecation warnings)
from riskcal import dpsgd  # noqa: F401
from riskcal import conversions  # noqa: F401
from riskcal import blackbox  # noqa: F401
from riskcal import plrv  # noqa: F401

# Legacy top-level imports for backward compatibility
# These import from the deprecated modules, so they will also emit warnings
from riskcal.conversions import get_advantage_from_pld, get_beta_from_pld
from riskcal.dpsgd import (
    get_advantage_for_dpsgd,
    get_beta_for_dpsgd,
    find_noise_multiplier_for_advantage,
    find_noise_multiplier_for_err_rates,
    CTDAccountant,
)

__all__ = [
    # New API (recommended)
    "analysis",
    "calibration",
    "accountants",
    # Legacy modules (deprecated)
    "dpsgd",
    "conversions",
    "blackbox",
    "plrv",
    # Legacy top-level exports (deprecated)
    "get_advantage_from_pld",
    "get_beta_from_pld",
    "get_advantage_for_dpsgd",
    "get_beta_for_dpsgd",
    "find_noise_multiplier_for_advantage",
    "find_noise_multiplier_for_err_rates",
    "CTDAccountant",
]
