"""DEPRECATED: Use riskcal.analysis instead.

This module is deprecated and will be removed in v2.0.0.

For risk/metric functions, use:
    from riskcal.analysis import get_beta_from_pld, get_advantage_from_pld, ...

For conversions, use:
    from riskcal.analysis import plrvs_from_pld, get_epsilon_for_err_rates
"""

import warnings

warnings.warn(
    "The 'riskcal.conversions' module is deprecated and will be removed in v2.0.0. "
    "Use 'riskcal.analysis' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new locations for backward compatibility
from riskcal.analysis.conversions import (
    plrvs_from_pld,
    get_epsilon_for_err_rates,
    get_beta_from_pld,
    get_beta_for_mu,
    get_beta_for_epsilon_delta,
    get_advantage_from_pld,
    get_advantage_for_mu,
    get_advantage_for_epsilon_delta,
    get_bayes_risk_from_pld,
    get_bayes_risk_for_mu,
)

__all__ = [
    "plrvs_from_pld",
    "get_epsilon_for_err_rates",
    "get_beta_from_pld",
    "get_beta_for_mu",
    "get_beta_for_epsilon_delta",
    "get_advantage_from_pld",
    "get_advantage_for_mu",
    "get_advantage_for_epsilon_delta",
    "get_bayes_risk_from_pld",
    "get_bayes_risk_for_mu",
]
