"""DEPRECATED: Use riskcal.analysis instead.

This module is deprecated and will be removed in v2.0.0.

For PLRV functionality, use:
    from riskcal.analysis import PLRVs, get_beta_from_plrvs
"""

import warnings

warnings.warn(
    "The 'riskcal.plrv' module is deprecated and will be removed in v2.0.0. "
    "Use 'riskcal.analysis' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new locations for backward compatibility
from riskcal.analysis._plrv import PLRVs, get_beta

__all__ = [
    "PLRVs",
    "get_beta",
]
