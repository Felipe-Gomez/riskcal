from typing import Union

import numpy as np
from scipy import stats

from dp_accounting.pld import privacy_loss_distribution

from . import plrv
from . import zcdp


def plrvs_from_pld(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
) -> plrv.PLRVs:
    """
    Extract PLRVs from a Google PLD object.
    """

    def _get_plrv(pld):
        pld = pld.to_dense_pmf()
        pmf = pld._probs
        lower_loss = pld._lower_loss
        return lower_loss, pmf

    lower_loss_Y, pmf_Y = _get_plrv(pld._pmf_remove)
    lower_loss_Z, pmf_Z = _get_plrv(pld._pmf_add)
    return plrv.PLRVs(
        lower_loss_Y=lower_loss_Y, lower_loss_Z=lower_loss_Z, pmf_Y=pmf_Y, pmf_Z=pmf_Z
    )


def get_beta_from_pld(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
    alpha: Union[float, np.ndarray],
    alpha_grid_step: float = 1e-4,
) -> Union[float, np.ndarray]:
    """
    Get the trade-off curve for a given PLD object.
    """
    return plrv.get_beta(plrvs_from_pld(pld), alpha, alpha_grid_step=alpha_grid_step)


def get_advantage_from_pld(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
) -> float:
    """
    Get advantage for a given PLD object.
    """
    return pld.get_delta_for_epsilon(0)


def get_advantage_for_mu(mu: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Get advantage for Gaussian Differential Privacy.

    Corollary 2.13 in https://arxiv.org/abs/1905.02383
    """
    return stats.norm.cdf(mu / 2) - stats.norm.cdf(-mu / 2)


def get_beta_for_mu(
    mu: float, alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Get beta for Gaussian Differential Privacy.

    Eq. 6 in https://arxiv.org/abs/1905.02383
    """
    return stats.norm.cdf(stats.norm.ppf(1 - alpha) - mu)


def get_beta_for_epsilon_delta(
    epsilon: float, delta: float, alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Derive the error rate (e.g., FNR) for a given epsilon, delta, and the other error rate (e.g., FPR).

    Eq. 5 in https://arxiv.org/abs/1905.02383

    >>> np.round(get_beta_for_epsilon_delta(1.0, 0.001, 0.8), 3)
    0.073
    """
    form1 = np.array(1 - delta - np.exp(epsilon) * alpha)
    form2 = np.array(np.exp(-epsilon) * (1 - delta - alpha))
    return np.maximum.reduce([form1, form2, np.zeros_like(form1)])


def get_advantage_for_epsilon_delta(epsilon: float, delta: float) -> float:
    """Derive advantage from a given epsilon and delta.

    >>> np.round(get_advantage_for_epsilon_delta(0., 0.001), 3)
    0.001
    """
    return (np.exp(epsilon) + 2 * delta - 1) / (np.exp(epsilon) + 1)


def get_epsilon_for_err_rates(delta: float, alpha: float, beta: float) -> float:
    """Derive epsilon from given FPR/FNR error rates and delta.

    >>> np.round(get_epsilon_for_err_rates(0.001, 0.001, 0.8), 3)
    5.293
    """
    epsilon1 = np.log((1 - delta - alpha) / beta)
    epsilon2 = np.log((1 - delta - beta) / alpha)
    return np.maximum.reduce([epsilon1, epsilon2, np.zeros_like(epsilon1)])


def get_advantage_for_rho(
    rho: float,
    alpha_grid_step: float = 1e-4,
    orders: Union[np.ndarray, int] = None,
    verbose: bool = False,
) -> np.ndarray:
    alpha = np.linspace(0, 1, int(1 / alpha_grid_step))
    beta = zcdp.get_beta(
        rho, alpha, alpha_grid_step=alpha_grid_step, orders=orders, verbose=verbose
    )

    for alpha_val, beta_val in reversed(list(zip(alpha, beta))):
        if np.abs(beta_val - alpha_val) <= alpha_grid_step:
            # As we are going from right to left, we will pick a lower beta,
            # resulting in a (slight) overestimate of advantage, which is what we want.
            alpha_star = beta_val

    return 1 - 2 * alpha_star


def get_beta_for_rho(
    rho: float,
    alpha: Union[float, np.ndarray],
    alpha_grid_step: float = 1e-4,
    orders: Union[np.ndarray, int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Derive the error rate (e.g., FNR) for a given level of rho-zCDP, and the other error rate (e.g., FPR)."""
    return zcdp.get_beta(
        rho, alpha, alpha_grid_step=alpha_grid_step, orders=orders, verbose=verbose
    )
