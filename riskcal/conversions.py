from typing import Union, Callable

import numpy as np
from scipy import stats

from dp_accounting.pld import privacy_loss_distribution

from riskcal import plrv


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


def get_beta_from_privacy_profile(
    delta_func: Callable[[float], float],
    alpha: Union[float, np.ndarray],
    num_points: int = 1000,
    max_abs_epsilon: float = 20,
):
    """Derive the error rate (e.g., FNR) for a given privacy profile function and alpha.

    Vectorized version of Algorithm 4 in https://arxiv.org/abs/2302.07956
    """

    eps = np.linspace(-max_abs_epsilon, max_abs_epsilon, num_points)
    deltas = delta_func(eps)

    # Precompute exponentials to avoid repeated exp calculations
    exp_eps = np.exp(eps)  # shape: (len(eps),)
    exp_neg_eps = np.exp(-eps)  # shape: (len(eps),)

    def fdp_vectorized(alphas):
        """
        Computes the false negative rate at level alpha implied by the
        collection of (epsilon, delta) pairs, i.e. for each alpha the output is:

        max{0, 1 - deltas - np.exp(eps) * alpha, np.exp(-eps) * (1 - deltas - alpha)}

        This is done with fully vectorized operations, done in-place as much
        as possible to allow arrays for alpha, epsilon, and delta.
        """

        num_alphas = alphas.shape[0]

        # Create a 2D view of alphas for broadcasting
        alpha_2d = alphas[:, np.newaxis]  # shape: (len(alphas), 1)

        term1 = np.empty(
            (num_alphas, len(eps)), dtype=float
        )  # shape: (len(alphas), len(eps))

        # make term1 = max(0, 1 - deltas - np.exp(eps) * alphas)
        # using a temporary array
        term1[:] = (1.0 - deltas).reshape(1, -1)
        tmp = exp_eps.reshape(1, -1) * alpha_2d  # shape: (num_alphas, len(eps))
        term1 -= tmp
        np.maximum(term1, 0.0, out=term1)

        # use same tmp array to compute exp(-eps) * (1 - deltas - alpha)
        tmp[:] = (1.0 - deltas).reshape(1, -1)
        tmp -= alpha_2d
        tmp *= exp_neg_eps.reshape(1, -1)

        # Take elementwise maximum between term1 and tmp in place
        np.maximum(term1, tmp, out=term1)

        # Finally, take the max across the eps dimension, shape: (len(alphas),)
        return np.max(term1, axis=1)

    return fdp_vectorized(alpha)


def get_mu_from_privacy_profile(
    delta_func: Callable[[float], float],
    alpha_grid_step: float = 1e-4,
    num_points: int = 1000,
    max_abs_epsilon: float = 20,
):
    """Derive the mu for a given privacy profile function and alpha."""
    alpha = np.linspace(0, 1, int(1 / alpha_grid_step) + 1)
    beta = get_beta_from_privacy_profile(
        delta_func, alpha, num_points=num_points, max_abs_epsilon=max_abs_epsilon
    )
    mu_candidates = -stats.norm.ppf(alpha) - stats.norm.ppf(beta)

    # Replace infinity with NaN
    sanitized_mu_candidates = np.where(mu_candidates == np.inf, np.nan, mu_candidates)

    # Compute the maximum, ignoring NaN
    return np.nanmax(sanitized_mu_candidates)
