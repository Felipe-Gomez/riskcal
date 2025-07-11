from typing import Union

import warnings
import numpy as np

from scipy import stats
from scipy import optimize
from dp_accounting.pld import privacy_loss_distribution

from riskcal import plrv
from riskcal.utils import _ensure_array


def plrvs_from_pld(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
) -> plrv.PLRVs:
    """
    Extract PLRVs from a PLD object. See the docstring in plrv.py for notation details.

    A Google PLD object comes with a remove PLD and an add PLD.
    Without loss of generality, the code below chooses the remove pld to be Y
    and the add pld to be X. Moreover, the add pld follows Z = log Q(o') / P(o'),
    o' ~ Q. This means X = -Z, and this minus sign is accounted for in this function.

    Since the integer z_0 = lower_loss_Z is the smallest value of Z, it follows that
    -z_0 is the largest value for X and -(z_1 + len(pmf_Z) - 1) is the smallest value of X,
    and hence we set this to be x_0.
    """

    def _get_plrv(pld):
        pld = pld.to_dense_pmf()
        pmf = pld._probs
        lower_loss = pld._lower_loss
        infinity_mass = pld._infinity_mass
        return lower_loss, infinity_mass, pmf

    lower_loss_Y, infinity_mass_Y, pmf_Y = _get_plrv(pld._pmf_remove)
    lower_loss_Z, infinity_mass_Z, pmf_Z = _get_plrv(pld._pmf_add)

    upper_loss_Z = lower_loss_Z + len(pmf_Z) - 1

    # clean pmfs. Sometimes float errors cause probs to be negative
    pmf_Y = np.where(pmf_Y < 0, 0, pmf_Y)
    pmf_Z = np.where(pmf_Z < 0, 0, pmf_Z)

    pmf_Y = pmf_Y * (1 - infinity_mass_Y) / np.sum(pmf_Y)
    pmf_Z = pmf_Z * (1 - infinity_mass_Z) / np.sum(pmf_Z)

    is_symmetric = pld._symmetric
    return plrv.PLRVs(
        y0=lower_loss_Y,
        x0=-upper_loss_Z,
        pmf_Y=pmf_Y,
        pmf_X=pmf_Z[::-1],
        minus_infinity_mass_X=infinity_mass_Z,
        infinity_mass_Y=infinity_mass_Y,
        is_symmetric=is_symmetric,
    )


def get_beta_from_pld(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
    alpha: Union[float, np.ndarray] = None,
    alphas: Union[float, np.ndarray] = None,  # Deprecated.
) -> Union[float, np.ndarray]:
    """
    Get the trade-off curve for a given PLD object.
    """
    if alpha is None and alphas is None:
        raise ValueError("Must specify alpha.")

    elif alpha is not None and alphas is not None:
        raise ValueError("Must pass either alpha or alphas.")

    elif alphas is not None:
        warnings.warn(
            "Parameter 'alphas' is deprecated and will be removed in a future version. "
            "Use 'alpha' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        alpha = alphas

    return plrv.get_beta(plrvs_from_pld(pld), alpha)


def get_beta_for_mu(
    mu: float, alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Get the trade-off curve for a given Gaussian differential privacy parameter.

    Uses Eq. 6 in https://arxiv.org/abs/1905.02383 along with identity:
    Phi^{-1}(1-alpha) = - Phi^{-1}(alpha)
    """
    return stats.norm.cdf(-stats.norm.ppf(alpha) - mu)


def get_beta_for_epsilon_delta(
    epsilon: float, delta: float, alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Get the trade-off curve for given approximate differential privacy parameters.

    Uses Eq. 5 in https://arxiv.org/abs/1905.02383

    >>> np.round(get_beta_for_epsilon_delta(1.0, 0.001, 0.8), 3)
    0.073
    """
    form1 = np.array(1 - delta - np.exp(epsilon) * alpha)
    form2 = np.array(np.exp(-epsilon) * (1 - delta - alpha))
    return np.maximum.reduce([form1, form2, np.zeros_like(form1)])


def get_advantage_from_pld(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
) -> float:
    """
    Get advantage for a given PLD object.
    """
    return pld.get_delta_for_epsilon(0)


def get_advantage_for_mu(mu: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Get advantage for a given Gaussian differential privacy parameter.

    Corollary 2.13 in https://arxiv.org/abs/1905.02383
    """
    return stats.norm.cdf(mu / 2) - stats.norm.cdf(-mu / 2)


def get_advantage_for_epsilon_delta(epsilon: float, delta: float) -> float:
    """Get advantage for given approximate differential privacy parameters.

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


def get_bayes_risk_for_mu(mu, prior):
    """
    Derive Bayes Risk for a given Gaussian Differential Privacy parameter and prior.

    Based on Eq. 2 in https://arxiv.org/abs/2406.08918
    """
    assert mu >= 0, "mu must be >=0"
    prior, is_scalar = _ensure_array(prior)

    bayes_risk = []
    for prior_val in prior:
        pi = np.array([1 - prior_val, prior_val])
        result = np.zeros_like(pi, dtype=float)
        # Set result to 0 where pi is 1
        result[pi == 1] = 0

        mask = (pi != 0) & (pi != 1)
        a = ((-(mu**2) / 2) - np.log(pi[mask]) - np.log(-1 / (pi[mask] - 1))) / mu
        b = ((mu**2 / 2) - np.log(pi[mask]) - np.log(-1 / (pi[mask] - 1))) / mu
        result[mask] = pi[mask] * stats.norm.cdf(a) + (1 - pi[mask]) * stats.norm.sf(b)
        bayes_risk.append(result[1])

    if is_scalar:
        return bayes_risk[0]
    else:
        return np.array(bayes_risk)


def get_bayes_risk_for_mu(mu, prior):
    """
    Derive Bayes Risk for a given Gaussian Differential Privacy parameter and prior.

    Based on Eq. 2 in https://arxiv.org/abs/2406.08918
    """
    prior, is_scalar = _ensure_array(prior)

    bayes_risk = []
    for prior_val in prior:
        pi = np.array([1 - prior_val, prior_val])
        result = np.zeros_like(pi, dtype=float)
        result[pi == 1] = 0

        mask = (pi != 0) & (pi != 1)
        a = ((-(mu**2) / 2) - np.log(pi[mask]) - np.log(-1 / (pi[mask] - 1))) / mu
        b = ((mu**2 / 2) - np.log(pi[mask]) - np.log(-1 / (pi[mask] - 1))) / mu
        result[mask] = pi[mask] * stats.norm.cdf(a) + (1 - pi[mask]) * stats.norm.sf(b)
        bayes_risk.append(result[1])

    if is_scalar:
        return bayes_risk[0]
    else:
        return np.array(bayes_risk)


def get_bayes_risk_from_pld(pld, prior):
    """
    Derive Bayes Risk for a given PLD object.

    Directly uses Eq. 2 in https://arxiv.org/abs/2406.08918
    """
    prior, is_scalar = _ensure_array(prior)

    bayes_risk = []
    for prior_val in prior:
        result = optimize.minimize_scalar(
            lambda x: prior_val * x + (1 - prior_val) * get_beta_from_pld(pld, alpha=x),
            bounds=(0, 1),
            method="bounded",
        )
        if not result.success:
            warnings.warn(f"Optimization failed for prior = {prior_val:.4f}")
            bayes_risk.append(np.nan)
        else:
            bayes_risk.append(result.fun)

    if is_scalar:
        return bayes_risk[0]
    else:
        return np.array(bayes_risk)
