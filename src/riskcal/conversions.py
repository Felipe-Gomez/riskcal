import warnings
from typing import Union

import warnings
import numpy as np
from scipy import stats
from scipy import optimize

from dp_accounting.pld import privacy_loss_distribution

from riskcal import plrv


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


def get_advantage_from_pld(
    pld: privacy_loss_distribution.PrivacyLossDistribution,
) -> float:
    """
    Get advantage for a given PLD object.
    """
    return pld.get_delta_for_epsilon(0)  # type: ignore


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
    Get the trade-off curve for Gaussian Differential Privacy.

    Eq. 6 in https://arxiv.org/abs/1905.02383

    along with identity

    Phi^{-1}(1-alpha) = - Phi^{-1}(alpha)
    """
    return stats.norm.cdf(-stats.norm.ppf(alpha) - mu)


def get_beta_for_epsilon_delta(
    epsilon: float, delta: float, alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Get the trade-off curve for given epsilon, delta.

    Eq. 5 in https://arxiv.org/abs/1905.02383

    >>> np.round(get_beta_for_epsilon_delta(1.0, 0.001, 0.8), 3)
    0.073
    """
    form1 = np.array(1 - delta - np.exp(epsilon) * alpha)
    form2 = np.array(np.exp(-epsilon) * (1 - delta - alpha))
    return np.maximum.reduce([form1, form2, np.zeros_like(form1)])


def get_advantage_for_epsilon_delta(epsilon: float, delta: float) -> float:
    """Derive advantage for a given epsilon and delta.

    >>> np.round(get_advantage_for_epsilon_delta(0., 0.001), 3)
    0.001
    """
    return (np.exp(epsilon) + 2 * delta - 1) / (np.exp(epsilon) + 1)


def get_epsilon_for_err_rates(delta: float, alpha: float, beta: float) -> float:
    """Derive epsilon for a given FPR/FNR error rates and delta.

    >>> np.round(get_epsilon_for_err_rates(0.001, 0.001, 0.8), 3)
    5.293
    """
    epsilon1 = np.log((1 - delta - alpha) / beta)
    epsilon2 = np.log((1 - delta - beta) / alpha)
    return np.maximum.reduce([epsilon1, epsilon2, np.zeros_like(epsilon1)])


def get_advantage_for_rho(
    rho: float,
    max_bisection_steps: int = 100,
    max_order: float = 2,
    linear_search_step: float = 0.5 * 1e-3,
    opt_method: str = "bounded",
    tol: float = 1e-5,
) -> float:
    """Derive advantage from a given zCDP rho."""
    result = optimize.minimize_scalar(
        lambda alpha: -(
            1
            - alpha
            - get_beta_for_rho(
                rho=rho,
                alpha=alpha,
                max_bisection_steps=max_bisection_steps,
                max_order=max_order,
                linear_search_step=linear_search_step,
                opt_method=opt_method,
                tol=tol,
            )
        ),
        bounds=(0, 1),
        method=opt_method,
        tol=tol,
    )
    if not result.success:
        return np.nan
    else:
        return -result.fun


def _check_renyi_constraints(epsilon: float, y: float, x: float, order: float) -> bool:
    # In the paper's notation:
    # alpha (from input) -> x (Type I error)
    # beta (the return value) -> y (Type II error)
    # order -> alpha (Rényi divergence order)

    # Precompute terms in log space
    logx = np.log(x)
    log1mx = np.log1p(-x)
    logy = np.log(y)
    log1my = np.log1p(-y)

    # Case where order != 1
    if order != 1:
        upper = (order - 1) * epsilon
        sign_order = np.sign(order - 1)
        one_minus_order = 1 - order

        log_f2 = np.logaddexp(
            order * logx + one_minus_order * log1my,
            order * log1mx + one_minus_order * logy,
        )
        constraint2 = sign_order * (upper - log_f2) >= 0

        log_f1 = np.logaddexp(
            order * log1my + one_minus_order * logx,
            order * logy + one_minus_order * log1mx,
        )
        constraint1 = sign_order * (upper - log_f1) >= 0

    # Case where order == 1
    else:
        upper = epsilon
        f1 = x * (logx - log1my) + (1 - x) * (log1mx - logy)
        constraint1 = upper - f1 >= 0
        f2 = y * (logy - log1mx) + (1 - y) * (log1my - logx)
        constraint2 = upper - f2 >= 0

    return constraint1 and constraint2


def _get_beta_for_alpha_and_order(
    rho,
    alpha,
    order,
    linear_search_step=1e-3,
    max_bisection_steps=100,
    tol=1e-5,
):
    LOW_RHO_REGIME = 0.01
    HIGH_RHO_REGIME = 50
    SPECIAL_REGIME_BRACKET = 0.1

    epsilon = rho * order

    # In special regimes, set the bracket for bisection with heuristics.
    if rho <= LOW_RHO_REGIME:
        beta_low = max(0, 1 - alpha - SPECIAL_REGIME_BRACKET)
        beta_high = 1 - alpha

    elif rho >= HIGH_RHO_REGIME:
        if alpha <= tol:
            beta_low = 1 - SPECIAL_REGIME_BRACKET
            beta_high = 1
        else:
            beta_low = 0
            beta_high = SPECIAL_REGIME_BRACKET

    # Otherwise, run linear search to get the bracket for bisection.
    else:
        beta1 = 0
        while not _check_renyi_constraints(epsilon, beta1, alpha, order) or beta1 >= 1:
            beta1 += linear_search_step
        beta1 = min(beta1, 1)

        beta2 = 1
        while not _check_renyi_constraints(epsilon, beta2, alpha, order) or beta2 <= 0:
            beta2 -= linear_search_step
        beta2 = max(beta2, 0)

        beta_low = min(beta1, beta2)
        beta_high = max(beta1, beta2)

    # Bisection.
    for _ in range(max_bisection_steps):
        if np.abs(beta_high - beta_low) <= tol:
            break
        beta_mid = (beta_low + beta_high) / 2
        if _check_renyi_constraints(epsilon, beta_mid, alpha, order):
            beta_high = beta_mid
        else:
            beta_low = beta_mid

    any_feasible_solution = any(
        [
            _check_renyi_constraints(epsilon, beta_high, alpha, order),
            _check_renyi_constraints(epsilon, beta_low, alpha, order),
        ]
    )
    if not any_feasible_solution:
        # No feasible solution exists
        return 0.0
    else:
        # Return a conservative solution.
        return beta_low


def get_beta_for_rho(
    rho: float,
    alpha: Union[np.ndarray, float],
    max_bisection_steps: int = 100,
    max_order: float = 2,
    linear_search_step: float = 1e-3,
    opt_method: str = "bounded",
    tol: float = 1e-5,
):
    """Get the trade-off curve for given zCDP rho.

    This implementation uses binary search to find the optimal beta for each alpha,
    directly implementing the trade-off from Section F.1 in Zhu et al.,
    https://arxiv.org/pdf/2106.085677
    """

    print(f"{rho=}")

    beta = []
    for alpha_val in np.atleast_1d(alpha):
        if alpha_val <= tol:
            beta.append(1.0)
        elif alpha_val >= 1 - tol:
            beta.append(0.0)
        else:
            solution = optimize.minimize_scalar(
                lambda order: -_get_beta_for_alpha_and_order(
                    rho=rho,
                    alpha=alpha_val,
                    order=order,
                    linear_search_step=linear_search_step,
                    max_bisection_steps=max_bisection_steps,
                    tol=tol,
                ),
                bounds=(0.5, max_order),
                method=opt_method,
                tol=tol,
            )
            if solution.success:
                beta.append(-solution.fun)
            else:
                beta.append(np.nan)

    return beta[0] if isinstance(alpha, (int, float)) else np.array(beta)
