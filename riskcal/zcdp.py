from typing import Optional, Union
import warnings
import numpy as np
from scipy.optimize import minimize

from . import utils


def _log_add(logx: float, logy: float) -> float:
    """
    Adds two numbers in the log space.
    Input is log(x) and log(y), output is log(x + y)
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return np.log1p(np.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
    """Subtracts two numbers in the log space. Answer must be non-negative."""
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return np.log(np.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _get_log_y_from_rho(x, alpha, rho=1):
    """
    Based off of discussion following equation 13 of Zhu et al.:
    https://arxiv.org/abs/2106.08567

    The notation in this function uses the notation from the paper. This
    function outputs y (FNR) for a given x (FPR), alpha (Renyi DP order,
    unlike the notation in the rest of the code), and rho (Renyi DP parameter).

    It computes epsilon_RDP = alpha * rho, then converts the
    RDP guarantee to a trade-off curve guarantee.

    It solves the optimization problem:

    minimize y

    subject to:
    function1(y) <= upper = exp( (alpha - 1) epsilon_RDP)
    function2(y) <= upper = exp( (alpha - 1) epsilon_RDP)

    It does this in log space to ensure numerical stability.
    """

    logx = np.log(x)
    log1mx = np.log1p(-np.exp(logx))
    epsilon = rho * alpha

    if alpha != 1:
        upper = (alpha - 1) * epsilon
        sign_alpha = np.sign(alpha - 1)

        # Do this outside of function for speed
        one_minus_alpha_times_log1mx = (1 - alpha) * log1mx
        one_minus_alpha_times_logx = (1 - alpha) * logx
        alpha_times_logx = alpha * logx
        alpha_times_log1mx = alpha * log1mx

        def function1(logy):
            log1my = np.log1p(-np.exp(logy))
            term1 = alpha * log1my + one_minus_alpha_times_logx
            term2 = alpha * logy + one_minus_alpha_times_log1mx
            return _log_add(term1, term2)

        def function2(logy):
            log1my = np.log1p(-np.exp(logy))
            term1 = alpha_times_logx + (1 - alpha) * log1my
            term2 = alpha * log1mx + (1 - alpha) * logy
            return _log_add(term1, term2)

    else:
        upper = epsilon
        sign_alpha = 1
        x = np.exp(logx)
        one_minus_x = np.exp(log1mx)

        def function1(logy):
            log1my = np.log1p(-np.exp(logy))
            term1 = x * (logx - log1my)
            term2 = one_minus_x * (log1mx - logy)
            return term1 + term2

        def function2(logy):
            y = np.exp(logy)
            log1my = np.log1p(-np.exp(logy))
            one_minus_y = np.exp(log1my)

            term1 = y * (logy - log1mx)
            term2 = one_minus_y * (log1my - logx)
            return term1 + term2

    # Define the objective function (minimize log y)
    def objective(logy):
        return logy

    # Define the first constraint (f(y) <= epsilon)
    def constraint_f(logy):
        return sign_alpha * (upper - function1(logy[0]))

    # Define the second constraint (g(y) <= epsilon)
    def constraint_g(logy):
        return sign_alpha * (upper - function2(logy[0]))

    # Combine constraints into a list
    constraints = [
        {"type": "ineq", "fun": constraint_f},
        {"type": "ineq", "fun": constraint_g},
    ]

    # Set the bounds for log y
    bounds = [(-300, -1e-15)]

    # Initial guess for y
    initial_guess = [np.log(0.5)]

    # Perform the optimization
    result = minimize(
        objective, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints
    )

    # Check and print the results
    if result.success:
        return result.x[0]
    else:
        warnings.warn(
            f"Optimization failed for x = {x}, alpha = {alpha}: {result.message}"
        )
        return np.nan


def get_beta(
    rho: float,
    alpha: Union[np.ndarray, float],
    alpha_grid_step: float = 1e-4,
    orders: Optional[np.ndarray] = None,
    verbose: bool = False,
):
    """
    Get the trade-off curve from rho-zCDP.
    """
    MAX_ORDER = 64
    MIN_ORDER = 0.5
    NUM_ORDERS = 20

    if isinstance(orders, int):
        orders = np.linspace(MIN_ORDER, MAX_ORDER, orders)
    elif orders is None:
        orders = np.linspace(MIN_ORDER, MAX_ORDER, NUM_ORDERS)

    alpha_prime = np.linspace(
        alpha_grid_step,
        1.0 - alpha_grid_step,
        int(np.ceil((1.0 - 2 * alpha_grid_step) / alpha_grid_step)),
    )

    beta_mat = np.zeros((len(orders), len(alpha_prime)))
    it = enumerate(orders)
    if verbose:
        from tqdm import auto as tqdm

        it = tqdm.tqdm(list(it))

    for i, order in it:
        for j, alpha_val in enumerate(alpha_prime):
            beta_mat[i, j] = np.exp(
                _get_log_y_from_rho(rho=rho, x=alpha_val, alpha=order)
            )

    beta = np.max(beta_mat, axis=0)
    assert len(beta) == len(alpha_prime)

    return 1 - np.interp(
        utils.ensure_array(alpha), alpha_prime, 1 - beta, left=0.0, right=1.0
    )
