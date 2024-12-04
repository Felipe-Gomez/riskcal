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


def _get_log_y_from_rho_grid(x, alpha, rho, y_grid_step=1e-4):
    """
    Vectorized grid search implementation for finding optimal log_y value.
    Uses a discrete grid search instead of continuous optimization.

    Args:
        x (float): FPR value (between 0 and 1)
        alpha (float): Renyi DP order
        rho (float): Renyi DP parameter (default=1)
        grid_points (int): Number of points in the grid search (default=1000)
        min_log_y (float): Minimum value for log_y search space (default=-50)
        max_log_y (float): Maximum value for log_y search space (default=0)

    Returns:
        float: Optimal log_y value that minimizes y while satisfying constraints
    """
    # Create grid of log_y values to search over
    log_y_grid = np.log(
        np.linspace(y_grid_step, 1, int(np.ceil((1.0 - 2 * y_grid_step) / y_grid_step)))
    )

    # Precompute common terms
    logx = np.log(x)
    log1mx = np.log1p(-np.exp(logx))
    epsilon = rho * alpha

    if alpha != 1:
        upper = (alpha - 1) * epsilon
        sign_alpha = np.sign(alpha - 1)

        # Precompute terms used in both functions
        one_minus_alpha_times_log1mx = (1 - alpha) * log1mx
        one_minus_alpha_times_logx = (1 - alpha) * logx
        alpha_times_logx = alpha * logx
        alpha_times_log1mx = alpha * log1mx

        # Vectorized computation for all log_y values
        log1my = np.log1p(-np.exp(log_y_grid))

        # Compute function1 for all points
        term1_f1 = alpha * log1my + one_minus_alpha_times_logx
        term2_f1 = alpha * log_y_grid + one_minus_alpha_times_log1mx
        function1_values = np.logaddexp(term1_f1, term2_f1)

        # Compute function2 for all points
        term1_f2 = alpha_times_logx + (1 - alpha) * log1my
        term2_f2 = alpha * log1mx + (1 - alpha) * log_y_grid
        function2_values = np.logaddexp(term1_f2, term2_f2)

    else:
        upper = epsilon
        sign_alpha = 1
        x_val = np.exp(logx)
        one_minus_x = np.exp(log1mx)

        # Vectorized computation for all log_y values
        log1my = np.log1p(-np.exp(log_y_grid))
        y_vals = np.exp(log_y_grid)
        one_minus_y = np.exp(log1my)

        # Compute function1 for all points
        term1_f1 = x_val * (logx - log1my)
        term2_f1 = one_minus_x * (log1mx - log_y_grid)
        function1_values = term1_f1 + term2_f1

        # Compute function2 for all points
        term1_f2 = y_vals * (log_y_grid - log1mx)
        term2_f2 = one_minus_y * (log1my - logx)
        function2_values = term1_f2 + term2_f2

    # Check constraints for all points
    constraint1_satisfied = sign_alpha * (upper - function1_values) >= 0
    constraint2_satisfied = sign_alpha * (upper - function2_values) >= 0

    # Find points that satisfy both constraints
    valid_points = np.logical_and(constraint1_satisfied, constraint2_satisfied)

    # If no valid points found, return None or raise exception
    if not np.any(valid_points):
        raise ValueError("No valid solutions found in the grid search range")

    # Among valid points, find the one that minimizes log_y
    valid_log_y_values = log_y_grid[valid_points]
    optimal_idx = np.argmin(valid_log_y_values)

    return valid_log_y_values[optimal_idx]


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
    MAX_ORDER = 128
    MIN_ORDER = 0.5
    NUM_ORDERS = 64

    if isinstance(orders, int):
        orders = np.linspace(MIN_ORDER, MAX_ORDER, orders)
    elif orders is None:
        orders = np.linspace(MIN_ORDER, MAX_ORDER, NUM_ORDERS)

    alpha_prime = np.linspace(
        alpha_grid_step,
        1.0 - alpha_grid_step,
        int(np.ceil((1.0 - 2 * alpha_grid_step) / alpha_grid_step)),
    )

    orders_column = np.asarray(orders)[:, np.newaxis]  # Shape: (n_orders, 1)
    alpha_prime_row = alpha_prime[np.newaxis, :]  # Shape: (1, n_alpha)

    log_y_values = np.vectorize(_get_log_y_from_rho_grid)(
        x=alpha_prime_row,  # Will broadcast across orders
        alpha=orders_column,  # Will broadcast across alpha_prime
        rho=rho,  # Scalar value
    )

    # Convert to beta values
    beta_mat = np.exp(log_y_values)

    beta = np.max(beta_mat, axis=0)
    assert len(beta) == len(alpha_prime)

    return 1 - np.interp(
        utils.ensure_array(alpha), alpha_prime, 1 - beta, left=0.0, right=1.0
    )
