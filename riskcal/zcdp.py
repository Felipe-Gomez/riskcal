from typing import Optional, Union
import warnings
import numpy as np
from scipy.optimize import minimize

from . import utils


def get_beta(
    rho: float,
    alpha: Union[np.ndarray, float],
    alpha_grid_step: float = 1e-2,
    beta_grid_step: float = 1e-4,
    orders: Optional[np.ndarray] = None,
    verbose: bool = False,
):
    """
    Vectorized implementation of the trade-off curve calculation from rho-zCDP.

    Based on Section F.1 in Zhu et al. https://arxiv.org/pdf/2106.08567
    """
    MAX_ORDER = 128
    MIN_ORDER = 0.5
    NUM_ORDERS = 128

    if isinstance(orders, int):
        orders = np.linspace(MIN_ORDER, MAX_ORDER, orders)
    elif orders is None:
        orders = np.linspace(MIN_ORDER, MAX_ORDER, NUM_ORDERS)

    alpha_prime = np.linspace(
        alpha_grid_step,
        1.0 - alpha_grid_step,
        int(np.ceil((1.0 - 2 * alpha_grid_step) / alpha_grid_step)),
    )

    # At this point, the notation switches to the notation of Zhu et al. for
    # consistency with their derivations.
    # alpha -> x
    # beta -> y
    # order -> alpha

    # Create the grids
    orders_column = orders[:, np.newaxis, np.newaxis]  # Shape: (n_orders, 1, 1)
    x_prime_row = alpha_prime[np.newaxis, :, np.newaxis]  # Shape: (1, n_alpha, 1)
    log_y_grid = np.log(np.linspace(0, 1, int(np.ceil(1 / beta_grid_step))))
    log_y_grid = log_y_grid[np.newaxis, np.newaxis, :]  # Shape: (1, 1, n_beta)

    # Precompute common terms
    logx = np.log(x_prime_row)
    log1mx = np.log1p(-np.exp(logx))
    epsilon = rho * orders_column

    # Handle alpha != 1 and alpha == 1 cases separately but still vectorized
    alpha_mask = orders_column != 1

    # Initialize arrays for results
    function1_values = np.zeros_like(orders_column * x_prime_row * log_y_grid)
    function2_values = np.zeros_like(function1_values)

    # Case where alpha != 1
    if np.any(alpha_mask):
        upper = (orders_column - 1) * epsilon
        sign_alpha = np.sign(orders_column - 1)

        # Precompute terms
        one_minus_alpha = 1 - orders_column
        log1my = np.log1p(-np.exp(log_y_grid))

        # Compute function1
        term1_f1 = orders_column * log1my + one_minus_alpha * logx
        term2_f1 = orders_column * log_y_grid + one_minus_alpha * log1mx
        function1_values = np.where(
            alpha_mask, np.logaddexp(term1_f1, term2_f1), function1_values
        )

        # Compute function2
        term1_f2 = orders_column * logx + one_minus_alpha * log1my
        term2_f2 = orders_column * log1mx + one_minus_alpha * log_y_grid
        function2_values = np.where(
            alpha_mask, np.logaddexp(term1_f2, term2_f2), function2_values
        )

    # Case where alpha == 1 (KL divergence)
    if np.any(~alpha_mask):
        upper = epsilon
        x_val = np.exp(logx)
        one_minus_x = np.exp(log1mx)

        log1my = np.log1p(-np.exp(log_y_grid))
        y_vals = np.exp(log_y_grid)
        one_minus_y = np.exp(log1my)

        # Compute function1
        term1_f1 = x_val * (logx - log1my)
        term2_f1 = one_minus_x * (log1mx - log_y_grid)
        function1_values = np.where(~alpha_mask, term1_f1 + term2_f1, function1_values)

        # Compute function2
        term1_f2 = y_vals * (log_y_grid - log1mx)
        term2_f2 = one_minus_y * (log1my - logx)
        function2_values = np.where(~alpha_mask, term1_f2 + term2_f2, function2_values)

    # Check constraints
    sign_alpha = np.where(orders_column != 1, np.sign(orders_column - 1), 1)
    upper = np.where(orders_column != 1, (orders_column - 1) * epsilon, epsilon)

    constraint1_satisfied = sign_alpha * (upper - function1_values) >= 0
    constraint2_satisfied = sign_alpha * (upper - function2_values) >= 0
    valid_points = np.logical_and(constraint1_satisfied, constraint2_satisfied)

    # Switching back to the package notation.
    # alpha -> order
    # x -> alpha
    # y -> beta
    log_y_values = np.full((len(orders), len(alpha_prime)), np.inf)
    for i in range(len(orders)):
        for j in range(len(alpha_prime)):
            valid_indices = valid_points[i, j]
            if np.any(valid_indices):
                log_y_values[i, j] = np.min(log_y_grid[0, 0, valid_indices])

    beta_mat = np.exp(log_y_values)
    beta = np.max(beta_mat, axis=0)

    return 1 - np.interp(
        utils.ensure_array(alpha), alpha_prime, 1 - beta, left=0.0, right=1.0
    )
