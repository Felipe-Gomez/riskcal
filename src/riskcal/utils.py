"""Utility functions for riskcal."""

from typing import Callable, Tuple
import numpy as np


def _ensure_array(x):
    """
    Convert input to numpy array and track if it was originally a scalar.

    Args:
        x: Input value (scalar or array-like)

    Returns:
        Tuple of (np.ndarray, bool) where bool indicates if input was scalar
    """
    is_scalar = isinstance(x, (int, float))
    if is_scalar:
        return np.asarray([x]), is_scalar
    return np.asarray(x), False


def inverse_monotone_function(
    f: Callable[[float], float],
    f_target: float,
    bounds: Tuple[float, float],
    func_threshold: float = 1e-3,
    param_threshold: float = 1e-3,
    max_iter: int = 100,
    increasing: bool = False,
) -> float:
    """
    Find the value of x such that the monotonic function f(x) is approximately
    equal to f_target within a given threshold using binary search.

    Args:
        f: A monotonic function (increasing or decreasing) to invert.
        f_target: The target value for f(x).
        bounds: A tuple (lower_x, upper_x) defining the search interval.
        func_threshold: Acceptable error for |f(x) - f_target|.
        increasing: Indicates if f is increasing (True) or decreasing (False).

    Returns:
        The value of x that satisfies the threshold conditions.

    Notes:
        It is guaranteed that the returned x is within the thresholds of the
        smallest (for monotonically decreasing func) or the largest (for
        monotonically increasing func) such x.
    """

    # Initialize bounds and midpoint
    lower_x, upper_x = bounds
    mid_x = (upper_x + lower_x) / 2
    f_mid = f(mid_x)

    # setup check function
    if increasing:
        check = lambda f_value, target_value: f_value <= target_value

        def continue_condition(upper_x, lower_x):
            return (abs(f(lower_x) - f_target) > func_threshold) or (
                abs(lower_x - upper_x) > param_threshold
            )

    else:
        check = lambda f_value, target_value: f_value > target_value

        def continue_condition(upper_x, lower_x):
            return (abs(f(upper_x) - f_target) > func_threshold) or (
                abs(lower_x - upper_x) > param_threshold
            )

    # run bisection
    for i in range(max_iter):

        if not continue_condition(upper_x, lower_x):
            break

        mid_x = (upper_x + lower_x) / 2
        f_mid = f(mid_x)
        if check(f_mid, f_target):
            lower_x = mid_x
        else:
            upper_x = mid_x

    if continue_condition(upper_x, lower_x):
        raise RuntimeError(
            f"Could not achieve the desired tolerance. Consider increasing "
            f"tolerance or increasing the parameter range. "
            f"{lower_x=} {upper_x=} / "
            f"param_diff={abs(upper_x - lower_x)} / target_diff={f(mid_x) - f_target}"
        )

    if increasing:
        return lower_x
    else:
        return upper_x
