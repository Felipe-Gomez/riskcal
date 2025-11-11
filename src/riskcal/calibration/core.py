"""Generic privacy parameter calibration algorithm.

This module provides a unified calibration interface that works with any
privacy mechanism, provided you can supply an evaluator function that computes
privacy metrics for a given parameter value.
"""

from typing import Protocol, Literal, Any
from dataclasses import dataclass
import numpy as np

from riskcal.utils import inverse_monotone_function


class PrivacyEvaluator(Protocol):
    """Protocol for functions that evaluate privacy metrics for given parameter value."""

    def __call__(self, parameter_value: float) -> "PrivacyMetrics":
        """
        Evaluate privacy metrics for given parameter value.

        Args:
            parameter_value: Value of the parameter being calibrated
                (e.g., noise_multiplier, epsilon, sample_rate)

        Returns:
            PrivacyMetrics containing computed metrics
        """
        ...


@dataclass
class PrivacyMetrics:
    """Privacy metrics computed for a given parameter value."""

    # f-DP metrics
    advantage: float | None = None
    alpha: float | None = None  # FPR (if evaluating specific alpha)
    beta: float | None = None  # FNR (corresponding to alpha)

    # (ε,δ)-DP metrics
    epsilon: float | None = None
    delta: float | None = None

    # Additional metadata
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CalibrationTarget:
    """Specification of target privacy level to calibrate to."""

    kind: Literal["advantage", "err_rates", "epsilon_delta"]

    # For advantage target
    advantage: float | None = None

    # For err_rates target
    alpha: float | None = None  # Target FPR
    beta: float | None = None  # Target FNR

    # For epsilon_delta target
    epsilon: float | None = None
    delta: float | None = None


@dataclass
class CalibrationConfig:
    """Configuration for calibration search."""

    increasing: bool = True
    param_min: float = 0.0  # Lower bound for parameter search
    param_max: float = 100.0  # Upper bound for parameter search
    target_tol: float = 1e-3  # Convergence tolerance for target metric
    param_tol: float = 1e-3  # Convergence tolerance for parameter
    max_iterations: int = 100


@dataclass
class CalibrationResult:
    """Result of parameter calibration."""

    parameter_value: float  # Calibrated parameter value
    parameter_name: str = "noise_multiplier"  # Name of calibrated parameter

    # Achieved metrics
    achieved_advantage: float | None = None
    achieved_alpha: float | None = None
    achieved_beta: float | None = None
    achieved_epsilon: float | None = None
    achieved_delta: float | None = None

    # Convergence info
    converged: bool = True
    iterations: int = 0
    method: str = "generic"

    # Additional info
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    # Backward compatibility property
    @property
    def noise_multiplier(self) -> float:
        """Backward compatibility: return parameter_value as noise_multiplier."""
        return self.parameter_value


def calibrate_parameter(
    evaluator: PrivacyEvaluator,
    target: CalibrationTarget,
    config: CalibrationConfig | None = None,
    parameter_name: str = "noise_multiplier",
) -> CalibrationResult:
    """
    Generic privacy parameter calibration algorithm.

    Finds the parameter value such that the privacy guarantee (as measured
    by the evaluator) meets or exceeds the specified target.

    This is the core calibration algorithm that works with any privacy mechanism,
    provided you can supply an evaluator function that computes privacy metrics
    for a given parameter value.

    Args:
        evaluator: Callable that maps parameter_value → PrivacyMetrics.
            Should compute the relevant privacy metrics (advantage, beta, epsilon)
            for a given parameter value.
        target: Target privacy level to calibrate to. Specifies what metric
            to optimize (advantage, err_rates, or epsilon_delta) and the target value.
        config: Calibration configuration (search bounds, tolerances, etc.).
            If None, uses default CalibrationConfig().
        parameter_name: Name of the parameter being calibrated (for documentation).
            Default: "noise_multiplier"

    Returns:
        CalibrationResult containing:
            - parameter_value: Calibrated parameter value
            - parameter_name: Name of the calibrated parameter
            - achieved_*: Actually achieved privacy metrics at this parameter value
            - converged: Whether calibration converged
            - iterations: Number of search iterations

    Raises:
        ValueError: If target specification is invalid or inconsistent
        RuntimeError: If calibration fails to converge

    Example:
        >>> # Define evaluator for DP-SGD
        >>> def evaluate_dpsgd(noise_mult):
        ...     pld = create_dpsgd_pld(noise_mult, sample_rate=0.002, num_steps=10000)
        ...     advantage = get_advantage_from_pld(pld)
        ...     return PrivacyMetrics(advantage=advantage)
        >>>
        >>> # Calibrate to advantage target
        >>> target = CalibrationTarget(kind='advantage', advantage=0.1)
        >>> result = calibrate_parameter(evaluate_dpsgd, target)
        >>> print(f"Noise: {result.parameter_value:.3f}")

    Notes:
        - Uses binary search for monotonic objectives (advantage, epsilon, beta)
        - Assumes higher parameter values → stronger privacy (lower advantage/epsilon/beta)
          For parameters where lower is better, adjust bounds accordingly
    """
    config = config or CalibrationConfig()

    # Validate target
    _validate_target(target)

    # Dispatch based on target type
    if target.kind == "advantage":
        return _calibrate_to_advantage(evaluator, target, config, parameter_name)
    elif target.kind == "err_rates":
        return _calibrate_to_err_rates(evaluator, target, config, parameter_name)
    elif target.kind == "epsilon_delta":
        return _calibrate_to_epsilon_delta(evaluator, target, config, parameter_name)
    else:
        raise ValueError(f"Unknown target kind: {target.kind}")


def _calibrate_to_advantage(
    evaluator: PrivacyEvaluator,
    target: CalibrationTarget,
    config: CalibrationConfig,
    parameter_name: str,
) -> CalibrationResult:
    """Calibrate to target advantage using binary search."""

    def objective(param_value: float) -> float:
        """Compute advantage for given parameter value."""
        metrics = evaluator(param_value)
        if metrics.advantage is None:
            raise ValueError("Evaluator must return advantage for advantage target")
        return metrics.advantage

    # Binary search: higher parameter → lower advantage (decreasing function)
    parameter_value = inverse_monotone_function(
        f=objective,
        f_target=target.advantage,
        bounds=(config.param_min, config.param_max),
        func_threshold=config.target_tol,
        max_iter=config.max_iterations,
        increasing=config.increasing,
    )

    # Evaluate final metrics
    final_metrics = evaluator(parameter_value)

    return CalibrationResult(
        parameter_value=parameter_value,
        parameter_name=parameter_name,
        achieved_advantage=final_metrics.advantage,
        achieved_epsilon=final_metrics.epsilon,
        achieved_delta=final_metrics.delta,
        converged=True,
        method="binary_search",
    )


def _calibrate_to_err_rates(
    evaluator: PrivacyEvaluator,
    target: CalibrationTarget,
    config: CalibrationConfig,
    parameter_name: str,
) -> CalibrationResult:
    """Calibrate to target (alpha, beta) using binary search on beta."""

    def objective(param_value: float) -> float:
        """Compute beta at target alpha for given parameter value."""
        metrics = evaluator(param_value)
        if metrics.beta is None:
            raise ValueError("Evaluator must return beta for err_rates target")
        return metrics.beta

    # Binary search: higher parameter → lower beta (decreasing function)
    parameter_value = inverse_monotone_function(
        f=objective,
        f_target=target.beta,
        bounds=(config.param_min, config.param_max),
        func_threshold=config.target_tol,
        max_iter=config.max_iterations,
        increasing=config.increasing,
    )

    # Evaluate final metrics
    final_metrics = evaluator(parameter_value)

    return CalibrationResult(
        parameter_value=parameter_value,
        parameter_name=parameter_name,
        achieved_alpha=target.alpha,
        achieved_beta=final_metrics.beta,
        achieved_advantage=final_metrics.advantage,
        converged=True,
        method="binary_search",
    )


def _calibrate_to_epsilon_delta(
    evaluator: PrivacyEvaluator,
    target: CalibrationTarget,
    config: CalibrationConfig,
    parameter_name: str,
) -> CalibrationResult:
    """Calibrate to target (epsilon, delta) using binary search on epsilon."""

    def objective(param_value: float) -> float:
        """Compute epsilon at target delta for given parameter value."""
        metrics = evaluator(param_value)
        if metrics.epsilon is None:
            raise ValueError("Evaluator must return epsilon for epsilon_delta target")
        return metrics.epsilon

    # Binary search: higher parameter → lower epsilon (decreasing function)
    parameter_value = inverse_monotone_function(
        f=objective,
        f_target=target.epsilon,
        bounds=(config.param_min, config.param_max),
        func_threshold=config.target_tol,
        max_iter=config.max_iterations,
        increasing=config.increasing,
    )

    # Evaluate final metrics
    final_metrics = evaluator(parameter_value)

    return CalibrationResult(
        parameter_value=parameter_value,
        parameter_name=parameter_name,
        achieved_epsilon=final_metrics.epsilon,
        achieved_delta=target.delta,
        converged=True,
        method="binary_search",
    )


def _validate_target(target: CalibrationTarget) -> None:
    """Validate target specification."""
    if target.kind == "advantage":
        if target.advantage is None:
            raise ValueError("advantage target requires advantage value")
        if not 0 <= target.advantage <= 1:
            raise ValueError(f"advantage must be in [0,1], got {target.advantage}")

    elif target.kind == "err_rates":
        if target.alpha is None or target.beta is None:
            raise ValueError("err_rates target requires both alpha and beta")
        if not 0 <= target.alpha <= 1:
            raise ValueError(f"alpha must be in [0,1], got {target.alpha}")
        if not 0 <= target.beta <= 1:
            raise ValueError(f"beta must be in [0,1], got {target.beta}")
        if target.alpha + target.beta >= 1:
            raise ValueError(
                f"Invalid trade-off: alpha + beta = {target.alpha + target.beta} >= 1"
            )

    elif target.kind == "epsilon_delta":
        if target.epsilon is None or target.delta is None:
            raise ValueError("epsilon_delta target requires both epsilon and delta")
        if target.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {target.epsilon}")
        if not 0 <= target.delta <= 1:
            raise ValueError(f"delta must be in [0,1], got {target.delta}")
