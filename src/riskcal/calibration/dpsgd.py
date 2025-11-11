"""DP-SGD specific calibration using direct PLD composition (fast path)."""

from typing import Union
import warnings
import numpy as np
from dp_accounting.pld import privacy_loss_distribution as pld_module

from riskcal.calibration.core import (
    PrivacyEvaluator,
    PrivacyMetrics,
    CalibrationTarget,
    CalibrationConfig,
    CalibrationResult,
    calibrate_parameter,
)


def create_dpsgd_evaluator(
    sample_rate: float,
    num_steps: int,
    grid_step: float = 1e-4,
    target_alpha: float | None = None,
) -> PrivacyEvaluator:
    """
    Create a privacy evaluator for DP-SGD.

    Returns a function that maps noise_multiplier → PrivacyMetrics for DP-SGD
    with the specified parameters.

    Args:
        sample_rate: Poisson sampling rate (typically batch_size / dataset_size).
        num_steps: Number of DP-SGD steps (typically num_epochs * steps_per_epoch).
        grid_step: Discretization interval for PLD computation.
        target_alpha: If provided, evaluator will compute beta at this alpha value.
            Required for err_rates calibration.

    Returns:
        PrivacyEvaluator function that computes metrics for given noise_multiplier.

    Example:
        >>> evaluator = create_dpsgd_evaluator(sample_rate=0.002, num_steps=10000)
        >>> metrics = evaluator(noise_multiplier=1.0)
        >>> print(f"Advantage: {metrics.advantage:.4f}")
    """
    from riskcal.analysis import get_advantage_from_pld, get_beta_from_pld

    def evaluator(noise_multiplier: float) -> PrivacyMetrics:
        """Evaluate DP-SGD privacy metrics for given noise."""
        # Create PLD for single step
        pld = pld_module.from_gaussian_mechanism(
            standard_deviation=noise_multiplier,
            sensitivity=1.0,
            value_discretization_interval=grid_step,
        )

        # Compose for num_steps with Poisson sampling
        pld = pld_module.from_gaussian_mechanism(
            standard_deviation=noise_multiplier,
            sampling_prob=sample_rate,
            use_connect_dots=True,
            value_discretization_interval=grid_step,
        ).self_compose(num_steps)

        # Compute metrics
        advantage = get_advantage_from_pld(pld)

        metrics = PrivacyMetrics(advantage=advantage)

        # Compute beta if target_alpha specified (for err_rates calibration)
        if target_alpha is not None:
            beta = get_beta_from_pld(pld, alpha=target_alpha)
            metrics.alpha = target_alpha
            metrics.beta = beta

        return metrics

    return evaluator


def create_dpsgd_epsilon_evaluator(
    sample_rate: float,
    num_steps: int,
    target_delta: float,
    grid_step: float = 1e-4,
) -> PrivacyEvaluator:
    """
    Create an epsilon-delta evaluator for DP-SGD.

    Returns a function that maps noise_multiplier → PrivacyMetrics with epsilon
    for the specified delta.

    Args:
        sample_rate: Poisson sampling rate.
        num_steps: Number of DP-SGD steps.
        target_delta: Delta parameter for epsilon computation.
        grid_step: Discretization interval for PLD computation.

    Returns:
        PrivacyEvaluator that computes epsilon at target_delta.
    """

    def evaluator(noise_multiplier: float) -> PrivacyMetrics:
        """Evaluate epsilon-delta for given noise."""
        pld = pld_module.from_gaussian_mechanism(
            standard_deviation=noise_multiplier,
            sampling_prob=sample_rate,
            use_connect_dots=True,
            value_discretization_interval=grid_step,
        ).self_compose(num_steps)

        epsilon = pld.get_epsilon_for_delta(target_delta)

        return PrivacyMetrics(epsilon=epsilon, delta=target_delta)

    return evaluator


def find_noise_multiplier_for_advantage(
    advantage: float,
    sample_rate: float,
    num_steps: int,
    grid_step: float = 1e-4,
    advantage_tol: float = 1e-3,
    noise_min: float = 0.1,
    noise_max: float = 50.0,
    # Deprecated parameter names for backward compatibility
    advantage_error: float = None,
    mu_error: float = None,
    mu_min: float = None,
    mu_max: float = None,
) -> float:
    """
    Calibrate DP-SGD noise to target advantage (legacy interface).

    .. deprecated:: 1.2.0
        Legacy parameter names (advantage_error, mu_error, mu_min, mu_max)
        will be removed in version 2.0.0.

    Finds minimum noise_multiplier such that advantage ≤ target.

    Args:
        advantage: Target advantage bound in [0, 1].
        sample_rate: Poisson sampling rate.
        num_steps: Number of DP-SGD steps.
        grid_step: PLD discretization interval.
        advantage_tol: Convergence tolerance for advantage.
        noise_min: Lower bound for search.
        noise_max: Upper bound for search.

    Returns:
        Calibrated noise_multiplier (float).

    Example:
        >>> noise = find_noise_multiplier_for_advantage(
        ...     advantage=0.1,
        ...     sample_rate=0.002,
        ...     num_steps=10000
        ... )
        >>> print(f"Use noise_multiplier: {noise:.3f}")

    Note:
        For new code, consider using the generic interface:
        >>> from riskcal.calibration import calibrate_parameter, CalibrationTarget
        >>> evaluator = create_dpsgd_evaluator(sample_rate=0.002, num_steps=10000)
        >>> target = CalibrationTarget(kind='advantage', advantage=0.1)
        >>> result = calibrate_parameter(evaluator, target)
    """
    # Handle backward compatibility for deprecated parameter names
    if advantage_error is not None:
        warnings.warn(
            "Parameter 'advantage_error' is deprecated and will be removed in v2.0.0. "
            "Use 'advantage_tol' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        advantage_tol = advantage_error
    elif mu_error is not None:
        warnings.warn(
            "Parameter 'mu_error' is deprecated and will be removed in v2.0.0. "
            "Use 'advantage_tol' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        advantage_tol = mu_error
    elif advantage_tol is None:
        advantage_tol = 1e-3

    if mu_min is not None:
        warnings.warn(
            "Parameter 'mu_min' is deprecated and will be removed in v2.0.0. "
            "Use 'noise_min' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        noise_min = mu_min

    if mu_max is not None:
        warnings.warn(
            "Parameter 'mu_max' is deprecated and will be removed in v2.0.0. "
            "Use 'noise_max' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        noise_max = mu_max

    evaluator = create_dpsgd_evaluator(
        sample_rate=sample_rate, num_steps=num_steps, grid_step=grid_step
    )

    target = CalibrationTarget(kind="advantage", advantage=advantage)

    config = CalibrationConfig(
        param_min=noise_min,
        param_max=noise_max,
        target_tol=advantage_tol,
        increasing=False,
    )

    result = calibrate_parameter(
        evaluator, target, config, parameter_name="noise_multiplier"
    )

    return result.parameter_value


def find_noise_multiplier_for_err_rates(
    alpha: float,
    beta: float,
    sample_rate: float,
    num_steps: int,
    grid_step: float = 1e-4,
    beta_tol: float = 1e-3,
    noise_min: float = 0.1,
    noise_max: float = 50.0,
    # Deprecated parameter names for backward compatibility
    beta_error: float = None,
    mu_error: float = None,
    mu_min: float = None,
    mu_max: float = None,
) -> float:
    """
    Calibrate DP-SGD noise to target error rates (legacy interface).

    .. deprecated:: 1.2.0
        Legacy parameter names (beta_error, mu_error, mu_min, mu_max)
        will be removed in version 2.0.0.

    Finds minimum noise_multiplier such that beta(alpha) ≤ target_beta.

    Args:
        alpha: Target false positive rate (FPR) in [0, 1].
        beta: Target false negative rate (FNR) in [0, 1].
        sample_rate: Poisson sampling rate.
        num_steps: Number of DP-SGD steps.
        grid_step: PLD discretization interval.
        beta_tol: Convergence tolerance for beta.
        noise_min: Lower bound for search.
        noise_max: Upper bound for search.

    Returns:
        Calibrated noise_multiplier (float).
    """
    # Handle backward compatibility for deprecated parameter names
    if beta_error is not None:
        warnings.warn(
            "Parameter 'beta_error' is deprecated and will be removed in v2.0.0. "
            "Use 'beta_tol' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        beta_tol = beta_error
    elif mu_error is not None:
        warnings.warn(
            "Parameter 'mu_error' is deprecated and will be removed in v2.0.0. "
            "Use 'beta_tol' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        beta_tol = mu_error
    elif beta_tol is None:
        beta_tol = 1e-3

    if mu_min is not None:
        warnings.warn(
            "Parameter 'mu_min' is deprecated and will be removed in v2.0.0. "
            "Use 'noise_min' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        noise_min = mu_min

    if mu_max is not None:
        warnings.warn(
            "Parameter 'mu_max' is deprecated and will be removed in v2.0.0. "
            "Use 'noise_max' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        noise_max = mu_max

    evaluator = create_dpsgd_evaluator(
        sample_rate=sample_rate,
        num_steps=num_steps,
        grid_step=grid_step,
        target_alpha=alpha,  # Pass alpha so evaluator computes beta
    )

    target = CalibrationTarget(kind="err_rates", alpha=alpha, beta=beta)

    config = CalibrationConfig(
        param_min=noise_min,
        param_max=noise_max,
        target_tol=beta_tol,
        increasing=True,
    )

    result = calibrate_parameter(
        evaluator, target, config, parameter_name="noise_multiplier"
    )

    return result.parameter_value


# Helper functions for analysis (not calibration)


def get_advantage_for_dpsgd(
    noise_multiplier: float, sample_rate: float, num_steps: int, grid_step: float = 1e-4
) -> float:
    """
    Compute advantage for DP-SGD with given parameters.

    Args:
        noise_multiplier: Noise scale parameter.
        sample_rate: Poisson sampling rate.
        num_steps: Number of DP-SGD steps.
        grid_step: PLD discretization interval.

    Returns:
        Attack advantage value.
    """
    evaluator = create_dpsgd_evaluator(sample_rate, num_steps, grid_step)
    metrics = evaluator(noise_multiplier)
    return metrics.advantage


def get_beta_for_dpsgd(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    alpha: Union[float, np.ndarray],
    grid_step: float = 1e-4,
) -> Union[float, np.ndarray]:
    """
    Compute beta (FNR) for DP-SGD at given alpha (FPR).

    Args:
        noise_multiplier: Noise scale parameter.
        sample_rate: Poisson sampling rate.
        num_steps: Number of DP-SGD steps.
        alpha: False positive rate(s) in [0, 1]. Can be scalar or array.
        grid_step: PLD discretization interval.

    Returns:
        False negative rate(s) corresponding to input alpha.
    """
    from riskcal.analysis import get_beta_from_pld

    pld = pld_module.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
        value_discretization_interval=grid_step,
    ).self_compose(num_steps)

    return get_beta_from_pld(pld, alpha=alpha)
