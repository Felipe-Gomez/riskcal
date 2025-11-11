"""Accountant-based calibration for generic mechanisms.

See Appendix A of Kulynych et al. (https://arxiv.org/abs/2407.02191)
for an overview of these functions. As mentioned in Appendix A,
the direct approach (using PLDs) is the preferred method when available.
"""

from dataclasses import dataclass
from typing import Type, Any
import warnings
import numpy as np
from scipy.optimize import minimize_scalar

from riskcal.calibration.core import (
    PrivacyEvaluator,
    PrivacyMetrics,
    CalibrationTarget,
    CalibrationConfig,
    calibrate_parameter,
    CalibrationResult as CoreCalibrationResult,
)


def create_accountant_evaluator(
    accountant_class: Type,
    sample_rate: float,
    num_steps: int,
    target_delta: float | None = None,
    target_alpha: float | None = None,
    **accountant_kwargs,
) -> PrivacyEvaluator:
    """
    Create a privacy evaluator from an Opacus-compatible accountant.

    Returns a function that uses the accountant's epsilon-delta interface
    to evaluate privacy for a given noise level.

    Args:
        accountant_class: Opacus-compatible accountant class (e.g., RDPAccountant).
        sample_rate: Poisson sampling rate.
        num_steps: Number of steps.
        target_delta: Delta for epsilon computation (required for epsilon_delta calibration).
        target_alpha: Alpha for beta computation (required for err_rates calibration).
        **accountant_kwargs: Additional arguments passed to accountant's get_epsilon.

    Returns:
        PrivacyEvaluator function.

    Note:
        For err_rates calibration via accountants, this uses conversion from
        (epsilon, delta) to (alpha, beta), which may be slower than direct PLD methods.
    """

    def evaluator(noise_multiplier: float) -> PrivacyMetrics:
        """Evaluate privacy using accountant."""
        # Create fresh accountant instance
        acct = accountant_class()

        # Compose for num_steps
        for _ in range(num_steps):
            acct.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        metrics = PrivacyMetrics()

        # Compute epsilon if target_delta specified
        if target_delta is not None:
            epsilon = acct.get_epsilon(delta=target_delta, **accountant_kwargs)
            metrics.epsilon = epsilon
            metrics.delta = target_delta

            # Compute advantage from epsilon (advantage = delta at epsilon=0)
            # For most accountants, this would be expensive to compute directly
            # We skip it here unless specifically needed

        # Compute beta if target_alpha specified (requires conversion)
        if target_alpha is not None and target_delta is not None:
            from riskcal.analysis import get_beta_for_epsilon_delta

            epsilon = acct.get_epsilon(delta=target_delta, **accountant_kwargs)
            beta = get_beta_for_epsilon_delta(epsilon, target_delta, target_alpha)
            metrics.alpha = target_alpha
            metrics.beta = beta

        return metrics

    return evaluator


# Legacy wrapper functions


def find_noise_multiplier_for_epsilon_delta(
    accountant: Type,
    sample_rate: float,
    num_steps: int,
    epsilon: float,
    delta: float,
    eps_error: float = 0.001,
    mu_error: float = 0.1,
    mu_min: float = 0.05,
    mu_max: float = 100.0,
    **accountant_kwargs,
) -> float:
    """
    Find a noise multiplier that satisfies a given target epsilon.

    Adapted from https://github.com/microsoft/prv_accountant/blob/main/prv_accountant/dpsgd.py

    Args:
        accountant: Opacus-compatible accountant class.
        sample_rate: Probability of a record being in batch for Poisson sampling.
        num_steps: Number of optimization steps.
        epsilon: Desired target epsilon.
        delta: Value of DP delta.
        eps_error: Numeric threshold for convergence in epsilon.
        mu_error: Numeric threshold for convergence in mu / noise multiplier.
        mu_min: Minimum value of noise multiplier of the search.
        mu_max: Maximum value of noise multiplier of the search.
        **accountant_kwargs: Parameters passed to the accountant's `get_epsilon`.

    Returns:
        Calibrated noise_multiplier (float).
    """
    evaluator = create_accountant_evaluator(
        accountant_class=accountant,
        sample_rate=sample_rate,
        num_steps=num_steps,
        target_delta=delta,
        **accountant_kwargs,
    )

    target = CalibrationTarget(kind="epsilon_delta", epsilon=epsilon, delta=delta)

    config = CalibrationConfig(
        param_min=mu_min,
        param_max=mu_max,
        target_tol=eps_error,
        increasing=False,
    )

    result = calibrate_parameter(
        evaluator, target, config, parameter_name="noise_multiplier"
    )

    return result.parameter_value


def find_noise_multiplier_for_advantage(
    accountant: Type,
    advantage: float,
    sample_rate: float,
    num_steps: int,
    eps_error: float = 0.001,
    mu_error: float = 0.1,
    mu_min: float = 0.05,
    mu_max: float = 100.0,
    **accountant_kwargs,
) -> float:
    """
    Find a noise multiplier that satisfies given levels of attack advantage.

    Args:
        accountant: Opacus-compatible accountant class.
        advantage: Attack advantage bound.
        sample_rate: Probability of a record being in batch for Poisson sampling.
        num_steps: Number of optimization steps.
        eps_error: Numeric threshold for convergence in epsilon.
        mu_error: Numeric threshold for convergence in mu / noise multiplier.
        mu_min: Minimum value of noise multiplier of the search.
        mu_max: Maximum value of noise multiplier of the search.
        **accountant_kwargs: Parameters passed to the accountant's `get_epsilon`.

    Returns:
        Calibrated noise_multiplier (float).
    """
    # Advantage calibration via epsilon=0, delta=advantage
    return find_noise_multiplier_for_epsilon_delta(
        accountant=accountant,
        sample_rate=sample_rate,
        num_steps=num_steps,
        epsilon=0.0,
        delta=advantage,
        eps_error=eps_error,
        mu_error=mu_error,
        mu_min=mu_min,
        mu_max=mu_max,
        **accountant_kwargs,
    )


class _ErrRatesAccountant:
    """Helper class for error rates calibration."""

    def __init__(
        self,
        accountant,
        alpha,
        beta,
        sample_rate,
        num_steps,
        eps_error,
        mu_min=0,
        mu_max=100.0,
        **accountant_kwargs,
    ):
        self.accountant = accountant
        self.alpha = alpha
        self.beta = beta
        self.sample_rate = sample_rate
        self.num_steps = num_steps
        self.eps_error = eps_error
        self.mu_max = mu_max
        self.mu_min = mu_min
        self.accountant_kwargs = accountant_kwargs

    def find_noise_multiplier(self, delta):
        """Find noise multiplier for given delta."""
        from riskcal.analysis import get_epsilon_for_err_rates

        epsilon = get_epsilon_for_err_rates(delta, self.alpha, self.beta)
        try:
            mu = find_noise_multiplier_for_epsilon_delta(
                epsilon=epsilon,
                delta=delta,
                accountant=self.accountant,
                sample_rate=self.sample_rate,
                num_steps=self.num_steps,
                eps_error=self.eps_error,
                mu_min=self.mu_min,
                mu_max=self.mu_max,
                **self.accountant_kwargs,
            )
            return mu

        except RuntimeError as e:
            warnings.warn(
                f"Error occurred in grid search w/ {epsilon=:.4f} {delta=:.4f}"
            )
            warnings.warn(str(e))
            return np.inf


@dataclass
class CalibrationResult:
    """
    Result of generic calibration (legacy format for backward compatibility).
    """

    noise_multiplier: float
    calibration_epsilon: float
    calibration_delta: float


def find_noise_multiplier_for_err_rates(
    accountant: Type,
    alpha: float,
    beta: float,
    sample_rate: float,
    num_steps: int,
    delta_error: float = 0.01,
    eps_error: float = 0.001,
    mu_min: float = 0.05,
    mu_max: float = 100.0,
    method: str = "bounded",
    **accountant_kwargs,
) -> CalibrationResult:
    """
    Find a noise multiplier that limits attack FPR/FNR rates.

    Requires minimizing the function find_noise_multiplier(delta)
    over all delta. Currently, only the bounded method is supported
    to do this minimization.

    Args:
        accountant: Opacus-compatible accountant class.
        alpha: Attack FPR bound.
        beta: Attack FNR bound.
        sample_rate: Probability of a record being in batch for Poisson sampling.
        num_steps: Number of optimization steps.
        delta_error: Error allowed for delta used for calibration.
        eps_error: Error allowed for final epsilon.
        mu_min: Minimum value of noise multiplier of the search.
        mu_max: Maximum value of noise multiplier of the search.
        method: Optimization method. Only ['bounded'] supported for now.
        **accountant_kwargs: Parameters passed to the accountant's `get_epsilon`.

    Returns:
        CalibrationResult with noise_multiplier, calibration_epsilon, and calibration_delta.

    Note:
        This is slower than DP-SGD direct method as it requires
        optimization over delta parameter and conversion between representations.
    """
    if alpha + beta >= 1:
        raise ValueError(
            f"The guarantees are vacuous when alpha + beta >= 1. Got {alpha=}, {beta=}"
        )

    max_delta = 1 - alpha - beta
    err_rates_acct_obj = _ErrRatesAccountant(
        accountant=accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        eps_error=eps_error,
        mu_min=mu_min,
        mu_max=mu_max,
        **accountant_kwargs,
    )

    if max_delta < delta_error:
        raise ValueError(f"{delta_error=} too low for the requested error rates.")

    if method == "bounded":

        opt_result = minimize_scalar(
            err_rates_acct_obj.find_noise_multiplier,
            bounds=[delta_error, max_delta],
            options=dict(xatol=delta_error),
            method="bounded",
        )
        if not opt_result.success:
            raise RuntimeError(f"Optimization failed: {opt_result.message}")
        calibration_delta = opt_result.x
        noise_multiplier = opt_result.fun

    else:
        raise ValueError(f"Unknown optimization method: {method}")

    from riskcal.analysis import get_epsilon_for_err_rates

    return CalibrationResult(
        noise_multiplier=noise_multiplier,
        calibration_delta=calibration_delta,
        calibration_epsilon=get_epsilon_for_err_rates(calibration_delta, alpha, beta),
    )
