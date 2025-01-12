from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize_scalar

import riskcal
from riskcal.dpsgd import inverse_monotone_function

'''
See Appendix A of Kulynych et al. (https://arxiv.org/pdf/2407.02191) 
for an overview of these functions. As mentioned in the Appendix A, 
the direct approach, implemented in plrv.py, is the preferred method. 
'''

def find_noise_multiplier_for_epsilon_delta(
    accountant: "opacus.accountants.accountant.IAccountant",
    sample_rate: float,
    num_steps: int,
    epsilon: float,
    delta: float,
    eps_error: float = 0.001,
    mu_error: float = 0.1,
    mu_min: float = 0, 
    mu_max: float = 100.0,
    **accountant_kwargs,
) -> float:
    """
    Find a noise multiplier that satisfies a given target epsilon.
    Adapted from https://github.com/microsoft/prv_accountant/blob/main/prv_accountant/dpsgd.py

    :param accountant: Opacus-compatible accountant
    :param sample_rate: Probability of a record being in batch for Poisson sampling
    :param num_steps: Number of optimisation steps
    :param epsilon: Desired target epsilon
    :param delta: Value of DP delta
    :param float eps_error: numeric threshold for convergence in epsilon.
    :param float mu_error: numeric threshold for convergence in mu / noise multiplier
    :param float mu_min: Minimum value of noise multiplier of the search.
    :param float mu_max: Maximum value of noise multiplier of the search.
    :param accountant_kwargs: Parameters passed to the accountant's `get_epsilon`
    """

    def _compute_epsilon(mu: float) -> float:
        acc = accountant()
        for step in range(num_steps):
            acc.step(noise_multiplier=mu, sample_rate=sample_rate)

        return acc.get_epsilon(delta=delta, **accountant_kwargs)
    
    bounds = [mu_min, mu_max]
    mu = inverse_monotone_function(f = _compute_epsilon, 
                                   f_target = epsilon, 
                                   bounds = bounds, 
                                   func_threshold = eps_error,  
                                   arg_threshold = mu_error,
                                   increasing = False
                                  )
    
    return mu

def find_noise_multiplier_for_advantage(
    accountant: "opacus.accountants.accountant.IAccountant",
    advantage: float,
    sample_rate: float,
    num_steps: float,
    eps_error: float = 0.001,
    mu_error: float = 0.1,
    mu_min: float = 0, 
    mu_max: float = 100.0,
    **accountant_kwargs,
):
    """
    Find a noise multiplier that satisfies given levels of attack advantage.

    :param accountant: Opacus-compatible accountant
    :param advantage: Attack advantage bound
    :param sample_rate: Probability of a record being in batch for Poisson sampling
    :param num_steps: Number of optimisation steps
    :param float eps_error: numeric threshold for convergence in epsilon.
    :param float mu_error: numeric threshold for convergence in mu / noise multiplier 
    :param float mu_min: Minimum value of noise multiplier of the search.
    :param float mu_max: Maximum value of noise multiplier of the search
    :param accountant_kwargs: Parameters passed to the accountant's `get_epsilon`
    """
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
        self.mu_min=mu_min
        self.accountant_kwargs = accountant_kwargs

    def find_noise_multiplier(self, delta):
        epsilon = riskcal.conversions.get_epsilon_for_err_rates(
            delta, self.alpha, self.beta
        )
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
                f"Error occured in grid search w/ {epsilon=:.4f} {delta=:.4f}"
            )
            warnings.warn(e)
            return np.inf


@dataclass
class CalibrationResult:
    """
    Result of generic calibration.
    """

    noise_multiplier: float
    calibration_epsilon: float
    calibration_delta: float


    
def find_noise_multiplier_for_err_rates(
    accountant: "opacus.accountants.accountant.IAccountant",
    alpha: float,
    beta: float,
    sample_rate: float,
    num_steps: float,
    delta_error: float = 0.01,
    eps_error: float = 0.001,
    mu_min: float = 0,
    mu_max: float = 100.0,
    method: str ="bounded",
    **accountant_kwargs,
):
    """
    Find a noise multiplier that limits attack FPR/FNR rates. 
    Requires minimizing the function find_noise_multiplier(delta)
    over all delta. Currently, only the bounded bethod is supported
    to do this minimization. 

    :param accountant: Opacus-compatible accountant
    :param alpha: Attack FPR bound
    :param beta: Attack FNR bound
    :param sample_rate: Probability of a record being in batch for Poisson sampling
    :param num_steps: Number of optimisation steps
    :param float delta_error: Error allowed for delta used for calibration
    :param float eps_error: Error allowed for final epsilon
    :param float mu_min: Minimum value of noise multiplier of the search.
    :param float mu_max: Maximum value of noise multiplier of the search
    :param str method: Optimization method. Only ['bounded'] supported for now
    :param accountant_kwargs: Parameters passed to the accountant's `get_epsilon`
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
            method = 'bounded'
        )
        if not opt_result.success:
            raise RuntimeError(f"Optimization failed: {opt_result.message}")
        calibration_delta = opt_result.x
        noise_multiplier = opt_result.fun

    else:
        raise ValueError(f"Unknown optimization method: {method}")

    return CalibrationResult(
        noise_multiplier=noise_multiplier,
        calibration_delta=calibration_delta,
        calibration_epsilon=riskcal.conversions.get_epsilon_for_err_rates(
            calibration_delta, alpha, beta
        ),
    )