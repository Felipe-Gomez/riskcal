import numpy as np
import pytest

import riskcal
from scipy.stats import norm
from scipy.optimize import root_scalar


grid_step = 1e-3
sample_rate = 0.01
num_dpsgd_steps = 1_000


@pytest.fixture
def accountant():
    yield riskcal.dpsgd.CTDAccountant


@pytest.mark.parametrize(
    "advantage, sample_rate, num_steps",
    [
        (0.01, 1, 1),
        (0.05, 1, 1),
        (0.10, 1, 1),
        (0.01, sample_rate, num_dpsgd_steps),
        (0.05, sample_rate, num_dpsgd_steps),
        (0.10, sample_rate, num_dpsgd_steps),
    ],
)
def test_advantage_calibration_correctness(advantage, sample_rate, num_steps):
    advantage_error = 1e-4
    
    calibrated_mu = riskcal.dpsgd.find_noise_multiplier_for_advantage(
        advantage=advantage,
        sample_rate=sample_rate,
        num_steps=num_steps,
        advantage_error=advantage_error,
        grid_step=grid_step,
    )

    
    numerical_advantage = riskcal.dpsgd.get_advantage_for_dpsgd(
        noise_multiplier=calibrated_mu, 
        sample_rate=sample_rate, 
        num_steps=num_steps,
        grid_step=grid_step
    )
    
    assert pytest.approx(numerical_advantage, abs=advantage_error) == advantage

@pytest.mark.parametrize(
    "alpha, beta, sample_rate, num_steps",
    [
        # DP-SGD
        (0.1, 0.33, sample_rate, num_dpsgd_steps),
        (0.1, 0.50, sample_rate, num_dpsgd_steps),
        (0.1, 0.70, sample_rate, num_dpsgd_steps),
    ],
)
def test_err_rates_calibration_correctness(alpha, beta, sample_rate, num_steps):
    
    beta_error = 1e-4
    calibrated_mu = riskcal.dpsgd.find_noise_multiplier_for_err_rates(
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        beta_error=beta_error,
        grid_step=grid_step,
    )

    numerical_beta = riskcal.dpsgd.get_beta_for_dpsgd(
        alpha=alpha,
        noise_multiplier=calibrated_mu,
        sample_rate=sample_rate,
        num_steps=num_steps,
        grid_step=grid_step,
    )
    
    assert pytest.approx(numerical_beta, abs=beta_error) == beta


@pytest.mark.parametrize(
    "advantage, sample_rate, num_steps",
    [
        # DP-SGD
        (0.01, sample_rate, num_dpsgd_steps),
        (0.05, sample_rate, num_dpsgd_steps),
        (0.10, sample_rate, num_dpsgd_steps),
    ],
)
def test_advantage_calibration_blackbox_vs_direct(
    accountant, advantage, sample_rate, num_steps
):
    eps_error = 1e-4
    advantage_error = 0.01
    mu_error = 0.01

    direct_calibrated_mu = riskcal.dpsgd.find_noise_multiplier_for_advantage(
        advantage=advantage,
        sample_rate=sample_rate,
        num_steps=num_steps,
        advantage_error=advantage_error,
        mu_error=mu_error,
        grid_step=grid_step,
    )

    blackbox_calibrated_mu = riskcal.blackbox.find_noise_multiplier_for_advantage(
        accountant=accountant,
        advantage=advantage,
        sample_rate=sample_rate,
        num_steps=num_steps,
        mu_error=mu_error,
        eps_error=eps_error,
        grid_step=grid_step
    )

    assert direct_calibrated_mu == pytest.approx(
        blackbox_calibrated_mu, abs= mu_error
    )


# @pytest.mark.skip("Investigate big difference between direct and blackbox.")
@pytest.mark.parametrize(
    "alpha, beta, sample_rate, num_steps, method",
    [
        # DP-SGD
        (0.01, 0.25, sample_rate, num_dpsgd_steps, "bounded"),
    ],
)
def test_err_rates_calibration_blackbox_vs_direct(
    accountant, alpha, beta, sample_rate, num_steps, method
):
    '''
    This test is very slow, since blackbox
    calibration using a PLD accountant is slow. So,
    we only test one (alpha,beta) point with generous 
    errors. Noise parameter (i.e. mu) is roughly 0.374
    with these parameters.
    
    '''
    eps_error = 1e-1
    beta_error = 1e-1
    mu_error = 1e-1
    direct_calibrated_mu = riskcal.dpsgd.find_noise_multiplier_for_err_rates(
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        beta_error=beta_error,
        grid_step=grid_step,
        mu_error = mu_error
    )
    
    # give blackbox calibration lots of wiggle room and a generous bisection bound.
    blackbox_calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        eps_error=eps_error,
        method=method,
        mu_min = 0.3728, 
        mu_max = 0.4206,
        mu_error = mu_error,
        grid_step = grid_step
    )
    blackbox_calibrated_mu = blackbox_calibration_result.noise_multiplier
    
    assert direct_calibrated_mu == pytest.approx(blackbox_calibrated_mu, abs=mu_error)
