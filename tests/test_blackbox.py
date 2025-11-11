import numpy as np
import pytest

import riskcal
from opacus import accountants
from scipy.stats import norm
from scipy.optimize import root_scalar


@pytest.fixture(params=[accountants.rdp.RDPAccountant])
def accountant(request):
    return request.param


sample_rate = 0.01
num_dpsgd_steps = 100


@pytest.mark.parametrize(
    "advantage, sample_rate, num_steps",
    [
        # Gaussian mechanism.
        (0.01, 1, 1),
        (0.10, 1, 1),
        (0.25, 1, 1),
        # DP-SGD.
        (0.01, sample_rate, num_dpsgd_steps),
        (0.10, sample_rate, num_dpsgd_steps),
        (0.25, sample_rate, num_dpsgd_steps),
    ],
)
def test_adv_calibration_correctness(accountant, advantage, sample_rate, num_steps):

    # chosen ad hoc for speed
    rdp_alphas = np.linspace(1.5, 101, 50)

    epsilon_error = 1e-4
    calibrated_mu = riskcal.blackbox.find_noise_multiplier_for_advantage(
        accountant,
        advantage=advantage,
        sample_rate=sample_rate,
        num_steps=num_steps,
        eps_error=epsilon_error,
        alphas=rdp_alphas,
    )

    acct_obj = accountant()
    for _ in range(num_steps):
        acct_obj.step(noise_multiplier=calibrated_mu, sample_rate=sample_rate)

    # Verify that mu is calibrated for (0, adv)-DP:
    numerical_epsilon = acct_obj.get_epsilon(delta=advantage, alphas=rdp_alphas)
    assert pytest.approx(numerical_epsilon, abs=epsilon_error) == 0


@pytest.mark.parametrize(
    "beta, sample_rate, num_steps, method",
    [
        # Gaussian mechanism:
        (0.25, 1, 1, "bounded"),
        (0.50, 1, 1, "bounded"),
        (0.75, 1, 1, "bounded"),
        # DP-SGD
        (0.25, sample_rate, num_dpsgd_steps, "bounded"),
        (0.50, sample_rate, num_dpsgd_steps, "bounded"),
        (0.75, sample_rate, num_dpsgd_steps, "bounded"),
    ],
)
def test_err_rates_calibration_correctness(
    accountant, beta, sample_rate, num_steps, method
):
    alpha = 0.01
    classical_delta = 1e-5
    delta_error = 0.001
    eps_error = 0.01
    rdp_alphas = np.linspace(1 + 1e-2, 17, 31)
    calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        delta_error=delta_error,
        alphas=rdp_alphas,
    )
    calibrated_mu = calibration_result.noise_multiplier
    calibrated_delta = calibration_result.calibration_delta

    acct_obj = accountant()
    for _ in range(num_steps):
        acct_obj.step(noise_multiplier=calibrated_mu, sample_rate=sample_rate)

    epsilon = acct_obj.get_epsilon(delta=calibrated_delta, alphas=rdp_alphas)
    expected_epsilon = calibration_result.calibration_epsilon

    print(f"CHECK 1: {alpha=}, {beta=} // {epsilon=}, {expected_epsilon=}")
    assert epsilon == pytest.approx(expected_epsilon, abs=eps_error)

    obtained_beta = riskcal.conversions.get_beta_for_epsilon_delta(
        epsilon, calibrated_delta, alpha
    )
    print(f"CHECK 2: {alpha=}, {beta=} // {epsilon=}, {obtained_beta=}")
    assert beta == pytest.approx(obtained_beta, abs=delta_error)


@pytest.mark.parametrize(
    "epsilon, sample_rate, num_steps",
    [
        (1.0, 1.0, 1),
        (4.0, 1.0, 1),
        (1.0, sample_rate, num_dpsgd_steps),
        (4.0, sample_rate, num_dpsgd_steps),
    ],
)
def test_err_rates_calibration_improvement(accountant, epsilon, sample_rate, num_steps):
    alpha = 0.01
    delta = 1e-5
    delta_error = 0.001
    method = "bounded"
    rdp_alphas = np.linspace(1 + 1e-4, 520, 45)

    standard_mu = riskcal.blackbox.find_noise_multiplier_for_epsilon_delta(
        accountant,
        epsilon=epsilon,
        delta=delta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        alphas=rdp_alphas,
    )

    # What is the FNR at alpha = 0.1 for the target epsilon?
    beta = riskcal.conversions.get_beta_for_epsilon_delta(epsilon, delta, alpha)
    # Let's calibrate directly for these alpha / beta:
    calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        accountant,
        alpha=alpha,
        beta=beta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        delta_error=delta_error,
        method=method,
        alphas=rdp_alphas,
    )
    calibrated_mu = calibration_result.noise_multiplier
    calibrated_delta = calibration_result.calibration_delta

    # We should get less noise with direct calibration:
    assert standard_mu > calibrated_mu

    # Check that alpha beta guarantees are correct.
    acct_obj = accountant()
    for step in range(num_steps):
        acct_obj.step(noise_multiplier=calibrated_mu, sample_rate=sample_rate)
    obtained_epsilon = acct_obj.get_epsilon(delta=calibrated_delta, alphas=rdp_alphas)

    obtained_beta = riskcal.conversions.get_beta_for_epsilon_delta(
        obtained_epsilon, calibrated_delta, alpha
    )
    assert beta == pytest.approx(obtained_beta, abs=delta_error)


@pytest.mark.parametrize(
    "beta, method",
    [
        (0.25, "bounded"),
        (0.50, "bounded"),
        (0.75, "bounded"),
    ],
)
def test_generic_err_rates_calibration_worse_than_exact(beta, method):
    alpha = 0.1
    delta_error = 1e-2

    class GaussianMechanismAccountant:
        def __init__(self):
            pass

        def step(self, noise_multiplier: float, **kwargs):
            self.noise_multiplier = noise_multiplier

        def get_epsilon(self, delta):
            def get_delta(epsilon):
                return norm.cdf(
                    0.5 / self.noise_multiplier - epsilon * self.noise_multiplier
                ) - np.exp(epsilon) * norm.cdf(
                    -0.5 / self.noise_multiplier - epsilon * self.noise_multiplier
                )

            return root_scalar(lambda epsilon: get_delta(epsilon) - delta, x0=0.0).root

    calibration_result = riskcal.blackbox.find_noise_multiplier_for_err_rates(
        GaussianMechanismAccountant,
        alpha=alpha,
        beta=beta,
        sample_rate=1,
        num_steps=1,
        delta_error=delta_error,
        method=method,
    )

    exact_noise_multiplier = 1 / (norm.ppf(1 - alpha) - norm.ppf(beta))
    # assert exact_noise_multiplier <= calibration_result.noise_multiplier
    assert exact_noise_multiplier == pytest.approx(
        calibration_result.noise_multiplier, abs=delta_error
    )
