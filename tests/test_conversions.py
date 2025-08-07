import pytest
import numpy as np

from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism

import riskcal


# Global tolerances for test assertions
REL_TOLERANCE = 1e-4  # Relative tolerance for numerical comparisons
ABS_TOLERANCE = 1e-12  # Absolute tolerance for small numbers


def get_gaussian_plrv_data():
    mu = 2.0
    pld = from_gaussian_mechanism(
        standard_deviation=1.0 / mu, value_discretization_interval=1e-3
    )
    return mu, pld, riskcal.conversions.plrvs_from_pld(pld)


def get_subsampled_gaussian_plrv_data():
    mu = 2.5
    pld = from_gaussian_mechanism(
        standard_deviation=1.0 / mu, sampling_prob=0.1
    ).self_compose(5)
    return mu, pld, riskcal.conversions.plrvs_from_pld(pld)


@pytest.mark.parametrize(
    "alpha",
    [
        0.1,
        np.array([0.1, 0.2, 0.3]),
    ],
)
def test_get_beta_matches_analytic_curve(alpha):
    mu, pld, plrvs = get_gaussian_plrv_data()
    analytic_beta = riskcal.conversions.get_beta_for_mu(mu, alpha)
    numerical_beta1 = riskcal.plrv.get_beta(plrvs, alpha=alpha)
    numerical_beta2 = riskcal.conversions.get_beta_from_pld(pld, alpha=alpha)

    # check that numerical beta is close to analytic beta
    assert (
        pytest.approx(numerical_beta1, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)
        == analytic_beta
    )
    assert (
        pytest.approx(numerical_beta2, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)
        == analytic_beta
    )


def test_deprecated_get_beta_interface():
    mu, pld, plrvs = get_gaussian_plrv_data()
    alpha = 0.5
    analytic_beta = riskcal.conversions.get_beta_for_mu(mu, alpha)

    with pytest.warns(DeprecationWarning):
        numerical_beta1 = riskcal.plrv.get_beta(plrvs, alphas=alpha)
        numerical_beta2 = riskcal.conversions.get_beta_from_pld(pld, alphas=alpha)

    # check that numerical beta is close to analytic beta
    assert (
        pytest.approx(numerical_beta1, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)
        == analytic_beta
    )
    assert (
        pytest.approx(numerical_beta2, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)
        == analytic_beta
    )


def test_get_beta_symmetrized_in_the_middle():
    mu, pld, plrvs = get_subsampled_gaussian_plrv_data()
    analytic_advantage = pld.get_delta_for_epsilon(0)
    analytic_fixed_point_beta = 0.5 * (1 - analytic_advantage)

    alpha = np.linspace(0, 1, 100_000)
    beta = riskcal.conversions.get_beta_from_pld(
        pld,
        alpha=alpha,
    )
    numerical_fixed_point_idx = np.argmin(np.abs(beta - analytic_fixed_point_beta))
    numerical_fixed_point = beta[numerical_fixed_point_idx]

    assert (
        pytest.approx(numerical_fixed_point, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)
        == analytic_fixed_point_beta
    )


def test_get_advantage_matches_analytic_matching_expression():
    mu, pld, plrvs = get_gaussian_plrv_data()
    analytic_advantage = riskcal.conversions.get_advantage_for_mu(mu)
    numerical_advantage = riskcal.conversions.get_advantage_from_pld(pld)
    assert (
        pytest.approx(numerical_advantage, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)
        == analytic_advantage
    )


@pytest.mark.parametrize(
    "epsilon", [0.001, 0.01, 0.1, 0.15, 0.2, 1.0, 2.0, 10.0, 25.0, 50.0, 100.0]
)
def test_zcdp_of_randomized_response(epsilon):
    rho = 0.5 * epsilon**2

    # Check advantage.
    adv = riskcal.conversions.get_advantage_for_rho(
        rho=rho,
    )
    # Closed-form solution for RR.
    actual_adv = (np.exp(epsilon) - 1) / (np.exp(epsilon) + 1)
    # Estimated advantage does not overestimate the true advantage by more than 5 pp.
    assert adv - actual_adv <= 0.05

    # Alpha that achieves the highest advantage.
    alpha_star = (1 - adv) / 2

    alpha = np.linspace(0, 1, 20)
    beta_rr = riskcal.conversions.get_beta_for_epsilon_delta(
        epsilon=epsilon, delta=0.0, alpha=alpha
    )
    beta_zcdp = riskcal.conversions.get_beta_for_rho(
        rho=rho, alpha=alpha, tol=ABS_TOLERANCE
    )

    # Check that zCDP trade-off curve is always less than RR trade-off curve.
    diff = beta_zcdp - beta_rr
    max_violation = diff.max()
    assert max_violation <= ABS_TOLERANCE

    # Check that that the trade-off curve touches the advantage point.
    for alpha_val, beta_val in zip(alpha, beta_zcdp):
        if alpha_val == pytest.approx(alpha_star):
            assert beta_val == pytest.approx(alpha_val)
