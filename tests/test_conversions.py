import pytest
import numpy as np

from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism

import riskcal


def get_gaussian_plrv_data():
    mu = 2.0
    pld = from_gaussian_mechanism(standard_deviation=1.0 / mu)
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
    assert pytest.approx(analytic_beta) == riskcal.plrv.get_beta(plrvs, alpha=alpha)
    assert pytest.approx(analytic_beta) == riskcal.conversions.get_beta_from_pld(
        pld, alpha=alpha
    )


def test_get_beta_symmetrized_in_the_middle():
    alpha_grid_step = 1e-4
    mu, pld, plrvs = get_subsampled_gaussian_plrv_data()
    advantage = pld.get_delta_for_epsilon(0)
    fixed_point_beta = 0.5 * (1 - advantage)
    beta = riskcal.conversions.get_beta_from_pld(
        pld,
        alpha=np.linspace(0, 1, int(1 / alpha_grid_step)),
        alpha_grid_step=alpha_grid_step,
    )
    assert (
        pytest.approx(np.min(np.abs(beta - fixed_point_beta)), abs=2 * alpha_grid_step)
        == 0.0
    )


@pytest.mark.parametrize(
    "alpha",
    [
        0.1,
        np.array([0.1, 0.2, 0.3]),
    ],
)
def test_get_advantage_matches_analytic_matching_expression(alpha):
    mu, pld, plrvs = get_gaussian_plrv_data()
    analytic_advantage = riskcal.conversions.get_advantage_for_mu(mu)
    assert pytest.approx(
        analytic_advantage
    ) == riskcal.conversions.get_advantage_from_pld(pld)


@pytest.mark.parametrize("epsilon", [0.1, 1.0, 2.0])
def test_zcdp_of_randomized_response(epsilon):
    rho = 0.5 * epsilon**2
    orders = 5

    # Check advantage.
    adv = riskcal.conversions.get_advantage_for_rho(
        rho=rho, orders=orders, verbose=True
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
        rho=rho, alpha=alpha, verbose=True, orders=orders
    )

    # Check that zCDP trade-off curve is always less than RR trade-off curve.
    assert np.all(beta_zcdp <= beta_rr)

    # Check that that the trade-off curve touches the advantage point.
    for alpha_val, beta_val in zip(alpha, beta_zcdp):
        if alpha_val == pytest.approx(alpha_star):
            assert beta_val == pytest.approx(alpha_val)
