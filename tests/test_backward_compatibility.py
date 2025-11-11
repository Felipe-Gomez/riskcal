"""Tests for backward compatibility and deprecation warnings."""

import pytest
import numpy as np
import sys
import importlib
from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism


# Global tolerances for test assertions
REL_TOLERANCE = 1e-4
ABS_TOLERANCE = 1e-12


# ============================================================================
# Deprecated Module Imports
# ============================================================================


def test_dpsgd_module_import_warns():
    """Test that importing riskcal.dpsgd emits deprecation warning."""
    # Remove from sys.modules if already imported
    if "riskcal.dpsgd" in sys.modules:
        del sys.modules["riskcal.dpsgd"]

    with pytest.warns(DeprecationWarning, match="riskcal.dpsgd.*deprecated"):
        import riskcal.dpsgd  # noqa: F401


def test_conversions_module_import_warns():
    """Test that importing riskcal.conversions emits deprecation warning."""
    # Remove from sys.modules if already imported
    if "riskcal.conversions" in sys.modules:
        del sys.modules["riskcal.conversions"]

    with pytest.warns(DeprecationWarning, match="riskcal.conversions.*deprecated"):
        import riskcal.conversions  # noqa: F401


def test_blackbox_module_import_warns():
    """Test that importing riskcal.blackbox emits deprecation warning."""
    # Remove from sys.modules if already imported
    if "riskcal.blackbox" in sys.modules:
        del sys.modules["riskcal.blackbox"]

    with pytest.warns(DeprecationWarning, match="riskcal.blackbox.*deprecated"):
        import riskcal.blackbox  # noqa: F401


def test_plrv_module_import_warns():
    """Test that importing riskcal.plrv emits deprecation warning."""
    # Remove from sys.modules if already imported
    if "riskcal.plrv" in sys.modules:
        del sys.modules["riskcal.plrv"]

    with pytest.warns(DeprecationWarning, match="riskcal.plrv.*deprecated"):
        import riskcal.plrv  # noqa: F401


# ============================================================================
# Deprecated Parameter Names
# ============================================================================


def test_get_beta_from_pld_alphas_parameter_warns():
    """Test that 'alphas' parameter emits deprecation warning."""
    pld = from_gaussian_mechanism(1.0, value_discretization_interval=1e-3)

    # Import from new location
    from riskcal.analysis import get_beta_from_pld

    with pytest.warns(DeprecationWarning, match="'alphas'.*deprecated"):
        beta = get_beta_from_pld(pld, alphas=0.1)
        assert isinstance(beta, (float, np.floating))


def test_find_noise_multiplier_advantage_error_param_warns():
    """Test that deprecated 'advantage_error' parameter warns."""
    from riskcal.calibration.dpsgd import find_noise_multiplier_for_advantage

    with pytest.warns(DeprecationWarning, match="'advantage_error'.*deprecated"):
        noise = find_noise_multiplier_for_advantage(
            advantage=0.1,
            sample_rate=1,
            num_steps=1,
            grid_step=1e-2,
            advantage_error=1e-2,
        )
        assert noise > 0


def test_find_noise_multiplier_mu_min_param_warns():
    """Test that deprecated 'mu_min' parameter warns."""
    from riskcal.calibration.dpsgd import find_noise_multiplier_for_advantage

    with pytest.warns(DeprecationWarning, match="'mu_min'.*deprecated"):
        noise = find_noise_multiplier_for_advantage(
            advantage=0.5,
            sample_rate=1,
            num_steps=1,
            grid_step=1e-2,
            mu_min=0.1,
        )
        assert noise > 0


def test_find_noise_multiplier_mu_max_param_warns():
    """Test that deprecated 'mu_max' parameter warns."""
    from riskcal.calibration.dpsgd import find_noise_multiplier_for_advantage

    with pytest.warns(DeprecationWarning, match="'mu_max'.*deprecated"):
        noise = find_noise_multiplier_for_advantage(
            advantage=0.5,
            sample_rate=1,
            num_steps=1,
            grid_step=1e-2,
            mu_max=50.0,
        )
        assert noise > 0


def test_find_noise_multiplier_mu_error_param_warns():
    """Test that deprecated 'mu_error' parameter warns."""
    from riskcal.calibration.dpsgd import find_noise_multiplier_for_advantage

    with pytest.warns(DeprecationWarning, match="'mu_error'.*deprecated"):
        noise = find_noise_multiplier_for_advantage(
            advantage=0.5,
            sample_rate=1,
            num_steps=1,
            grid_step=1e-1,
            mu_error=1e-2,
        )
        assert noise > 0


def test_find_noise_multiplier_err_rates_beta_error_param_warns():
    """Test that deprecated 'beta_error' parameter warns."""
    from riskcal.calibration.dpsgd import find_noise_multiplier_for_err_rates

    with pytest.warns(DeprecationWarning, match="'beta_error'.*deprecated"):
        noise = find_noise_multiplier_for_err_rates(
            alpha=0.1,
            beta=0.2,
            sample_rate=1,
            num_steps=1,
            grid_step=1e-1,
            beta_error=1e-2,
        )
        assert noise > 0


def test_find_noise_multiplier_err_rates_mu_min_param_warns():
    """Test that deprecated 'mu_min' parameter warns in err_rates function."""
    from riskcal.calibration.dpsgd import find_noise_multiplier_for_err_rates

    with pytest.warns(DeprecationWarning, match="'mu_min'.*deprecated"):
        noise = find_noise_multiplier_for_err_rates(
            alpha=0.1,
            beta=0.2,
            sample_rate=1,
            num_steps=1,
            grid_step=1e-1,
            mu_min=0.1,
        )
        assert noise > 0


def test_find_noise_multiplier_err_rates_mu_max_param_warns():
    """Test that deprecated 'mu_max' parameter warns in err_rates function."""
    from riskcal.calibration.dpsgd import find_noise_multiplier_for_err_rates

    with pytest.warns(DeprecationWarning, match="'mu_max'.*deprecated"):
        noise = find_noise_multiplier_for_err_rates(
            alpha=0.1,
            beta=0.2,
            sample_rate=1,
            num_steps=1,
            grid_step=1e-1,
            mu_max=50.0,
        )
        assert noise > 0


# ============================================================================
# Deprecated Function Aliases
# ============================================================================


def test_get_beta_for_mu_alias_works():
    """Test that get_beta_for_mu is an alias for get_beta_from_gdp."""
    from riskcal.analysis import get_beta_for_mu, get_beta_from_gdp

    noise_multiplier = 1.0
    alpha = 0.1

    beta_alias = get_beta_for_mu(noise_multiplier, alpha)
    beta_new = get_beta_from_gdp(noise_multiplier, alpha)

    assert beta_alias == pytest.approx(beta_new, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)


def test_get_advantage_for_mu_alias_works():
    """Test that get_advantage_for_mu is an alias for get_advantage_from_gdp."""
    from riskcal.analysis import get_advantage_for_mu, get_advantage_from_gdp

    noise_multiplier = 1.0

    advantage_alias = get_advantage_for_mu(noise_multiplier)
    advantage_new = get_advantage_from_gdp(noise_multiplier)

    assert advantage_alias == pytest.approx(
        advantage_new, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )


def test_get_bayes_risk_for_mu_alias_works():
    """Test that get_bayes_risk_for_mu is an alias for get_bayes_risk_from_gdp."""
    from riskcal.analysis import get_bayes_risk_for_mu, get_bayes_risk_from_gdp

    noise_multiplier = 1.0
    prior = 0.5

    risk_alias = get_bayes_risk_for_mu(noise_multiplier, prior)
    risk_new = get_bayes_risk_from_gdp(noise_multiplier, prior)

    assert risk_alias == pytest.approx(risk_new, rel=REL_TOLERANCE, abs=ABS_TOLERANCE)


def test_plrvs_from_pld_alias_works():
    """Test that plrvs_from_pld is an alias for pld_to_plrvs."""
    from riskcal.analysis import plrvs_from_pld, pld_to_plrvs

    pld = from_gaussian_mechanism(1.0, value_discretization_interval=1e-3)

    plrvs_alias = plrvs_from_pld(pld)
    plrvs_new = pld_to_plrvs(pld)

    # Check that both produce the same structure
    assert plrvs_alias.y0 == plrvs_new.y0
    assert plrvs_alias.x0 == plrvs_new.x0
    assert np.allclose(plrvs_alias.pmf_Y, plrvs_new.pmf_Y)
    assert np.allclose(plrvs_alias.pmf_X, plrvs_new.pmf_X)


def test_get_epsilon_for_err_rates_alias_works():
    """Test that get_epsilon_for_err_rates is an alias for get_epsilon_from_err_rates."""
    from riskcal.analysis import get_epsilon_for_err_rates, get_epsilon_from_err_rates

    delta = 0.001
    alpha = 0.01
    beta = 0.1

    epsilon_alias = get_epsilon_for_err_rates(delta, alpha, beta)
    epsilon_new = get_epsilon_from_err_rates(delta, alpha, beta)

    assert epsilon_alias == pytest.approx(
        epsilon_new, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
    )


# ============================================================================
# Legacy API Still Works
# ============================================================================


def test_legacy_dpsgd_get_advantage_for_dpsgd_works():
    """Test that get_advantage_for_dpsgd from deprecated module works."""
    # Import emits warning (may already be imported, so we don't check for warning)
    from riskcal.dpsgd import get_advantage_for_dpsgd

    advantage = get_advantage_for_dpsgd(
        noise_multiplier=1.0, sample_rate=0.01, num_steps=100
    )
    assert 0 < advantage < 1


def test_legacy_dpsgd_get_beta_for_dpsgd_works():
    """Test that get_beta_for_dpsgd from deprecated module works."""
    # Import emits warning (may already be imported, so we don't check for warning)
    from riskcal.dpsgd import get_beta_for_dpsgd

    beta = get_beta_for_dpsgd(
        noise_multiplier=1.0, sample_rate=0.01, num_steps=100, alpha=0.1
    )
    assert 0 < beta < 1


def test_legacy_dpsgd_find_noise_multiplier_for_advantage_works():
    """Test that find_noise_multiplier_for_advantage from deprecated module works."""
    # Import emits warning (may already be imported, so we don't check for warning)
    from riskcal.dpsgd import find_noise_multiplier_for_advantage

    noise = find_noise_multiplier_for_advantage(
        advantage=0.5, sample_rate=1, num_steps=1, grid_step=1e-2
    )
    assert noise > 0


def test_legacy_conversions_get_beta_from_pld_works():
    """Test that get_beta_from_pld from deprecated module works."""
    pld = from_gaussian_mechanism(1.0, value_discretization_interval=1e-3)

    # Import emits warning (may already be imported, so we don't check for warning)
    from riskcal.conversions import get_beta_from_pld

    beta = get_beta_from_pld(pld, alpha=0.1)
    assert 0 < beta < 1


def test_legacy_conversions_get_advantage_from_pld_works():
    """Test that get_advantage_from_pld from deprecated module works."""
    pld = from_gaussian_mechanism(1.0, value_discretization_interval=1e-3)

    # Import emits warning (may already be imported, so we don't check for warning)
    from riskcal.conversions import get_advantage_from_pld

    advantage = get_advantage_from_pld(pld)
    assert 0 < advantage < 1


def test_legacy_CTDAccountant_works():
    """Test that CTDAccountant imported from deprecated module works."""
    # Import emits warning (may already be imported, so we don't check for warning)
    from riskcal.dpsgd import CTDAccountant

    acct = CTDAccountant()
    for _ in range(10):
        acct.step(noise_multiplier=1.0, sample_rate=0.01)

    beta = acct.get_beta(alpha=0.1)
    assert 0 < beta < 1

    advantage = acct.get_advantage()
    assert 0 < advantage < 1


# ============================================================================
# CalibrationResult Backward Compatibility
# ============================================================================


def test_calibration_result_noise_multiplier_property():
    """Test that CalibrationResult.noise_multiplier property works."""
    from riskcal.calibration.dpsgd import create_dpsgd_evaluator
    from riskcal.calibration.core import (
        CalibrationTarget,
        CalibrationConfig,
        calibrate_parameter,
    )

    evaluator = create_dpsgd_evaluator(sample_rate=1, num_steps=1)
    target = CalibrationTarget(kind="advantage", advantage=0.5)
    config = CalibrationConfig(param_min=0.05, param_max=10.0, increasing=False)

    result = calibrate_parameter(
        evaluator, target, config, parameter_name="noise_multiplier"
    )

    # Test backward compatibility property
    assert result.noise_multiplier == result.parameter_value
    assert result.noise_multiplier > 0
