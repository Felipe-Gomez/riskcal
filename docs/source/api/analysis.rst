riskcal.analysis
================

PLD (Privacy Loss Distribution)
---------------------------------

Functions for computing risk metrics from Privacy Loss Distributions.

.. autofunction:: riskcal.analysis.get_beta_from_pld

.. autofunction:: riskcal.analysis.get_advantage_from_pld

.. autofunction:: riskcal.analysis.get_bayes_risk_from_pld

GDP (Gaussian Differential Privacy)
-------------------------------------

Functions for computing risk metrics from Gaussian DP parameters.

.. autofunction:: riskcal.analysis.get_beta_from_gdp

.. autofunction:: riskcal.analysis.get_advantage_from_gdp

.. autofunction:: riskcal.analysis.get_bayes_risk_from_gdp

ADP (Approximate Differential Privacy)
---------------------------------------

Functions for computing risk metrics from (epsilon, delta)-DP parameters.

.. autofunction:: riskcal.analysis.get_beta_from_adp

.. autofunction:: riskcal.analysis.get_advantage_from_adp

.. autofunction:: riskcal.analysis.get_epsilon_from_err_rates

RDP (Renyi Differential Privacy)
----------------------------------

Functions for computing risk metrics from Renyi DP parameters.

.. autofunction:: riskcal.analysis.get_beta_from_rdp

zCDP (Zero-Concentrated Differential Privacy)
-----------------------------------------------

Functions for computing risk metrics from zCDP parameters.

.. autofunction:: riskcal.analysis.get_beta_from_zcdp

.. autofunction:: riskcal.analysis.get_advantage_from_zcdp

zCDP approximations
-------------------

Fast approximate conversions from zCDP to GDP-based risk metrics using fitted mappings.
These provide near-instant computation with <0.025 error for typical parameter ranges.

.. autofunction:: riskcal.analysis.get_mu_from_zcdp_approx

.. autofunction:: riskcal.analysis.get_beta_from_zcdp_approx

.. autofunction:: riskcal.analysis.get_advantage_from_zcdp_approx

Internal utilities
------------------

.. autofunction:: riskcal.analysis.pld_to_plrvs

.. autoclass:: riskcal.analysis.PLRVs
   :members:
   :show-inheritance:
