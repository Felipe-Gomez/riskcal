riskcal.calibration
===================

Core calibration interface
--------------------------

Generic calibration framework.

.. autofunction:: riskcal.calibration.calibrate_parameter

.. autoclass:: riskcal.calibration.CalibrationTarget
   :members:
   :show-inheritance:

.. autoclass:: riskcal.calibration.CalibrationConfig
   :members:
   :show-inheritance:

.. autoclass:: riskcal.calibration.CalibrationResult
   :members:
   :show-inheritance:

.. autoclass:: riskcal.calibration.PrivacyMetrics
   :members:
   :show-inheritance:

.. autoclass:: riskcal.calibration.PrivacyEvaluator
   :members:
   :show-inheritance:

DP-SGD calibration
------------------

Specialized functions for calibrating DP-SGD noise multipliers.

.. autofunction:: riskcal.calibration.find_noise_multiplier_for_advantage_dpsgd

.. autofunction:: riskcal.calibration.find_noise_multiplier_for_err_rates_dpsgd

.. autofunction:: riskcal.calibration.get_advantage_for_dpsgd

.. autofunction:: riskcal.calibration.get_beta_for_dpsgd

.. autofunction:: riskcal.calibration.create_dpsgd_evaluator

.. autofunction:: riskcal.calibration.create_dpsgd_epsilon_evaluator

Blackbox calibration
--------------------

Calibration using privacy profiles.

.. autofunction:: riskcal.calibration.find_noise_multiplier_for_epsilon_delta

.. autofunction:: riskcal.calibration.find_noise_multiplier_for_advantage_blackbox

.. autofunction:: riskcal.calibration.find_noise_multiplier_for_err_rates_blackbox

.. autofunction:: riskcal.calibration.create_accountant_evaluator
