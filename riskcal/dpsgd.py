from copy import deepcopy
from functools import reduce
from typing import Union
from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism
from scipy.optimize import root_scalar
import numpy as np

from riskcal import conversions
from typing import Callable, Tuple


class CTDAccountant:
    """
    Opacus-compatible Connect the Dots accountant.
    """

    def __init__(self):
        self.history = []

    def step(self, *, noise_multiplier, sample_rate):
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.history.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, 1))

        else:
            self.history.append((noise_multiplier, sample_rate, 1))

    def get_pld(self, grid_step=1e-4, use_connect_dots=True):
        
        noise_multiplier, sample_rate, num_steps =  self.history[0]
        pld = from_gaussian_mechanism(
                standard_deviation=noise_multiplier,
                sampling_prob=sample_rate,
                use_connect_dots=use_connect_dots,
                value_discretization_interval=grid_step,
            ).self_compose(num_steps)
        
        
        for noise_multiplier, sample_rate, num_steps in self.history[1:]:
            pld_new = from_gaussian_mechanism(
                standard_deviation=noise_multiplier,
                sampling_prob=sample_rate,
                use_connect_dots=use_connect_dots,
                value_discretization_interval=grid_step,
            ).self_compose(num_steps)
            pld = pld.compose(pld_new)
            
        return pld

    def get_epsilon(self, *, delta, **kwargs):
        pld = self.get_pld(**kwargs)
        return pld.get_epsilon_for_delta(delta)

    def get_beta(self, *, alpha, **kwargs):
        pld = self.get_pld(**kwargs)
        return conversions.get_beta_from_pld(pld, alpha)

    def get_advantage(self, **kwargs):
        pld = self.get_pld(**kwargs)
        return conversions.get_advantage_from_pld(pld)

    def __len__(self):
        total = 0
        for _, _, steps in self.history:
            total += steps
        return total

    def mechanism(self):
        return "ctd"

    # The following methods are copied from https://opacus.ai/api/_modules/opacus/accountants/accountant.html#IAccountant
    # to avoid the direct dependence on the opacus package.
    def get_optimizer_hook_fn(self, sample_rate: float):
        """
        Returns a callback function which can be used to attach to DPOptimizer
        Args:
            sample_rate: Expected sampling rate used for accounting
        """

        def hook_fn(optim):
            # This works for Poisson for both single-node and distributed
            # The reason is that the sample rate is the same in both cases (but in
            # distributed mode, each node samples among a subset of the data)
            self.step(
                noise_multiplier=optim.noise_multiplier,
                sample_rate=sample_rate * optim.accumulated_iterations,
            )

        return hook_fn

    def state_dict(self, destination=None):
        """
        Returns a dictionary containing the state of the accountant.
        Args:
            destination: a mappable object to populate the current state_dict into.
                If this arg is None, an OrderedDict is created and populated.
                Default: None
        """
        if destination is None:
            destination = {}
        destination["history"] = deepcopy(self.history)
        destination["mechanism"] = self.__class__.mechanism
        return destination

    def load_state_dict(self, state_dict):
        """
        Validates the supplied state_dict and populates the current
        Privacy Accountant's state dict.

        Args:
            state_dict: state_dict to load.

        Raises:
            ValueError if supplied state_dict is invalid and cannot be loaded.
        """
        if state_dict is None or len(state_dict) == 0:
            raise ValueError(
                "state dict is either None or empty and hence cannot be loaded"
                " into Privacy Accountant."
            )
        if "history" not in state_dict.keys():
            raise ValueError(
                "state_dict does not have the key `history`."
                " Cannot be loaded into Privacy Accountant."
            )
        if "mechanism" not in state_dict.keys():
            raise ValueError(
                "state_dict does not have the key `mechanism`."
                " Cannot be loaded into Privacy Accountant."
            )
        if self.__class__.mechanism != state_dict["mechanism"]:
            raise ValueError(
                f"state_dict of {state_dict['mechanism']} cannot be loaded into "
                f" Privacy Accountant with mechanism {self.__class__.mechanism}"
            )
        self.history = state_dict["history"]


def get_advantage_for_dpsgd(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    grid_step=1e-4,
):
    pld = from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
        value_discretization_interval=grid_step,
    ).self_compose(num_steps)
    return conversions.get_advantage_from_pld(pld)


def get_beta_for_dpsgd(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    alpha: Union[float, np.ndarray],
    grid_step=1e-4,
):
    pld = from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
        value_discretization_interval=grid_step,
    ).self_compose(num_steps)
    return conversions.get_beta_from_pld(pld, alpha)

def inverse_monotone_function(
    f: Callable[[float], float],
    f_target: float,
    bounds: Tuple[float, float],
    func_threshold: float = np.inf,
    arg_threshold: float = np.inf,
    increasing: bool = False,
):
    """
    Finds the value of x such that the monotonic function f(x) 
    is approximately equal to f_target within a given threshold.

    :param f: A monotonic function (increasing or decreasing) to invert.
    :param f_target: The target value for f(x).
    :param bounds: A tuple (lower_x, upper_x) defining the search interval.
    :param func_threshold: Acceptable error for |f(x) - f_target|.
    :param arg_threshold: Acceptable error for |upper_x - lower_x|.
    :param increasing: Indicates if f is increasing (True) or decreasing (False).
    :return: The value of x that satisfies the threshold conditions.
    
    It is guaranteed that the returned x is within the thresholds of the
    smallest (for monotonically decreasing func) or the largest (for
    monotonically increasing func) such x. 
    """
    
    # Initialize bounds and midpoint 
    lower_x, upper_x = bounds
    mid_x = (upper_x + lower_x)/2
    f_mid = f(mid_x)
    
    # setup check function
    if increasing:
        check = lambda f_value, target_value: f_value <= target_value
        def continue_condition(upper_x, lower_x):
            return (upper_x - lower_x > arg_threshold) or (abs(f(lower_x) - f_target) > func_threshold)
    else:
        check = lambda f_value, target_value: f_value > target_value
        def continue_condition(upper_x, lower_x):
            return (upper_x - lower_x > arg_threshold) or (abs(f(upper_x) - f_target) > func_threshold)
    
    # run bisection
    while continue_condition(upper_x, lower_x):

        mid_x = (upper_x + lower_x) / 2
        f_mid = f(mid_x)
        if check(f_mid, f_target):
            lower_x = mid_x
        else:
            upper_x = mid_x
    
    if increasing:
        return lower_x
    else:
        return upper_x

def find_noise_multiplier_for_err_rates(
    alpha: float,
    beta: float,
    sample_rate: float,
    num_steps: float,
    grid_step: float = 1e-4,
    mu_min: float = 0,
    mu_max: float = 100.0,
    beta_error: float = 0.001,
    mu_error: float = 0.1
):
    """
    Find a noise multiplier that satisfies a given (FPR, FNR) bound
    Adapted from https://github.com/microsoft/prv_accountant/blob/main/prv_accountant/dpsgd.py

    :param alpha: Attack FPR bound
    :param beta: Attack FNR bound
    :param sample_rate: Probability of a record being in batch for Poisson sampling
    :param num_steps: Number of optimisation steps
    :param grid_step: Discretization grid step
    :param float mu_min: Minimum value of noise multiplier of the search.
    :param float mu_max: Maximum value of noise multiplier of the binary search.
    :param beta_error: numeric threshold for convergence in beta / FNR.
    :param mu_error: numeric threshold for convergence in mu.
    """
    
    def _get_beta(mu):
        return get_beta_for_dpsgd(
            noise_multiplier=mu,
            sample_rate=sample_rate,
            num_steps=num_steps,
            alpha=alpha,
            grid_step=grid_step,
        )
    bounds = [mu_min, mu_max]
    mu = inverse_monotone_function(f = _get_beta, 
                                   f_target = beta, 
                                   bounds = bounds, 
                                   func_threshold = beta_error,
                                   arg_threshold = mu_error,
                                   increasing = True)
    
    return mu

def find_noise_multiplier_for_advantage(
    advantage: float,
    sample_rate: float,
    num_steps: float,
    grid_step: float = 1e-4,
    advantage_error: float = 0.001,
    mu_error: float = 0.1,
    mu_min: float = 0, 
    mu_max: float = 100.0
):
    """
    Find a noise multiplier that satisfies a given target advantage.
    Adapted from https://github.com/microsoft/prv_accountant/blob/main/prv_accountant/dpsgd.py

    :param advantage: Attack advantage
    :param sample_rate: Probability of a record being in batch for Poisson sampling
    :param num_steps: Number of optimisation steps
    :param grid_step: Discretization grid step
    :param advantage_error: numeric threshold for convergence in advantage
    :param mu_error: numeric threshold for convergence in mu.
    :param float mu_min: Minimum value of noise multiplier of the search.
    :param float mu_max: Maximum value of noise multiplier of the search.
    """

    def _get_advantage(mu):
        pld = from_gaussian_mechanism(
            standard_deviation = mu,
            value_discretization_interval= grid_step,
            sampling_prob=sample_rate).self_compose(num_steps)
        return pld.get_delta_for_epsilon(0)
    
    bounds = [mu_min, mu_max]
    mu = inverse_monotone_function(f = _get_advantage, 
                                   f_target = advantage, 
                                   bounds = bounds, 
                                   func_threshold = advantage_error,
                                   arg_threshold = mu_error,
                                   increasing = False)
    
    return mu