"""Connect the Dots (CTD) accountant for DP-SGD."""

from copy import deepcopy
from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism


class CTDAccountant:
    """
    Opacus-compatible Connect the Dots accountant.

    This accountant tracks privacy loss for DP-SGD using Google's Connect the Dots
    (CTD) method via the dp_accounting library. It maintains a history of
    (noise_multiplier, sample_rate, num_steps) tuples and composes them to
    compute overall privacy loss.

    Example:
        >>> acct = CTDAccountant()
        >>> for _ in range(1000):
        ...     acct.step(noise_multiplier=1.0, sample_rate=0.01)
        >>> epsilon = acct.get_epsilon(delta=1e-5)
        >>> print(f"Privacy: ({epsilon:.2f}, 1e-5)-DP")
    """

    def __init__(self):
        self.history = []

    def step(self, *, noise_multiplier, sample_rate):
        """
        Record a single DP-SGD step.

        Args:
            noise_multiplier: Noise scale for this step.
            sample_rate: Poisson sampling probability for this step.
        """
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
        """
        Get the composed privacy loss distribution.

        Args:
            grid_step: Discretization interval for PLD computation.
            use_connect_dots: Whether to use Connect the Dots composition.

        Returns:
            Composed PrivacyLossDistribution from dp_accounting.
        """
        noise_multiplier, sample_rate, num_steps = self.history[0]
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
        """
        Get epsilon for given delta.

        Args:
            delta: Target delta value.
            **kwargs: Additional arguments passed to get_pld (e.g., grid_step).

        Returns:
            Epsilon value for (epsilon, delta)-DP.
        """
        pld = self.get_pld(**kwargs)
        return pld.get_epsilon_for_delta(delta)

    def get_beta(self, *, alpha, **kwargs):
        """
        Get FNR (beta) for given FPR (alpha).

        Args:
            alpha: False positive rate.
            **kwargs: Additional arguments passed to get_pld (e.g., grid_step).

        Returns:
            False negative rate corresponding to alpha.
        """
        from riskcal.analysis import get_beta_from_pld

        pld = self.get_pld(**kwargs)
        return get_beta_from_pld(pld, alpha)

    def get_advantage(self, **kwargs):
        """
        Get attack advantage.

        Args:
            **kwargs: Additional arguments passed to get_pld (e.g., grid_step).

        Returns:
            Maximum attack advantage.
        """
        from riskcal.analysis import get_advantage_from_pld

        pld = self.get_pld(**kwargs)
        return get_advantage_from_pld(pld)

    def __len__(self):
        """Return total number of steps recorded."""
        total = 0
        for _, _, steps in self.history:
            total += steps
        return total

    @classmethod
    def mechanism(cls):
        """Return mechanism name for state dict compatibility."""
        return "ctd"

    # The following methods are copied from https://opacus.ai/api/_modules/opacus/accountants/accountant.html#IAccountant
    # to avoid the direct dependence on the opacus package.

    def get_optimizer_hook_fn(self, sample_rate: float):
        """
        Returns a callback function which can be attached to DPOptimizer.

        Args:
            sample_rate: Expected sampling rate used for accounting.

        Returns:
            Hook function for DPOptimizer.
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
            destination: A mappable object to populate the current state_dict into.
                If this arg is None, an OrderedDict is created and populated.
                Default: None.

        Returns:
            State dictionary.
        """
        if destination is None:
            destination = {}
        destination["history"] = deepcopy(self.history)
        destination["mechanism"] = self.mechanism()
        return destination

    def load_state_dict(self, state_dict):
        """
        Validates the supplied state_dict and populates the current
        Privacy Accountant's state dict.

        Args:
            state_dict: State dict to load.

        Raises:
            ValueError: If supplied state_dict is invalid and cannot be loaded.
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
        if self.mechanism() != state_dict["mechanism"]:
            raise ValueError(
                f"state_dict of {state_dict['mechanism']} cannot be loaded into "
                f" Privacy Accountant with mechanism {self.mechanism()}"
            )
        self.history = state_dict["history"]
