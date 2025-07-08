## riskcal

[![CI](https://github.com/bogdan-kulynych/riskcal/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bogdan-kulynych/riskcal/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2407.02191-b31b1b.svg)](https://arxiv.org/abs/2407.02191)

---

⚠️  This is a research prototype. Avoid or be extra careful when using in production.

---

The library provides tools for computing the f-DP trade-off curves for common differentially private
algorithms, and calibrating their noise scale to notions of operational privacy risk (attack
accuracy/advantage, or attack TPR and FPR) instead of the (epsilon, delta) parameters.  The library
enables to reduce the noise scale at the same level of targeted attack risk.

### References
The library implements methods described in the [associated paper](https://arxiv.org/abs/2407.02191), published at NeurIPS 2024:

 - The direct method for computing the trade-off curve based on privacy loss random variables is described in Algorithm 1.
 - The mapping between f-DP and operational privacy risk, and the idea of direct noise calibration to risk
   instead of the standard calibration to a given (epsilon, delta) is described in Sections 2 and 3.

If you make use of the library or methods, please cite:
```
@article{kulynych2024attack,
  title={Attack-aware noise calibration for differential privacy},
  author={Kulynych, Bogdan and Gomez, Juan F and Kaissis, Georgios and du Pin Calmon, Flavio and Troncoso, Carmela},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={134868--134901},
  year={2024}
}
```

### Using the Library

Install with:
```
pip install riskcal
```

#### Quickstart

##### Computing f-DP / Getting the Trade-Off Curve for a DP Mechanism
To measure the attack trade-off curve (equivalent to attack's receiver-operating curve) for DP-SGD,
you can run
```python
import riskcal
import numpy as np

noise_multiplier = 0.5
sample_rate = 0.002
num_steps = 10000

alpha = np.array([0.01, 0.05, 0.1])
beta = riskcal.dpsgd.get_beta_for_dpsgd(
    alpha=alpha,
    noise_multiplier=noise_multiplier,
    sample_rate=sample_rate,
    num_steps=num_steps,
)
```

The library also provides an opacus-compatible account which uses the Connect the Dots accounting
from Google's DP accounting library, with extra methods to get the trade-off curve and advantage.
Thus, the above snippet is equivalent:

```python
import riskcal
import numpy as np

noise_multiplier = 0.5
sample_rate = 0.002
num_steps = 10000

acct = riskcal.dpsgd.CTDAccountant()
for _ in range(num_steps):
    acct.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

alpha = np.array([0.01, 0.05, 0.1])
beta  = acct.get_beta(alpha=alpha)
```

You can also get the trade-off curve for any DP mechanism
[supported](https://github.com/google/differential-privacy/tree/0b109e959470c43e9f177d5411603b70a56cdc7a/python/dp_accounting)
by Google's DP accounting library, given its privacy loss distribution (PLD) object:
```python
import riskcal
import numpy as np

from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism
from dp_accounting.pld.privacy_loss_distribution import from_laplace_mechanism

pld = from_gaussian_mechanism(1.0).compose(from_laplace_mechanism(0.1))

alpha = np.array([0.01, 0.05, 0.1])
beta = riskcal.conversions.get_beta_from_pld(pld, alpha=alpha)
```

##### Calibrating DP-SGD to attack FNR/FPR
To calibrate noise scale in DP-SGD to a given advantage, run:
```python
import riskcal

sample_rate = 0.002
num_steps = 10000

noise_multiplier = riskcal.dpsgd.find_noise_multiplier_for_advantage(
    advantage=0.1,
    sample_rate=sample_rate,
    num_steps=num_steps
)
```

To calibrate noise scale in DP-SGD to a given attack FPR (beta) and FNR (alpha), run:
```python
import riskcal

sample_rate = 0.002
num_steps = 10000

noise_multiplier = riskcal.dpsgd.find_noise_multiplier_for_err_rates(
    beta=0.2,
    alpha=0.01,
    sample_rate=sample_rate,
    num_steps=num_steps
)
```
