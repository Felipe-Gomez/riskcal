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

### What's this about?

#### Privacy risk and f-DP

Machine learning models, their outputs, synthetic datasets, releases of
statistics—all can leak information about the dataset on which they were trained
or fitted.  [Differential
privacy](https://en.wikipedia.org/wiki/Differential_privacy) (DP) is a theory of
ensuring privacy guarantees in these computations through the addition of
controlled random noise. Usually, in DP, the privacy guarantees are quantified
using a parameter called epsilon which roughly corresponds to the amount of
information the worst-case attacker can gain about any single record in
[nats](https://en.wikipedia.org/wiki/Nat_(unit)).  In practice, we might want to
quantify the privacy guarantees in terms of maximum allowable risk of standard
and practical attacks, such as singling out or attribute inference attacks.
f-Differential privacy (f-DP) is a notion of privacy which is immediately
interpretable in terms of such attack risks.

Concretely, f-DP parameterizes privacy using a trade-off curve between the error
rates (like the ROC curve, but flipped vertically) of the hypothetical
worst-case attacker. We refer to the false negative rate (FNR) as `beta`, and
false positive rate (FPR) as `alpha`. Thus, the trade-off curve is the set of
`alpha`, `beta` pairs which show the lowest possible error rates for any
attacker. We also use the notion of `advantage` which is the highest `1 - beta -
alpha`.  These error rates and advantage then can be immediately and directly
translated to a number of notions of privacy risk in terms of attack success:

- *Membership inference attacks.* These attacks directly correspond to the
  trade-off curve. The TPR of an attack at a given FPR is bounded by `1 - beta` at
  FPR `alpha`. The advantage of attacks, TPR - FPR, is bounded by `advantage`[^1].
- *Singling out.* The probability of singling out an individual is bounded
  by `1 - beta` for `alpha` which corresponds to the baseline probability of singling
  out prior to the adversary observing the release. We can also get the maximum
  probability of singling out across all baselines using `advantage`[^2].
- *Record reconstruction and attribute inference.* The probability of successful
  partial or full reconstruction of a single training data record, e.g., accuracy
  of guessing a set of attributes, is bounded by `1 - beta` for `alpha` which
  correspond to the baseline probability of such reconstruction in the data
  distribution. As before, we can also get the maximum probability of successful
  reconstruction out across all baselines using `advantage`[^2].

Most available tools for ensuring DP do not support analyses in terms of f-DP.
The goal of this library is to provide building blocks to analyze many standard
privacy-preserving mechanisms in terms of f-DP.


#### Methods
The library implements methods described by Kulynych & Gomez et al., 2024[^1].

- The direct method for computing the trade-off curve based on privacy loss
  random variables is described in Algorithm 1. 
- The mapping between f-DP and operational privacy risk, and the idea of direct
  noise calibration to risk instead of the standard calibration to a given
  (epsilon, delta) is described in Sections 2 and 3.

#### References

[^1]: [Attack-Aware Noise Calibration for Differential
Privacy](https://arxiv.org/abs/2407.02191). NeurIPS 2024.
[^2]: [Unifying Re-Identification, Attribute Inference, and Data Reconstruction
Risks in Differential Privacy](https://arxiv.org/abs/2507.06969), 2025.

If you make use of the library or methods, please cite:
```bibtex
@article{kulynych2024attack,
  title={Attack-aware noise calibration for differential privacy},
  author={Kulynych, Bogdan and Gomez, Juan F and Kaissis, Georgios and du Pin Calmon, Flavio and Troncoso, Carmela},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={134868--134901},
  year={2024}
}

@article{kulynych2025unifying,
  title={Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy},
  author = {Kulynych, Bogdan and Gomez, Juan Felipe and Kaissis, Georgios and Hayes, Jamie and Balle, Borja and du Pin Calmon, Flavio and Raisaro, Jean Louis}
  journal={arXiv preprint arXiv:2507.06969},
  year={2025}
}
```

### Using the Library

##### Install

Install with:
```
pip install riskcal
```

If you want to set it up locally for development, clone the repository and run:
```
uv sync --dev
```

#### Quickstart

##### Computing f-DP / Getting the Trade-Off Curve for a DP Mechanism

To measure the trade-off curve for DP-SGD, you can run
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


##### Computing Bayes risk for a mechanism
Bayes risk shows the maximum accuracy of an attack against privacy of a single
record under a binary prior, e.g., accuracy of attribute inference assuming that
a record has one of two possible attributes, or accuracy of a membership
inference attack assuming that a record has a certain prior probability of
membership. To get the Bayes risk for a given PLD object, you can use:

```python
import riskcal
import numpy as np

from dp_accounting.pld.privacy_loss_distribution import from_laplace_mechanism

pld = from_laplace_mechanism(1.0)

prior = np.array([0.5, 0.95, 0.99])
risk = riskcal.conversions.get_bayes_risk_from_pld(pld, prior=prior)
```

