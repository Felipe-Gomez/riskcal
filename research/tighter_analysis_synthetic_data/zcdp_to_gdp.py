# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import riskcal
from tqdm import auto as tqdm
import seaborn as sns

from matplotlib import pyplot as plt

# +
eps_vals = np.linspace(0.1, 4, 20)
beta_vals = []
alpha = np.linspace(0, 1, 100)

for eps in tqdm.tqdm(eps_vals):
    rho = 1/8 * eps**2
    beta = riskcal.conversions.get_beta_for_rho(rho, alpha=alpha)
    beta_vals.append(beta)


# -

def find_mu(alpha, beta, tol=1e-4):
    mu_min = 1e-3
    mu_max = 100
    
    mu_left = mu_min
    mu_right = mu_max

    while np.abs(mu_left - mu_right) >= tol:
        mu_mid = 0.5 * (mu_right + mu_left)
        beta_gdp = riskcal.conversions.get_beta_for_mu(mu=mu_mid, alpha=alpha)
        if np.all(beta_gdp <= beta):
            mu_right = mu_mid
        else:
            mu_left = mu_mid

    return mu_right        


mus = []
for beta in beta_vals:
    mu = find_mu(alpha, beta)
    mus.append(mu)

np.sqrt(2)

# +
plt.plot(eps_vals, mus, label="observed")
plt.plot(eps_vals, 2/3 * eps_vals, label="model (2/3 eps)")
plt.plot(eps_vals, eps_vals / np.sqrt(2), label="model (eps / np.sqrt(2))")
plt.legend()

plt.ylabel("mu")
plt.xlabel("epsilon")
plt.xscale("log")
plt.yscale("log")

# +
fig, axes = plt.subplots(nrows=len(eps_vals), ncols=1, figsize=(5, 5 * len(eps_vals)))

for i, (eps, mu, beta) in enumerate(zip(eps_vals, mus, beta_vals)):
    axes[i].plot(alpha, beta, label="zCDP")
    axes[i].plot(alpha, riskcal.conversions.get_beta_for_mu(mu, alpha=alpha), label="fitted GDP")
    axes[i].set_title(f"epsilon = {eps:.4f}")

plt.legend()
plt.savefig("zcdp_to_gdp.png")
