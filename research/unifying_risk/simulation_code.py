# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="tASzMFOFMQpa" jp-MarkdownHeadingCollapsed=true
# ## Colab Dependencies
# Only if running in Google Colab.

# %% colab={"base_uri": "https://localhost:8080/"} id="5tAlGSNPkuz7" outputId="3675d7f3-08f0-4c6e-a27c-80ec27c968e7"
# # !pip install --no-deps dp_accounting opacus

# %% id="PlD7q_RUmuqZ"
# # !pip install latex
# import latex
# # !apt-get install -y texlive-latex-base dvipng cm-super

# %% id="t75nCt_-pvZ8"
# # !apt-get install -y texlive-fonts-recommended texlive-latex-recommended

# %% colab={"base_uri": "https://localhost:8080/"} id="_J_KnOkPqS2L" jupyter={"outputs_hidden": true} outputId="0a36767a-bffa-4bc4-b67c-7fc249ed2d21"
# # ! sudo apt-get install texlive-latex-recommended
# # ! sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended
# # ! wget http://mirrors.ctan.org/macros/latex/contrib/type1cm.zip
# # ! unzip type1cm.zip -d /tmp/type1cm
# # ! cd /tmp/type1cm/type1cm/ && sudo latex type1cm.ins
# # ! sudo mkdir /usr/share/texmf/tex/latex/type1cm
# # ! sudo cp /tmp/type1cm/type1cm/type1cm.sty /usr/share/texmf/tex/latex/type1cm
# # ! sudo texhash
# # !apt install cm-super

# %% colab={"base_uri": "https://localhost:8080/"} id="MGzugyQfkxTS" outputId="b221d8ff-c314-47d8-a494-128ff9d41b68"
# # ! git clone https://github.com/Felipe-Gomez/riskcal; cd riskcal; pip install --no-deps .

# %%
# # !mkdir -p images

# %% [markdown] id="gCUUZKhiMaft"
# ## Dependencies

# %% id="344b5a42-cfca-437c-9694-b5226069158a"
import itertools
import dataclasses

import riskcal
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm.auto import tqdm
from scipy.special import erf
from scipy.stats import entropy, norm, multivariate_normal
from scipy import optimize
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

from dp_accounting.pld import privacy_loss_distribution
from opacus.accountants.analysis.rdp import compute_rdp

# %% [markdown] id="8ff01e14-ee71-48a0-9655-e3673d10ff6b" jp-MarkdownHeadingCollapsed=true
# ## Utils
#
# Note: use pdflatex to make this code compatible with Google Colab.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Plotting

# %% id="867f7256-96ca-462b-9770-454e32a523fd"
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt

from matplotlib.backends.backend_pgf import FigureCanvasPgf

mpl.backend_bases.register_backend("pdf", FigureCanvasPgf)

sns.set(
    style="whitegrid",
    context="paper",
    font_scale=2,
    rc={"lines.linewidth": 2.5, "lines.markersize": 6, "lines.markeredgewidth": 0.0},
)
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",     # Use pdflatex instead of xelatex
        "font.family": "sans-serif",  # use serif/main font for text elements
        "font.serif": "Helvetica",
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    }
)


# %% colab={"base_uri": "https://localhost:8080/"} id="ca085c4b-5eae-4dbe-82ba-173d83de9a1c" outputId="cae393e2-c084-4e00-f289-bab45d09b798"
def ensure_target_index(vals, target, target_index):
    counter = 0
    result = []
    for i, val in enumerate(vals):
        if counter == target_index:
            counter += 1
        if val == target:
            result.append(target_index)
        else:
            result.append(counter)
            counter += 1
    return result

ensure_target_index(["a", "b", "c", "d", "e"], target="b", target_index=2)

# %% id="92b49279-359a-486a-9e7a-6caafcdb6f08"
markers = ["D", "o", "^", "v", "s", "P", "X"]
colors = sns.color_palette()

def ensure_marker_order(vals, target, target_marker_index=0):
    return [markers[i] for i in ensure_target_index(vals, target, target_index=target_marker_index)]

def ensure_color_order(vals, target, target_color_index=1):
    return [colors[i] for i in ensure_target_index(vals, target, target_index=target_color_index)]


# %% colab={"base_uri": "https://localhost:8080/", "height": 76} id="349b2120-79ee-452a-82dd-a3a6a9bd1eb0" outputId="0eb25af3-82bc-4856-9348-7b6d0c500f8f"
sns.color_palette()

# %%
aspect_ratio = 1.05


# %% [markdown] id="556eea02-832b-4d9d-ae11-eeb8ce5b9c82" jp-MarkdownHeadingCollapsed=true
# ### Renyi-based bounds

# %% id="1fb74676-bced-45a1-a253-f3cf7d9e8673"
def renyi_f_bound(alpha, epsilon, order):
    return 1 - (alpha * np.exp(epsilon))**((order-1)/order)

def get_beta_for_rho(rho, alpha):
    squeeze_return = False
    if not hasattr(alpha, "__len__"):
        squeeze_return = True
        alpha = np.array([alpha])

    log_one_over_alpha = np.log(1/alpha)
    fudge = 1e-10

    # if rho is too large, set order = 1 + fudge
    log_gamma = np.where(rho >= log_one_over_alpha, fudge / (1+fudge) * (log_one_over_alpha + rho + rho * fudge),
                         -(np.sqrt(log_one_over_alpha) - np.sqrt(rho))**2)

    output = -np.expm1(log_gamma) # 1 - exp(log_gamma)
    return output if not squeeze_return else output[0]

def get_beta_for_rdp(rdp, orders, alpha):
    squeeze_return = False
    if not hasattr(alpha, "__len__"):
        squeeze_return = True
        alpha = np.array([alpha])

    betas = []
    for alpha_val in alpha:
        betas.append(np.max(renyi_f_bound(alpha_val, rdp, orders)))
    return np.array(betas) if not squeeze_return else betas[0]


# %% [markdown] id="00b2a69d-f453-42be-9f15-23901ff59c62" jp-MarkdownHeadingCollapsed=true
# ## Bounds Comparison: Singling out

# %% [markdown] id="m2B4Bw2Fy5Ss" jp-MarkdownHeadingCollapsed=true
# ### Laplace

# %% id="85d60ba5-aeeb-462b-bca0-c7e5e129df6f"
noise_params = np.geomspace(0.1333, 8.0, 15)
sens = 1.0
delta = 0.
data_sizes = [500, 1000, 5000]
max_data_size = max(data_sizes)

data_chunks = []
for data_size, noise_param in itertools.product(data_sizes, noise_params):
    pld = privacy_loss_distribution.from_laplace_mechanism(noise_param, sensitivity=2 * sens)  # 2*sens for RO adjacency.
    epsilon = pld.get_epsilon_for_delta(delta)
    w = 1/max_data_size
    baseline_pso = data_size * w * (1 - w)**(data_size - 1)
    baseline_spso = w

    cohen_bound = np.exp(epsilon) * baseline_pso / (1 - w)**(data_size - 1) + data_size * delta - baseline_pso
    single_f = lambda alpha: riskcal.conversions.get_beta_from_pld(pld, alphas=alpha)
    pso_bound = data_size * (1 - single_f(w)) - baseline_pso
    spso_bound = 1 - single_f(baseline_spso) - baseline_spso

    data_chunks.append(dict(adv=min(cohen_bound,1), kind="cohen", noise=noise_param, eps=epsilon, n=data_size))
    data_chunks.append(dict(adv=min(pso_bound,1), kind="pso_f", noise=noise_param, eps=epsilon, n=data_size))
    data_chunks.append(dict(adv=min(spso_bound,1), kind="spso_f", noise=noise_param, eps=epsilon, n=data_size))

# %% id="c9158c96-4ae9-4bb4-8a98-605ec4759c12"
renaming_dict = {
    "cohen": r"PSO (Cohen and Nissim, via $(\varepsilon, \delta)$)",
    "pso_f": "PSO (via $f$-DP)",
    "spso_f": "Unifying (via $f$-DP)",

    "kind": "Method",
    "succ": "Success",
    "adv": "Risk (advantage)",
    "eps": r"$\varepsilon$",
    "mu": r"$\mu$",
    "n": "$n$"
}

# %% colab={"base_uri": "https://localhost:8080/", "height": 320} id="51a51e9f-feb3-4367-a4f5-682c3810d7ca" outputId="3a40820b-1912-4798-f73b-a4b60af0454b"
plot_data = pd.DataFrame(data_chunks).query("kind != 'pso_f'")
print(plot_data.query("kind == 'spso_f'").adv.max())

order = [renaming_dict[k] for k in ["cohen", "spso_f"]]
target = renaming_dict["spso_f"]

g = sns.relplot(
    data=plot_data.replace(renaming_dict).rename(columns=renaming_dict),
    x=renaming_dict["eps"],
    y=renaming_dict["adv"],
    hue=renaming_dict["kind"],
    style=renaming_dict["kind"],
    col=renaming_dict["n"],
    facet_kws={'sharey': False, 'sharex': True},
    hue_order=order,
    palette=ensure_color_order(order, target),
    markers=ensure_marker_order(order, target),
    dashes=False,
    kind="line",
    markersize=8,
)

for ax in g.axes[0]:
    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width
g.set(xlim=(0, 15))
g.set(ylim=(0, 1.05));

plt.savefig("images/laplace_pso_no_f.pdf", bbox_inches="tight", dpi=300)

# %% colab={"base_uri": "https://localhost:8080/", "height": 320} id="hIgfOOkFxkPM" outputId="22f1f17c-6309-4c8e-cfe0-feea19a42768"
plot_data = pd.DataFrame(data_chunks)
print(plot_data.query("kind == 'spso_f'").adv.max())

order = [renaming_dict[k] for k in ["cohen", "pso_f", "spso_f"]]
target = renaming_dict["spso_f"]

g = sns.relplot(
    data=plot_data.replace(renaming_dict).rename(columns=renaming_dict),
    x=renaming_dict["eps"],
    y=renaming_dict["adv"],
    hue=renaming_dict["kind"],
    style=renaming_dict["kind"],
    col=renaming_dict["n"],
    facet_kws={'sharey': False, 'sharex': True},
    hue_order=order,
    palette=ensure_color_order(order, target),
    markers=ensure_marker_order(order, target),
    dashes=False,
    kind="line",
    markersize=8,
)

for ax in g.axes[0]:
    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width
g.set(xlim=(0, 15))
g.set(ylim=(0, 1.05));

plt.savefig("images/laplace_pso_f.pdf", bbox_inches="tight", dpi=300)

# %% [markdown] id="oA9N2l8Vy8qS" jp-MarkdownHeadingCollapsed=true
# ### Gaussian

# %% id="04c97c55-e25a-48dd-af1d-bd5af04bfb9b"
noise_params = np.geomspace(0.3, 10.0, 15)
sens = 1.0
delta = 1e-5
data_sizes = [500, 1000, 5000]
max_data_size = max(data_sizes)

data_chunks = []
for data_size, noise_param in itertools.product(data_sizes, noise_params):
    mu = sens / noise_param
    pld = privacy_loss_distribution.from_gaussian_mechanism(standard_deviation=noise_param, sensitivity=2 * sens)  # 2*sens for RO adjacency.
    epsilon = pld.get_epsilon_for_delta(delta)
    w = 1/max_data_size

    baseline_pso = data_size * w * (1 - w)**(data_size - 1)
    baseline_spso = w
    cohen_bound = np.exp(epsilon) * baseline_pso / (1 - w)**(data_size - 1) + data_size * delta - baseline_pso
    single_f = lambda alpha: riskcal.conversions.get_beta_from_pld(pld, alphas=alpha)

    pso_bound = data_size * (1 - single_f(w)) - baseline_pso
    spso_bound = 1 - single_f(baseline_spso) - baseline_spso

    data_chunks.append(dict(adv=min(cohen_bound,1), kind="cohen", mu=sens / noise_param, eps=epsilon, n=data_size))
    data_chunks.append(dict(adv=min(pso_bound,1), kind="pso_f", noise=noise_param, eps=epsilon, n=data_size))
    data_chunks.append(dict(adv=spso_bound, kind="spso_f", mu=sens / noise_param, eps=epsilon, n=data_size))

# %% colab={"base_uri": "https://localhost:8080/", "height": 320} id="135d0aab-8a61-4be3-b92c-a024b62bc519" outputId="b2216138-8814-4847-906a-e5948f1c2c45"
plot_data = pd.DataFrame(data_chunks).query("kind != 'pso_f'")

print(plot_data.query("kind == 'spso_f'").adv.max())
order = [renaming_dict[k] for k in ["cohen", "spso_f"]]
target = renaming_dict["spso_f"]

g = sns.relplot(
    data=plot_data.replace(renaming_dict).rename(columns=renaming_dict),
    x=renaming_dict["eps"],
    y=renaming_dict["adv"],
    hue=renaming_dict["kind"],
    style=renaming_dict["kind"],
    col=renaming_dict["n"],
    hue_order=order,
    palette=ensure_color_order(order, target),
    markers=ensure_marker_order(order, target),
    dashes=False,
    facet_kws={'sharey': False, 'sharex': True},
    kind="line",
    markersize=8,
)

for ax in g.axes[0]:
    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width
g.set(xlim=(0, 50))
g.set(ylim=(0, 1.05));

plt.savefig("images/gaussian_pso_no_f.pdf", bbox_inches="tight", dpi=300)

# %% colab={"base_uri": "https://localhost:8080/", "height": 320} id="m4CBZXbRzmhg" outputId="14189b04-2be9-4f65-8798-8e5957eb14c1"
plot_data = pd.DataFrame(data_chunks)

print(plot_data.query("kind == 'spso_f'").adv.max())
order = [renaming_dict[k] for k in ["cohen", "pso_f", "spso_f"]]
target = renaming_dict["spso_f"]

g = sns.relplot(
    data=plot_data.replace(renaming_dict).rename(columns=renaming_dict),
    x=renaming_dict["eps"],
    y=renaming_dict["adv"],
    hue=renaming_dict["kind"],
    style=renaming_dict["kind"],
    col=renaming_dict["n"],
    hue_order=order,
    palette=ensure_color_order(order, target),
    markers=ensure_marker_order(order, target),
    dashes=False,
    facet_kws={'sharey': False, 'sharex': True},
    kind="line",
    markersize=8,
)

for ax in g.axes[0]:
    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width
g.set(xlim=(0, 50))
g.set(ylim=(0, 1.05));

plt.savefig("images/gaussian_pso_f.pdf", bbox_inches="tight", dpi=300)

# %% [markdown] id="fb17cd5b-9382-453d-9785-dd0a7dadb942" jp-MarkdownHeadingCollapsed=true
# ## Bounds Comparison: ReRo

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["c37cf1f533dc4bde9b8f4d7979d01c27", "58794eddc2ee4996bef7c4b8a24f881f", "266e3157c3ea4939809953a43b36147c", "0fd03490b8b749c587ba989375f16759", "48592b8d23e54a3e87e5bc86b1277634", "f3f80aaa20e04f8b975c0d77812aee3f", "56664092cbbe44feb16a93e9d2103054", "47705fdddbeb4993ba9bcb5af30abd12", "d3258f90f02f47159b3e5b2d1b0794cd", "1ff7b105d0fe429c99d58ff75b7bd123", "c325b39ae328441e9d4ecc552e23d97c"]} id="10864840-0de1-40b9-bb58-3d0e25ac94d9" outputId="11d2026b-0ac2-485d-b442-2cc69484a590"
delta = 1e-5
special_baselines = [0.01, 0.1, "Worst-case"]
# special_sigmas = [0.5, 1.0, 2.0]
# sigmas = list(np.geomspace(0.15, 5.0, 8)) + special_sigmas
# baselines = list(np.linspace(0, 1, 10)) + special_baselines
sigmas = np.geomspace(0.15, 5.0, 15)
baselines = special_baselines
sens = 1

data_chunks = []
for sigma, baseline in tqdm(list(itertools.product(sigmas, baselines))):
    rho = sens**2 / (2 * sigma**2)
    mu = sens / sigma
    pld = privacy_loss_distribution.from_gaussian_mechanism(standard_deviation=sigma, sensitivity=sens)
    epsilon = pld.get_epsilon_for_delta(delta)
    alpha = baseline

    # For the worst-case baseline, we cut a corner here and find it only for the unifying bound,
    # because it's computationally efficient, and evaluate the rest at this baseline.
    if baseline == "Worst-case":
        eta = riskcal.get_advantage_from_pld(pld)
        alpha = 0.5 - 0.5 * eta

    # zCDP-based bound from Balle et al.
    data_chunks.append(dict(
        adv=min(1 - get_beta_for_rho(rho=rho, alpha=alpha) - alpha, 1),
        baseline=baseline,
        method="zcdp",
        mu=mu,
        sigma=sigma,
        epsilon=epsilon,
    ))

    # Balle et al. Renyi DP bound for a fixed order.
    data_chunks.append(dict(
        adv=min(1 - renyi_f_bound(alpha=alpha, epsilon=2 * rho, order=2) - alpha, 1),
        baseline=baseline,
        method="renyi",
        mu=mu,
        sigma=sigma,
        epsilon=epsilon,
    ))

    # Balle et al. Renyi DP bound, optimized over orders.
    # data_chunks.append(dict(
    #     adv=1 - get_beta_for_rho(rho=rho, alpha=alpha) - alpha,
    #     baseline=baseline,
    #     method="renyi_opt",
    #     mu=mu,
    #     sigma=sigma,
    #     epsilon=epsilon,
    # ))

    # Our unifying bound.
    data_chunks.append(dict(
        adv=min(1 - riskcal.get_beta_from_pld(pld, alphas=alpha) - alpha,1),
        baseline=baseline,
        method="unified",
        mu=mu,
        sigma=sigma,
        epsilon=epsilon,
    ))

# %% id="7bbc8aef-6533-4499-ae8f-609a21946e0c"
renaming_dict = {
    "renyi_opt": r"SRR (Balle et al., via RDP, optimized)",
    "renyi": r"SRR (Balle et al., via RDP, fixed ord.)",
    "zcdp": r"SRR (Balle et al., via zCDP)",
    "cherubin": r"SRR (Cherubin et al.)",
    "unified": r"Unifying (via $f$-DP)",
    "bayes": r"SAI (via $f$-DP)",
    "guo_mi": r"SAI (Guo et al., via Fano, MI)",
    "guo_mc": r"SAI (Guo et al., via Fano, MC)",

    "method": "Method",
    "epsilon": r"$\varepsilon$",
    "mu": r"$\mu$",
    "k": r"$k$",
    "adv": "Risk (advantage)",
    "succ": "Success",
    "baseline": "Baseline",
}

# %% colab={"base_uri": "https://localhost:8080/", "height": 299} id="uL85T7RP7EZ1" outputId="9a7bcd76-8020-4eef-948a-842753975c60"
order = [renaming_dict[k] for k in ["renyi", "zcdp", "unified"]]
target = renaming_dict["unified"]

g = sns.relplot(
    data=(
        pd.DataFrame(data_chunks)
        .query("method != 'renyi_opt' or baseline == 'Worst-case'")
        .replace(renaming_dict)
        .rename(columns=renaming_dict)
    ),
    y=renaming_dict["adv"],
    x=renaming_dict["epsilon"],
    hue=renaming_dict["method"],
    style=renaming_dict["method"],
    col=renaming_dict["baseline"],

    hue_order=order,
    palette=ensure_color_order(order, target),
    markers=['^', 'o', 'D'],

    # facet_kws={'sharey': False, 'sharex': True},
    dashes=False,
    kind="line",
    markersize=8,
)

for ax in g.axes[0]:
    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width
g.set(xlim=(0, 36))
g.set(ylim=(0, 1.05));

plt.savefig("images/gaussian_rero_epsilon_vs_risk.pdf", bbox_inches="tight", dpi=300)


# %% [markdown] id="59806fd7-9eb6-4aa6-a542-cfbe899c7f05" jp-MarkdownHeadingCollapsed=true
# ## Bounds Comparison: Attribute Inference

# %% [markdown] id="490d6d5e-3374-4411-9043-14eb498e75bf" jp-MarkdownHeadingCollapsed=true
# ### Reimplementation of Guo et al. bound on attribute inference for Gaussian mechanism

# %% id="2f6a0b45-b944-4e2d-879c-caea575d01e0"
def get_gmm_logp(X, multi_covar, one_hot_targets):
    N = len(X)
    K = one_hot_targets.shape[1]
    totals = np.zeros((N, 1), dtype=np.float64)
    gamma_nk = np.zeros((N, K), dtype=np.float64)

    for k in range(K):
        pi_k = 1/K
        mu_k = one_hot_targets[k]
        cov_k = multi_covar

        gamma_nk[:, k] = (pi_k * multivariate_normal.pdf(X, mean=mu_k, cov=cov_k)).ravel()

    totals = np.log(np.sum(gamma_nk, 1))
    return totals

def gmm_kl(nm, one_hot_targets, n_samples=100_000):
    # KL(p||q) = E_p[ log(p(x) / q(x))]

    rand_idx = np.random.randint(0, one_hot_targets.shape[1])
    target = one_hot_targets[rand_idx]
    multi_normal_mean = target
    multi_covar = (nm**2.) * one_hot_targets

    X = multivariate_normal.rvs(mean=target, cov=multi_covar, size=n_samples)
    log_p_X = multivariate_normal.logpdf(X, mean=target, cov=multi_covar)
    log_q_X = get_gmm_logp(X, multi_covar, one_hot_targets)

    return log_p_X.mean() - log_q_X.mean()

def mutual_info(p,  nm, n_samples=100_000):
    M = len(p)
    targets = np.arange(M)
    one_hot_targets = np.eye(M)[targets]

    kl = gmm_kl(nm, one_hot_targets, n_samples=n_samples)
    return kl

def get_guo_bound(nm, p_max, p, sens=1, approx=False, n_samples=100_000):
  """
  Guo et al. upper bound on advantage. Algorithm 1 in
  https://arxiv.org/pdf/2210.13662.pdf and using Eq 12 to bound mutual info
  of the Gaussian mechanism

  nm: Sigma
  p_max: Maximum sampling probability from prior
  p: Sampling probabilities from prior
  sens: Sensitivity
  """
  tol = 1e-10
  assert np.abs(np.sum(p) - 1) < tol
  if np.abs(p_max - np.max(p)) >= tol:
    raise ValueError("Incompatible p_max")

  r = np.exp(-sens**2/((2 * nm**2.)))
  if approx:
    mutual_info_ub = mutual_info(p,  nm, n_samples=n_samples)
    # print('approx', nm, mutual_info_ub)
    # print()
  else:
    mutual_info_ub = -np.sum([p_i * np.log2(p_i + ((1 - p_i)*r)) for p_i in p])
    # print('ub', nm, mutual_info_ub)
  h_p = entropy(p) # Entropy of prior
  M = len(p) # Size of prior
  # If entropy is smaller than epsilon return 1a.
  if mutual_info_ub > h_p:
    return 1.
  else:
    # NB: t is defined as the probability of error in their paper.
    fn_t = lambda t: h_p - mutual_info_ub + (t * np.log2(t)) + ((1-t) * np.log2(1-t)) - (t * np.log2(M-1))
    # assert fn_t(0.01) > 0, fn_t(0.01)
    assert fn_t(1-(1/M)) <= 0, fn_t(1-(1/M))

    # Space of t that we search over.
    t_search = np.linspace(0.0001, 1-(1/M), num=100_000)

    # Evaluate fn_t for each t_search. Maybe inefficient but should be ok for
    # t_search sizes < 1M.
    all_fn_ts = np.array([fn_t(t) for t in t_search])

    # Find index where fn_t is first negative.
    min_idx = np.where(all_fn_ts < 0)[0][0]

    # Get the corresponding t.
    min_t = t_search[min_idx]

    # advantage_bound = (1 - min_t - p_max)/(1-p_max)
    # return advantage_bound
    return min_t


# Convert advantage to success probability.
get_success_prob = lambda adv, baseline: adv * (1 - baseline) + baseline


# %% [markdown] id="ec1ad873-ebe9-4b97-9d21-ae81263a823a" jp-MarkdownHeadingCollapsed=true
# ### Closed-form solution for prior-aware Bayes error of Gaussian mechanism

# %% id="437bf0f4-9321-42e3-85ab-f85db4c062d8"
def rmin_gauss(pi, mu):
	assert mu>=0, "mu must be >= 0"
	pi = np.array([1-pi, pi])
	result = np.zeros_like(pi, dtype=float)
	# Set result to 0 where pi is 1
	result[pi == 1] = 0
	# Where pi is not 0 or 1, perform the original computation
	mask = (pi != 0) & (pi != 1)
	A = ((-mu**2/2) - np.log(pi[mask]) - np.log(-1/(pi[mask]-1)))/mu
	B = ((mu**2/2) - np.log(pi[mask]) - np.log(-1/(pi[mask]-1)))/mu
	#1-CDF is SF
	result[mask] = pi[mask]*norm.cdf(A)+(1-pi[mask])*(norm.sf(B))
	return result[1]

# %% [markdown] id="3e769089-f06d-41a2-94c7-c60428eb6336" jp-MarkdownHeadingCollapsed=true
# ### Experiments

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["b0fdbbcb0d974e2aba4e433e8b35bd4c", "9fa01f8c7203453d82cde337a16e3156", "1d38eb4b30ed4ed7a18783e712d06394", "2647333f6f694396b11bd75a4ce1daea", "ae1996ae502b4875a8608c4779e7e010", "38dd3a41b91a4a6d8aad50bce7e5c972", "944606df4b014609874c123239b1969c", "a2ceaa335e21442aac4e70c3b6167cc3", "03aae53afd154f8a9473cab71bb599fc", "76a7d6bb8fda41539ceb4ff217d2c1bf", "3f9e17db1af7456d8dc103abb2742515"]} id="9c2cdfed-2087-4def-81cd-2e09b1cdfafe" outputId="27f9808e-7469-4124-ccb1-709ffdbf9625"
delta = 1e-5
num_mc_reps = 10
num_mc_samples = 10000
ks = [2]
# special_baselines = [0.1, 0.25, 0.5]
special_sigmas = [3.0, 4.0, 5.0]
# sigmas = list(np.geomspace(1.0, 8.0, 3)) + special_sigmas
sigmas = special_sigmas
# baselines = list(np.linspace(0, 1, 10)) + special_baselines
baselines = np.linspace(0.5, 1, 10)

# Replace-one relation.
sens = 2

data_chunks = []
for k, sigma, baseline in tqdm(list(itertools.product(ks, sigmas, baselines))):
    rho = sens**2 / (2 * sigma**2)
    mu = sens / sigma
    pld = privacy_loss_distribution.from_gaussian_mechanism(standard_deviation=sigma, sensitivity=sens)
    epsilon = pld.get_epsilon_for_delta(delta)
    alpha = baseline

    if baseline == "Uniform":
        alpha = 1/k

    is_valid_ai_baseline = alpha >= 1/k

    # Guo et al. bounds on AI.
    if is_valid_ai_baseline:
        prior_dist = np.array([alpha] + [(1 - alpha)/(k-1)] * (k - 1))
        p_max = max(prior_dist)
        err_prob = get_guo_bound(nm=sigma, sens=sens, p_max=alpha, p=prior_dist)
        data_chunks.append(dict(
            adv=min(1 - err_prob - alpha,1),
            baseline=baseline,
            method="guo_mi",
            k=k,
            sigma=sigma,
            epsilon=epsilon,
            is_valid_ai_baseline=is_valid_ai_baseline,
        ))

        for i in range(num_mc_reps):
            err_prob = get_guo_bound(nm=sigma, sens=sens, p_max=alpha, p=prior_dist,
                                     approx=True, n_samples=num_mc_samples)
            data_chunks.append(dict(
                adv=min(1 - err_prob - alpha,1),
                baseline=baseline,
                method="guo_mc",
                k=k,
                sigma=sigma,
                epsilon=epsilon,
                is_valid_ai_baseline=is_valid_ai_baseline,
            ))

    # zCDP-based bound from Balle et al.
    data_chunks.append(dict(
        adv=min(1 - get_beta_for_rho(rho=rho, alpha=alpha) - alpha, 1),
        baseline=baseline,
        method="zcdp",
        k=k,
        sigma=sigma,
        epsilon=epsilon,
        is_valid_ai_baseline = is_valid_ai_baseline
    ))

    # Balle et al. Renyi DP bound, optimized over orders.
    # data_chunks.append(dict(
    #     adv=1 - get_beta_for_rho(rho=rho, alpha=alpha) - alpha,
    #     baseline=baseline,
    #     method="renyi_opt",
    #     k=k,
    #     sigma=sigma,
    #     epsilon=epsilon,
    #     is_valid_ai_baseline = is_valid_ai_baseline
    # ))

    # Our Bayes error based bound on AI.
    data_chunks.append(dict(
        adv=min(1 - rmin_gauss(alpha, mu) - alpha,1),
        baseline=baseline,
        method="bayes",
        k=k,
        sigma=sigma,
        epsilon=epsilon,
        is_valid_ai_baseline = is_valid_ai_baseline
    ))

    # Our unifying bound.
    data_chunks.append(dict(
        # adv=1 - riskcal.get_beta_from_pld(pld, alphas=alpha) - alpha,
        adv = 1 - norm.cdf(norm.isf(alpha) - mu) - alpha,
        baseline=baseline,
        method="unified",
        k=k,
        sigma=sigma,
        epsilon=epsilon,
        is_valid_ai_baseline = is_valid_ai_baseline
    ))

# %% id="db8db2d6-9a44-40ab-b5e0-261b920418f2"
renaming_dict = {
    "renyi_opt": r"SRR (Balle et al., via RDP, optimized)",
    "renyi": r"SRR (Balle et al., via RDP, fixed ord.)",
    "zcdp": r"SRR (Balle et al., via zCDP)",
    "cherubin": r"SRR (Cherubin et al.)",
    "unified": r"Unifying (via $f$-DP)",
    "bayes": r"SAI (via $f$-DP)",
    "guo_mi": r"SAI (Guo et al., via Fano, MI)",
    "guo_mc": r"SAI (Guo et al., via Fano, MC)",

    "method": "Method",
    "epsilon": r"$\varepsilon$",
    "mu": r"$\mu$",
    "k": r"$k$",
    "adv": "Risk (advantage)",
    "succ": "Success",
    "baseline": "Baseline",
}

# %% colab={"base_uri": "https://localhost:8080/", "height": 310} id="3bc8899e-7569-4e43-b9e8-a57782ff97ab" outputId="02ec28fc-765d-4bf0-d28f-8f1143280dff"
methods = ["zcdp", "guo_mi", "guo_mc", "unified", "bayes"]
order = [renaming_dict[method] for method in methods]
target = renaming_dict["unified"]

g = sns.relplot(
    data=(
        pd.DataFrame(data_chunks)
        .query("k == 2")
        .query("is_valid_ai_baseline == True")
        .query(f"method in {methods}")
        .assign(succ=lambda df: df.adv + df.baseline)
        .query("succ > 0")
        .replace(renaming_dict)
        .rename(columns=renaming_dict)
        .round(2)
    ),
    y=renaming_dict["succ"],
    x=renaming_dict["baseline"],
    hue=renaming_dict["method"],
    style=renaming_dict["method"],
    col=renaming_dict["epsilon"],

    hue_order=order,
    palette=ensure_color_order(order, target),
    markers=ensure_marker_order(order, target),

    errorbar=("ci", 0.95),
    dashes=False,
    # facet_kws={'sharey': False, 'sharex': True},
    kind="line",
    markersize=8,
)


xticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for ax in g.axes[0]:
    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width
    ax.set_xticks(xticks)
g.set(xlim=(0.496, 1.01))
g.set(ylim=(0.56, 1.005));


plt.savefig("images/gaussian_sai.pdf", bbox_inches="tight", dpi=300)

# %% [markdown] id="c6458df4-5d17-4051-acfa-a4b2bcdb9adb" jp-MarkdownHeadingCollapsed=true
# ## US Census

# %% [markdown] id="3f95a025-4c76-4669-9f16-d289a62534f1"
# Analysis of US Census algorithm using the Connect-the-Dots accountant.

# %% id="f02db2e1-b71c-4f99-874d-f2ff9767f6f2"
levels = ['US', 'State', 'County', 'PEPG', 'Track Sub. Group', 'Track Subset', 'Block Group', 'Block']
a = np.array([2, 27.4, 8.5, 13.1, 13.1, 23.8, 11.8, 0.3]) / 100
compositions = 10
total_rho = 3.65

sigmas = np.sqrt(compositions / (2 * a * total_rho))

sigma = sigmas[0]
pld_lst = []
pld = privacy_loss_distribution.from_discrete_gaussian_mechanism(
            sigma = sigma,
            value_discretization_interval=1e-3,
            use_connect_dots = True
        ).self_compose(compositions)
pld_lst.append(pld)

for index in range(1, len(sigmas)):
    new_pld = privacy_loss_distribution.from_discrete_gaussian_mechanism(
            sigma = sigmas[index],
            value_discretization_interval=1e-3,
            use_connect_dots = True
        ).self_compose(compositions)
    pld_lst.append(new_pld)
    total_pld = pld.compose(new_pld)

# %% colab={"base_uri": "https://localhost:8080/"} id="9d124bf3-a559-4dae-a601-85aa2542ebff" outputId="d9a0fcb4-3b5d-4208-9f6b-3fdfff8b14eb"
delta = 1e-10
special_baselines = [0.01, 0.1, "Worst-case"]
baselines = list(np.linspace(0, 1, 15)) + special_baselines

data_chunks = []
for i, pld in enumerate(pld_lst):
    level = levels[i]
    rho = total_rho * a[i]

    for baseline in baselines:
        alpha = baseline
        if baseline == "Worst-case":
            eta = riskcal.get_advantage_from_pld(pld)
            alpha = 0.5 - 0.5 * eta

        # Balle et al. Renyi DP bound, optimized over orders.
        data_chunks.append(dict(
            adv=1 - get_beta_for_rho(rho=rho, alpha=alpha) - alpha,
            baseline=baseline,
            method="renyi_opt",
            alpha=alpha,
            level=level,
        ))

        # Our unifying bound.
        data_chunks.append(dict(
            adv=1 - riskcal.get_beta_from_pld(pld, alphas=alpha) - alpha,
            baseline=baseline,
            method="unified",
            alpha=alpha,
            level=level
        ))

# %% id="4c19738d-f012-44d3-87ec-289f50e5c8e3"
renaming_dict = {
    # "renyi_opt": r"SRR (Balle et al., via RDP, optimized)",
    "renyi_opt": r"SRR (Balle et al., via zCDP)",
    "unified": r"Unifying (via $f$-DP)",

    "method": "Method",
    "adv": "Risk (advantage)",
    "succ": "Success",
    "baseline": "Baseline",
    "level": "Level",
}

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="c78b05f1-88d1-4b44-bb19-97641d8dbe4b" outputId="38a43c0d-00f7-4383-98fd-eb3083e95453"
(
    pd.DataFrame(data_chunks)
    .query(f"baseline in {special_baselines}")
    .pivot_table(index=['baseline', 'level'], columns='method', values='adv')
    .assign(decrease=lambda x: ((x['renyi_opt'] - x['unified']) / x['renyi_opt']))[['decrease']]
    .reset_index()
    .groupby("baseline")[["decrease"]]
    .max()
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 313} id="0b0f2120-f021-451b-be92-d1064fe2b968" outputId="8c82cd8c-b541-488f-db4b-37bef7005829"
g = sns.catplot(
    data=(
        pd.DataFrame(data_chunks)
        .query("baseline == 'Worst-case'")
        .replace(renaming_dict)
        .rename(columns=renaming_dict)
    ),
    x=renaming_dict["level"],
    y=renaming_dict["adv"],
    hue=renaming_dict["method"],
    kind="bar",

    aspect=3,
)

ax = g.axes[0, 0]
for container in ax.containers:
    # ax.bar_label(container, fmt='%.2f', label_type='edge', padding=5, fontsize=18)
    ax.bar_label(
        container,
        fmt='%.2f',
        label_type='edge',
        padding=5,
        fontsize=18,
        labels=[f'{h:.2f}' for h in container.datavalues],  # manually format labels
    )


# Set border (edge) on all bars
for patch in ax.patches:
    patch.set_edgecolor('black')    # Solid black outline
    patch.set_linewidth(1.5)        # Thickness of the border

for position, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

# Get current y-limits
bottom, top = ax.get_ylim()

# Increase the upper limit by 5% to make space for the label
ax.set_ylim(bottom, top * 1.05);
plt.savefig("images/census_worst_case_risk.pdf", bbox_inches="tight", dpi=300)

# %% colab={"base_uri": "https://localhost:8080/", "height": 908} id="8fcc3f84-908c-4b5d-a249-e687e305b8c2" outputId="bdf32122-7da8-4dad-abd7-e6f80224d353"
g = sns.catplot(
    data=(
        pd.DataFrame(data_chunks)
        .query(f"baseline in {special_baselines}")
        .replace(renaming_dict)
        .rename(columns=renaming_dict)
    ),
    x=renaming_dict["level"],
    y=renaming_dict["adv"],
    hue=renaming_dict["method"],
    row=renaming_dict["baseline"],
    kind="bar",

    # facet_kws={'sharey': False, 'sharex': True},
    aspect=3,
)

for ax_row in g.axes:
    for ax in ax_row:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge', padding=5, fontsize=18)
        
        # Set border (edge) on all bars
        for patch in ax.patches:
            patch.set_edgecolor('black')    # Solid black outline
            patch.set_linewidth(1.5)        # Thickness of the border
        
        for position, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

plt.savefig("images/census_risk.pdf", bbox_inches="tight", dpi=300)

# %% [markdown] id="2c6ae76e-24b4-4ffb-88e1-9afd335d9328"
# ### Worked example: state-level PLD

# %% id="b3dfb9a8-e464-4077-8ad6-7d5c1bb6e132"
state_pld = pld_lst[1]

# %% [markdown] id="94188eb1-5028-458e-86ee-9c924789fe01"
# $\varepsilon$ per the conversion Census bureau's conversion formula

# %% colab={"base_uri": "https://localhost:8080/"} id="4469821c-0227-449e-bf1e-c22909fe00f3" outputId="6f7c9c5f-3b3f-40b3-84bf-9f09ccd6f7cc"
delta = 1e-10
rho = total_rho * a[1]
eps = rho + 2 * np.sqrt(-rho * np.log(delta))
eps

# %% [markdown] id="d855e409-e6c1-421a-9bce-b07534d3277e"
# $\varepsilon$ from the $f$-DP analysis with the Connect-the-Dots accountant – reproduces the manual $f$-DP analysis by Su et al. 2024.

# %% colab={"base_uri": "https://localhost:8080/"} id="673bd031-42e8-4d1c-b92e-a0b587ce81d4" outputId="526b2303-95cb-4211-9f17-a2f1e27539fb"
delta = 1e-10
eps_tight = state_pld.get_epsilon_for_delta(delta)
eps_tight

# %% [markdown] id="ce0d23a9-9c40-4c3f-be08-c9d1702b3c34"
# Standard Kairouz et al. bound on TV privacy, which bounds membership inference / attribute inference (per Salem et al.) advantage

# %% colab={"base_uri": "https://localhost:8080/"} id="ccb3fe8b-2b7a-42c7-a6ee-1ef4086616c1" outputId="423c8878-3da0-45c9-ff75-4f4674bb4a0e"
eta_loose = (np.exp(eps) - 1 + 2 * delta) / (np.exp(eps) + 1)
eta_loose

# %% [markdown] id="46d941d8-d6fd-45e5-b0c5-a5f5c200fd50"
# Convert to accuracy

# %% colab={"base_uri": "https://localhost:8080/"} id="6ccf1641-7c41-4c01-bd4a-95c3ee790f27" outputId="f20cadee-4eb3-48e0-c40e-67432c58ab96"
0.5 + 0.5 * eta_loose

# %% [markdown] id="6dd73477-34ad-4d9f-968d-8495267f93ab"
# Bound on reconstruction advantage via RDP, per Balle et al.

# %% colab={"base_uri": "https://localhost:8080/"} id="7a9d759a-e0da-4a58-9ef6-e8488e134f7a" outputId="57bc799e-e395-497d-fce1-4f83deb25482"
alphas = np.linspace(0.1, 0.2, 100)
adv_vals = 1 - get_beta_for_rho(rho=rho, alpha=alphas) - alphas
max(adv_vals)

# %% [markdown] id="0152a675-93c7-41e6-9b20-0b6b61ea2c66"
# Tight advantage computation.

# %% colab={"base_uri": "https://localhost:8080/"} id="c37dc946-17c5-47cd-a369-413f6548318e" outputId="c8a55809-a71e-4a95-9d67-84d517f801fa"
eta = riskcal.get_advantage_from_pld(state_pld)
eta

# %% [markdown] id="f1e408b6-4c0c-44a3-a891-d2a4eb1d44a7"
# Tight maximum membership inference accuracy

# %% colab={"base_uri": "https://localhost:8080/"} id="ac7d7ee6-cfe5-4fe0-841f-49c58b0b6dc6" outputId="09bd6e2c-c287-4a8d-a927-260321d49582"
0.5 + 0.5 * eta

# %% [markdown] id="836b4c4c-dd8e-4226-80bb-93ec671f5b69"
# Bound on unifying advantage under a uniform prior over a state population.

# %% colab={"base_uri": "https://localhost:8080/"} id="ff09eefc-8a39-4fe7-b95d-1120a1dd3d7f" outputId="e98a0ba5-6c53-4fba-86bb-7547e92f3d79"
# Bound under uniform prior in Wyoming, the smallest US state.
baseline = 1/582_328  # Wyoming population in 2020.
risk_uniform_prior = 1 - riskcal.get_beta_from_pld(state_pld, alphas=baseline) - baseline
risk_uniform_prior, risk_uniform_prior < 1e-3

# %% colab={"base_uri": "https://localhost:8080/"} id="3730330c-cb91-4016-b9f4-1b448a35dcde" outputId="610f521e-6197-4319-81e5-c1041b466b1f"
baseline = 1 - 1/10000. # e.g., disease with 1 in 10000 prevalence.
risk_nonuniform_prior = 1 - riskcal.get_beta_from_pld(state_pld, alphas=baseline) - baseline
risk_nonuniform_prior, risk_nonuniform_prior < 0.001 * 100

# %% colab={"base_uri": "https://localhost:8080/", "height": 479} id="b90c04f6-93e8-4895-8f34-5bfb73cf6d46" outputId="9b0db7d6-7acc-42ea-9626-d74c28907932"
baselines = list(np.linspace(0, 0.1, 1000)) + list(np.linspace(0.1, 1, 1000))

g = sns.relplot(
    x=baselines,
    y=1 - riskcal.get_beta_from_pld(state_pld, alphas=baselines) - baselines,
    color=sns.color_palette()[1],
    kind="line",
    height=5,
    aspect=aspect_ratio,
)

# plt.hlines(eta, 0, 1, color="grey", linestyle="--")
# plt.plot(baselines, 1 - np.array(baselines), color="grey", linestyle="--", alpha=0.5)
plt.vlines(0.5 - 0.5 * eta, 0, eta + 0.02, color="grey", linestyle=":", alpha=0.5)
plt.xlabel("Baseline")
plt.xlim(-0.01, 1)
# plt.ylim(0, eta + 0.015)
plt.ylim(0, 0.53)


# xticks = np.linspace(52, 72, 5).astype(int)
for ax in g.axes[0]:
    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width
    # ax.set_xticks(xticks)

# g.set(xlim=(51.5,72))
# g.set(ylim=(0.08, 0.31));
# g.set(ylabel=None)

# plt.ylabel("Risk (Success $-$ Baseline)")
plt.savefig("images/risk_tunability.pdf", bbox_inches="tight", dpi=300)

# %% id="4FHIqSJR42Jb"
# for the big plots. See later in notebook
x1 = baselines
y1 = 1 - riskcal.get_beta_from_pld(state_pld, alphas=baselines) - baselines

# %% [markdown] id="c0a5c916-9712-4c0a-b17f-910afa9fb807"
# ## DP-SGD

# %% [markdown] id="72e35447-24e2-44be-8095-128f1cc22b9c"
# ### Renyi accounting for DP-SGD

# %% id="c1158498-c03f-4c9a-8ec0-968fdec48f6b"
# Numbers from experimental runs.
exp_data_cifar_raw = {
    'test_acc': [71.01, 70.17, 69.25, 67.03, 64.94],
    'epochs': [99, 99, 99, 99, 75],
    'batch_size': [8192, 8192, 8192, 8192, 8192],
    'sigma': [4, 5, 6, 8, 10]
}

exp_data_gpt_raw = {
    'eps': [4.0, 3.25, 2.75, 2.0, 1.5],
    'sigma': [
        0.5715066078978425,
        0.607176132813643,
        0.636648345573876,
        0.6945392638626464,
        0.7497660219048379
    ],
    'q': [
        0.0038010957846441,
        0.0038010957846441,
        0.0038010957846441,
        0.0038010957846441,
        0.0038010957846441
    ],
    'steps': [792, 792, 792, 792, 792],
    'test_acc': [
        0.7052752293577982 * 100,
        0.698394495412844 * 100,
        0.655963302752294 * 100,
        0.569954128440367 * 100,
        0.5217889908256881 * 100
    ]
}


# %% id="cd8b7ecc-1a5f-4cd0-a815-a75ff1311cb8"
exp_data_cifar = pd.DataFrame(exp_data_cifar_raw)
exp_data_gpt = pd.DataFrame(exp_data_gpt_raw)


# %% [markdown] id="lJyCWcLBFZsB"
# ### GPT-2

# %% [markdown] id="yPUcnCIuUgiL"
# #### Better Utility

# %% id="sUM5oKiUFcN7"
def get_gpt_pld(noise_multiplier, steps, sample_rate):
    sgd_noise_multiplier = noise_multiplier

    pld_sgd = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=sgd_noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
    ).self_compose(steps)

    return pld_sgd

def get_gpt_rdp(noise_multiplier, steps, sample_rate, orders=None):
    if orders is None:
        orders = np.linspace(1, 128, 10_000)

    sgd_noise_multiplier = noise_multiplier

    rdp_sgd = compute_rdp(noise_multiplier=noise_multiplier, q=sample_rate, steps=steps, orders=orders)
    return rdp_sgd, orders


# %% colab={"base_uri": "https://localhost:8080/"} id="bCJMl4HvGUDq" outputId="33044bc9-d2bb-46cd-e2eb-c751107042c3"
delta = 1e-5
special_baselines = [0.01, 0.1, "Worst-case"]
baselines = special_baselines

data_chunks = []
for i, row in exp_data_gpt.iterrows():
    sigma = row.sigma
    steps = int(row.steps)
    sample_rate = row.q
    test_acc = row.test_acc

    pld = get_gpt_pld(noise_multiplier=sigma, steps=steps, sample_rate=sample_rate)
    rdp, orders = get_gpt_rdp(noise_multiplier=sigma, steps=steps, sample_rate=sample_rate)
    epsilon = pld.get_epsilon_for_delta(delta)

    for baseline in baselines:
        if baseline == "Worst-case":
            eta = riskcal.get_advantage_from_pld(pld)
            alpha = 0.5 - 0.5 * eta
        else:
            alpha = baseline
        
        # ADP based bound.
        data_chunks.append(dict(
            adv=1 - riskcal.conversions.get_beta_for_epsilon_delta(epsilon=epsilon, delta=delta, alpha=alpha) - alpha,
            alpha=alpha,
            baseline=baseline,
            method="adp",
            sigma=sigma,
            test_acc=test_acc,
            epsilon=epsilon,
        ))
        
        # Balle et al. Renyi DP bound optimized over orders.
        data_chunks.append(dict(
            adv=min(1, 1 - get_beta_for_rdp(rdp=rdp, orders=orders, alpha=alpha) - alpha),
            alpha=alpha,
            baseline=baseline,
            method="renyi_opt",
            sigma=sigma,
            test_acc=test_acc,
            epsilon=epsilon,
        ))

        # Our unifying bound.
        data_chunks.append(dict(
            adv=min(1, 1 - riskcal.get_beta_from_pld(pld, alphas=alpha) - alpha),
            alpha=alpha,
            baseline=baseline,
            method="unified",
            sigma=sigma,
            test_acc=test_acc,
            epsilon=epsilon,
        ))

# %% id="-di5e032HLnI"
renaming_dict = {
    "renyi_opt": r"Balle et al., via RDP",
    "adp": r"Standard",
    "unified": r"Ours",

    "method": "Method",
    "epsilon": r"$\varepsilon$",
    "mu": r"$\mu$",
    "sigma": r"Noise scale $\sigma$",
    # "adv": r"Risk (Success $-$ Baseline)",
    "adv": r"Privacy risk, \%",
    "succ": "Success",
    "alpha": "Baseline",
    "baseline": "Baseline",
    "test_acc": r"Test accuracy, \%",
}

# %% colab={"base_uri": "https://localhost:8080/", "height": 408} id="RWAuf_HRHU6t" outputId="f78cf316-d106-466c-faee-62513259e2de"
order = [renaming_dict[k] for k in ["renyi_opt", "unified"]]
target = renaming_dict["unified"]

sns.relplot(
    data=(
        pd.DataFrame(data_chunks)
        .assign(succ=lambda row: row.adv + row.alpha)
        .drop(columns=["alpha"])
        .replace(renaming_dict)
        .rename(columns=renaming_dict)
    ),
    x=renaming_dict["test_acc"],
    y=renaming_dict["adv"],
    hue=renaming_dict["method"],
    style=renaming_dict["method"],
    col=renaming_dict["baseline"],

    hue_order=order,
    palette=ensure_color_order(order, target),
    # markers=ensure_marker_order(order, target),
    markers=False,

    dashes=False,
    legend=False,
    kind="line",
    markersize=8,
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 497} id="o3qOcx-NRDuQ" outputId="fde74190-fef5-4d9b-f5d9-0180820f66f7"
order = [renaming_dict[k] for k in ["renyi_opt", "unified"]]
target = renaming_dict["unified"]

g = sns.relplot(
    data=(
        pd.DataFrame(data_chunks)
        .query("baseline == 'Worst-case'")
        .replace(renaming_dict)
        .assign(succ=lambda row: row.adv + row.alpha)
        .rename(columns=renaming_dict)
    ),
    x=renaming_dict["test_acc"],
    y=renaming_dict["adv"],
    hue=renaming_dict["method"],
    style=renaming_dict["method"],
    # col=renaming_dict["baseline"],

    hue_order=order,
    palette=ensure_color_order(order, target),
    markers=ensure_marker_order(order, target),

    dashes=False,
    legend=False,
    kind="line",
    markersize=8,
    height=5,
    aspect=aspect_ratio,
)

xticks = np.linspace(52, 72, 5).astype(int)
for ax in g.axes[0]:
    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width
    ax.set_xticks(xticks)

g.set(xlim=(51.5,72))
g.set(ylim=(0.08, 0.31));
g.set(ylabel=None)
plt.savefig("images/dpsgd_gpt_rero_acc_vs_risk.pdf", bbox_inches="tight", dpi=300)

# %%
print((pd.DataFrame(data_chunks)
    .query("baseline == 'Worst-case'")
    .query("method in ['unified', 'adp']")
    .replace(renaming_dict)
    .assign(succ=lambda row: row.adv + row.alpha)
    .rename(columns=renaming_dict)
)[[renaming_dict["adv"], renaming_dict["test_acc"], renaming_dict["method"]]].to_latex())

# %%
order = [renaming_dict[k] for k in ["adp", "unified"]]
target = renaming_dict["unified"]

g = sns.relplot(
    data=(
        pd.DataFrame(data_chunks)
        .query("baseline == 'Worst-case'")
        .replace(renaming_dict)
        .assign(succ=lambda row: row.adv + row.alpha)
        .assign(adv=lambda row: row.adv * 100)
        .rename(columns=renaming_dict)
    ),
    x=renaming_dict["test_acc"],
    y=renaming_dict["adv"],
    hue=renaming_dict["method"],
    style=renaming_dict["method"],
    # col=renaming_dict["baseline"],

    hue_order=order,
    palette=ensure_color_order(order, target),
    markers=ensure_marker_order(order, target),

    dashes=False,
    legend=False,
    kind="line",
    markersize=8,
    height=5,
    aspect=aspect_ratio,
)


for ax in g.axes[0]:
    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width
# g.set(xlim=(64,72))
# g.set(ylim=(0.1, 0.65));
g.set(xlim=(51,72))
g.set(ylim=(0, 75));
g.set(ylabel=None)

# %% id="7TxjeJ8qTCLd"
data = (
    pd.DataFrame(data_chunks)
    .query("baseline == 'Worst-case'")
    .assign(succ=lambda row: row.adv + row.alpha)
)
x23 = np.array(data[data['method'] == 'renyi_opt']['test_acc'])
y23 = np.array(data[data['method'] == 'renyi_opt']['adv'])
idx23 = np.argsort(x23)
x23 = x23[idx23]
y23 = y23[idx23]

x24 = np.array(data[data['method'] == 'unified']['test_acc'])
y24 = np.array(data[data['method'] == 'unified']['adv'])
idx24 = np.argsort(x24)
x24 = x24[idx24]
y24 = y24[idx24]

x33 = np.array(data[data['method'] == 'renyi_opt']['sigma'])
y33 = np.array(data[data['method'] == 'renyi_opt']['adv'])
idx33 = np.argsort(x33)
x33 = x33[idx33]
y33 = y33[idx33]

x34 = np.array(data[data['method'] == 'unified']['sigma'])
y34 = np.array(data[data['method'] == 'unified']['adv'])
idx34 = np.argsort(x34)
x34 = x34[idx34]
y34 = y34[idx34]

# %% colab={"base_uri": "https://localhost:8080/", "height": 479} id="v_O-LwWgUYbf" outputId="e04e1cf7-6bf9-4f37-a05f-66d41bc377b5"
order = [renaming_dict[k] for k in ["renyi_opt", "unified"]]
target = renaming_dict["unified"]

g = sns.relplot(
    data=(
        pd.DataFrame(data_chunks)
        .query("baseline == 'Worst-case'")
        .replace(renaming_dict)
        .assign(succ=lambda row: row.adv + row.alpha)
        .rename(columns=renaming_dict)
    ),
    x=renaming_dict["sigma"],
    y=renaming_dict["adv"],
    hue=renaming_dict["method"],
    style=renaming_dict["method"],
    # col=renaming_dict["baseline"],

    hue_order=order,
    palette=ensure_color_order(order, target),
    markers=ensure_marker_order(order, target),

    dashes=False,
    legend=False,
    kind="line",
    markersize=8,
    height=5,
    aspect=aspect_ratio,
)

# xticks = np.linspace(52, 72, 5).astype(int)
# for ax in g.axes[0]:
#     for position, spine in ax.spines.items():
#         spine.set_visible(True)           # Ensure all spines are visible
#         spine.set_edgecolor("black")      # Set spine color
#         spine.set_linewidth(1.5)          # Set spine width
#     ax.set_xticks(xticks)

# g.set(xlim=(51.5,72))
# g.set(ylim=(0.08, 0.31));
# g.set(ylabel=None)
plt.savefig("images/dpsgd_gpt_rero_acc_vs_risk.pdf", bbox_inches="tight", dpi=300)

# %% [markdown] id="_ZzLVM_VFVNy"
# ### CIFAR 10

# %% [markdown] id="HeMi5qf2MKc2"
# #### Better Utility

# %% id="ef5206d1-4ec3-4bf2-9181-a949a1b7b55a"
norm_noise_multiplier = 8.0

def get_cifar10_pld(noise_multiplier, epochs, batch_size=8192, data_size=50000):
    sgd_noise_multiplier = noise_multiplier
    sample_rate = batch_size / data_size
    steps = int(epochs * np.ceil(data_size / batch_size))

    # from https://github.com/ftramer/Handcrafted-DP/blob/main/cnns.py
    # compute the budget spent in normalization.
    pld_norm = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=norm_noise_multiplier,
        use_connect_dots=True,
    ).self_compose(2)

    pld_sgd = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=sgd_noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
    ).self_compose(steps)

    return pld_norm.compose(pld_sgd)

def get_cifar10_rdp(noise_multiplier, epochs, batch_size=8192, data_size=50000, orders=None):
    if orders is None:
        orders = np.linspace(1, 128, 10_000)
    rdp_norm = compute_rdp(noise_multiplier=norm_noise_multiplier, q=1, steps=2, orders=orders)

    sgd_noise_multiplier = noise_multiplier
    sample_rate = batch_size / data_size
    steps = int(epochs * np.ceil(data_size / batch_size))

    rdp_sgd = compute_rdp(noise_multiplier=noise_multiplier, q=sample_rate, steps=steps, orders=orders)
    return rdp_norm + rdp_sgd, orders


# %% colab={"base_uri": "https://localhost:8080/"} id="fba65b95-64d5-4097-8cda-53ab4ab9f001" outputId="16394c74-f874-428a-bda4-30851f024ab4"
delta = 1e-5
special_baselines = [0.01, 0.1, "Worst-case"]
baselines = special_baselines

data_chunks = []
for i, row in exp_data_cifar.iterrows():
    sigma = row.sigma
    epochs = row.epochs
    batch_size = row.batch_size
    test_acc = row.test_acc

    pld = get_cifar10_pld(noise_multiplier=sigma, epochs=epochs)
    rdp, orders = get_cifar10_rdp(noise_multiplier=sigma, epochs=epochs)
    epsilon = pld.get_epsilon_for_delta(delta)

    for baseline in baselines:
        if baseline == "Worst-case":
            eta = riskcal.get_advantage_from_pld(pld)
            alpha = 0.5 - 0.5 * eta
        else:
            alpha = baseline
        
        # ADP based bound.
        data_chunks.append(dict(
            adv=1 - riskcal.conversions.get_beta_for_epsilon_delta(epsilon=epsilon, delta=delta, alpha=alpha) - alpha,
            alpha=alpha,
            baseline=baseline,
            method="adp",
            sigma=sigma,
            test_acc=test_acc,
            epsilon=epsilon,
        ))

        # Balle et al. Renyi DP bound optimized over orders.
        data_chunks.append(dict(
            adv=1 - get_beta_for_rdp(rdp=rdp, orders=orders, alpha=alpha) - alpha,
            alpha=alpha,
            baseline=baseline,
            method="renyi_opt",
            sigma=sigma,
            test_acc=test_acc,
            epsilon=epsilon,
        ))

        # Our unifying bound.
        data_chunks.append(dict(
            adv=1 - riskcal.get_beta_from_pld(pld, alphas=alpha) - alpha,
            alpha=alpha,
            baseline=baseline,
            method="unified",
            sigma=sigma,
            test_acc=test_acc,
            epsilon=epsilon,
        ))

# %% id="9e8e914b-4bfa-419d-b1b0-6cefc73e1558"
renaming_dict = {
    "renyi_opt": r"Balle et al., via RDP",
    "unified": r"Ours",

    "method": "Method",
    "epsilon": r"$\varepsilon$",
    "mu": r"$\mu$",
    "sigma": r"Noise scale $\sigma$",
    "adv": r"Privac risk, \%",
    "succ": "Success",
    "alpha": "Baseline",
    "baseline": "Baseline",
    "test_acc": r"Model accuracy, \%",
}

# %% colab={"base_uri": "https://localhost:8080/", "height": 479} id="6362f4a7-cb80-4906-a890-bef546027426" outputId="3ce834aa-3f8a-4e1a-fa0b-7980f57b7f56"
order = [renaming_dict[k] for k in ["renyi_opt", "unified"]]
target = renaming_dict["unified"]

g = sns.relplot(
    data=(
        pd.DataFrame(data_chunks)
        .query("baseline == 'Worst-case'")
        .replace(renaming_dict)
        .assign(succ=lambda row: row.adv + row.alpha)
        .rename(columns=renaming_dict)
    ),
    x=renaming_dict["test_acc"],
    y=renaming_dict["adv"],
    hue=renaming_dict["method"],
    style=renaming_dict["method"],
    # col=renaming_dict["baseline"],

    hue_order=order,
    palette=ensure_color_order(order, target),
    markers=ensure_marker_order(order, target),

    dashes=False,
    legend=False,
    kind="line",
    markersize=8,
    height=5,
    aspect=aspect_ratio,
)


for ax in g.axes[0]:
    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width
# g.set(xlim=(64,72))
# g.set(ylim=(0.1, 0.65));
g.set(xlim=(64,72))
g.set(ylim=(0.15, 0.62));
g.set(ylabel=None)
plt.savefig("images/dpsgd_cifar_rero_acc_vs_risk.pdf", bbox_inches="tight", dpi=300)

# %% id="pAW4Dz8k4Mta"
# for big plot. See later in notebook

data = (
        pd.DataFrame(data_chunks)
        .query("baseline == 'Worst-case'")
        .assign(succ=lambda row: row.adv + row.alpha)
    )

# store adv and test acc for later
x21 = np.array(data[data['method'] == 'renyi_opt']['test_acc'])
y21 = np.array(data[data['method'] == 'renyi_opt']['adv'])
idx21 = np.argsort(x21)
x21 = x21[idx21]
y21 = y21[idx21]

x22 = np.array(data[data['method'] == 'unified']['test_acc'])
y22 = np.array(data[data['method'] == 'unified']['adv'])
idx22 = np.argsort(x22)
x22 = x22[idx22]
y22 = y22[idx22]

# store sigma and adv for later
x35 = np.array(data[data['method'] == 'renyi_opt']['sigma'])
y35 = np.array(data[data['method'] == 'renyi_opt']['adv'])
idx35 = np.argsort(x35)
x35 = x35[idx35]
y35 = y35[idx35]

x36 = np.array(data[data['method'] == 'unified']['sigma'])
y36 = np.array(data[data['method'] == 'unified']['adv'])
idx36 = np.argsort(x36)
x36 = x36[idx36]
y36 = y36[idx36]

# %% [markdown] id="zGTcSqmnMi7Q"
# #### Less Noise
#
# Note: This figure was removed

# %% colab={"base_uri": "https://localhost:8080/", "height": 120, "referenced_widgets": ["e7d0f80953a64f91baae7eb8249f1140", "d0b82b3ab94f4056aeb7ad5fc3913451", "5a2270a9b97c482e8ce445301aee20e7", "566a6fdd6bc641c3b85e135163feb338", "477b2e1a45164c48a56701ca7aaf590a", "66cb1ae413f2416a94adf85ae530d209", "003eaeb80481413b91de665a9b553e78", "365f12be8b17432b8bb20d0ca46d4200", "845c121e98904d6698bee3b3f90d3973", "f3158f40fe734f13a5276d9ceca1fdfc", "6aa7a81fce98403eb0f41242ea809686"]} id="8751f2e9-270a-41cd-9311-9f0856ced11b" outputId="9926845c-ce12-4568-dd12-89a4e6683aae"
delta = 1e-5
special_baselines = [0.01, 0.1, "Worst-case"]
baselines = special_baselines
sigmas = np.linspace(0.6, 1.0, 10)
sample_rate = 0.001
num_steps = 10_000

data_chunks = []
for baseline, sigma in tqdm(list(itertools.product(baselines, sigmas))):
    pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=sigma,
        sampling_prob=sample_rate,
        use_connect_dots=True,
    ).self_compose(num_steps)
    orders = np.linspace(0, 128, 10_000)
    rdp = compute_rdp(noise_multiplier=sigma, q=sample_rate, steps=num_steps, orders=orders)
    epsilon = pld.get_epsilon_for_delta(delta)

    for baseline in baselines:
        alpha = baseline
        if baseline == "Worst-case":
            eta = riskcal.get_advantage_from_pld(pld)
            alpha = 0.5 - 0.5 * eta

        # Balle et al. Renyi DP bound optimized over orders.
        data_chunks.append(dict(
            adv=1 - get_beta_for_rdp(rdp=rdp, orders=orders, alpha=alpha) - alpha,
            alpha=alpha,
            baseline=baseline,
            method="renyi_opt",
            sigma=sigma,
            epsilon=epsilon,
        ))

        # Our unifying bound.
        data_chunks.append(dict(
            adv=1 - riskcal.get_beta_from_pld(pld, alphas=alpha) - alpha,
            alpha=alpha,
            baseline=baseline,
            method="unified",
            sigma=sigma,
            epsilon=epsilon,
        ))

# %% id="t52984tz3UKp"
data=(
    pd.DataFrame(data_chunks)
    .assign(succ=lambda row: row.adv + row.alpha)
    .drop(columns=["alpha"])
    .query("baseline == 'Worst-case'")
)

x31 = np.array(data[data["method"] == "renyi_opt"]['sigma'])
y31 = np.array(data[data["method"] == "renyi_opt"]['adv'])
idx31 = np.argsort(x31)
x31 = x31[idx31]
y31 = y31[idx31]

x32 = np.array(data[data["method"] == "unified"]['sigma'])
y32 = np.array(data[data["method"] == "unified"]['adv'])
idx32 = np.argsort(x32)
x32 = x32[idx32]
y32 = y32[idx32]

# %% colab={"base_uri": "https://localhost:8080/", "height": 498} id="39b6ba9c-d10b-4c90-8532-1f20e09ce502" outputId="ba191937-da42-4dec-b21b-7efdc782e787"
order = [renaming_dict[k] for k in ["renyi_opt", "unified"]]
target = renaming_dict["unified"]

g = sns.relplot(
    data=(
        pd.DataFrame(data_chunks)
        .assign(succ=lambda row: row.adv + row.alpha)
        .drop(columns=["alpha"])
        .query("baseline == 'Worst-case'")
        .replace(renaming_dict)
        .rename(columns=renaming_dict)
    ),
    x=renaming_dict["sigma"],
    y=renaming_dict["adv"],
    hue=renaming_dict["method"],
    style=renaming_dict["method"],
    # col=renaming_dict["baseline"],

    hue_order=order,
    palette=ensure_color_order(order, target),
    markers=ensure_marker_order(order, target),
    # markers=False,

    dashes=False,
    legend=False,
    kind="line",
    markersize=8,
    height=5,
    aspect=aspect_ratio,
)

xticks = [0.6, 0.7, 0.8, 0.9, 1.0,1.1]
for ax in g.axes[0]:
    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width
    ax.set_xticks(xticks)

g.set(xlim=(0.596,1.004))
g.set(ylim=(0.0496,0.25));
g.set(ylabel=None) #try this
# g.set(xlim=(0.55,1.05))
# g.set(ylim=(0.04,0.25));
plt.savefig("images/dpsgd_cifar_rero_risk_vs_sigma.pdf", bbox_inches="tight", dpi=300)

# %% [markdown] id="cj3KiklF5C5W"
# ### Big Plot Cifar
#
# This figure is in the appendix. It plots side by side the risk of every value of sigma tested, and the accuracy vs risk.

# %% colab={"base_uri": "https://localhost:8080/", "height": 513} id="4oPq5zs7ZqNJ" outputId="22b2a7e2-75dd-4911-b1bf-7ae1b70919e3"
fig, ax = plt.subplots(1, 2, figsize=(10 * 1.12, 5))

ax[0].plot(x35, y35, label="Balle et al.")
ax[0].plot(x36, y36, label="Ours")
ax[0].set_xlabel("Noise scale $\sigma$")
ax[0].set_ylabel("Risk (advantage)")
xticks = [0.6, 0.7, 0.8, 0.9, 1.0,1.1]
for position, spine in ax[0].spines.items():
    spine.set_visible(True)           # Ensure all spines are visible
    spine.set_edgecolor("black")      # Set spine color
    spine.set_linewidth(1.5)          # Set spine width
# ax[0].set_xticks(xticks)

# ax[0].set_xlim(0.596,1.004)
# ax[0].set_ylim(0.0496,0.25)
ax[0].set_ylim(0.15, 0.62)
ax[0].set_title('Less Noise')

ax[1].plot(x21, y21, label="Balle et al.")
ax[1].plot(x22, y22, label="Ours")
ax[1].set_xlabel("Test accuracy \%")

for position, spine in ax[1].spines.items():
    spine.set_visible(True)           # Ensure all spines are visible
    spine.set_edgecolor("black")      # Set spine color
    spine.set_linewidth(1.5)          # Set spine width]
ax[1].set_xlim(64,72)
ax[1].set_ylim(0.15, 0.62)
ax[1].set_title('Better Utility')


# ax[2].plot(x1, y1, c = 'C1', label="Balle et al.")
# for position, spine in ax[2].spines.items():
#     spine.set_visible(True)           # Ensure all spines are visible
#     spine.set_edgecolor("black")      # Set spine color
#     spine.set_linewidth(1.5)          # Set spine width]

# xticks = [0,0.25, 0.5, 0.75,1.0]
# ax[2].set_xticks(xticks)
# ax[2].set_xlabel("Baseline")
# ax[2].set_title('Tunability')
# ax[2].set_xlim(64,72)
# ax[2].set_ylim(0.15, 0.62)

# plt.legend()
# plt.show()
plt.savefig("images/double_plot_cifar.pdf", bbox_inches="tight", dpi=300)

# %% colab={"base_uri": "https://localhost:8080/"} id="KfzgOtzWamsh" outputId="69ab469f-dc74-45df-f944-9868a444f7db"
# SUMMARY OF GAINS

target_risk = 0.3

idx_acc_rdp = np.argmin(np.abs(y21 - target_risk))
idx_acc_pld = np.argmin(np.abs(y22 - target_risk))
accuracy_rdp = x21[idx_acc_rdp]
accuracy_pld = x22[idx_acc_pld]

idx_risk_rdp = np.argmin(np.abs(y35 - target_risk))
idx_risk_pld = np.argmin(np.abs(y36 - target_risk))
sigma_rdp = x35[idx_risk_rdp]
sigma_pld = x36[idx_risk_pld]

print(f'RDP sigma is {sigma_rdp:0.4}')
print(f'f-DP sigma is {sigma_pld:0.4}')
print(f'Percent improvement: {(sigma_rdp - sigma_pld)/sigma_rdp*100:0.3}%')

print(f'RDP accuracy is {accuracy_rdp:0.3}%')
print(f'f-DP accuracy is {accuracy_pld:0.3}%')
print(f'pp improvement: {accuracy_pld - accuracy_rdp:0.3}%')

# %% [markdown] id="Q3vCFEE6TWbh"
# ### Big Plot GPT
#
# This is Figure 1.

# %% colab={"base_uri": "https://localhost:8080/", "height": 496} id="FPky7AhBWgvI" outputId="4698ff68-4d14-4b08-e95e-0fe7308d0c9f"
fig, ax = plt.subplots(1, 3, figsize = (15 * 1.15, 5))

ax[0].plot(x33, y33, label="Balle et al.")
ax[0].plot(x34, y34, label="Ours")
ax[0].set_xlabel("Noise scale $\sigma$")
ax[0].set_ylabel("Risk (Success $-$ Baseline)")
xticks = [0.5, 0.55, 0.6,0.65, 0.7, 0.75]
for position, spine in ax[0].spines.items():
    spine.set_visible(True)           # Ensure all spines are visible
    spine.set_edgecolor("black")      # Set spine color
    spine.set_linewidth(1.5)          # Set spine width
# ax[0].set_xticks(xticks)

# ax[0].set_xlim(0.596,1.004)
# ax[0].set_ylim(0.0496,0.25)


ax[0].set_title('Less Noise')

ax[1].plot(x23, y23, label="Balle et al.")
ax[1].plot(x24, y24, label="Ours")
ax[1].set_xlabel("Test accuracy \%")

xticks = [50,55,60,65,70]
for position, spine in ax[1].spines.items():
    spine.set_visible(True)           # Ensure all spines are visible
    spine.set_edgecolor("black")      # Set spine color
    spine.set_linewidth(1.5)          # Set spine width]
ax[1].set_xticks(xticks)

ax[1].set_xlim(51,72)
ax[1].set_ylim(0.08, 0.31)
ax[1].set_title('Better Utility')

ax[2].plot(x1, y1, c = 'C1', label="Balle et al.")
for position, spine in ax[2].spines.items():
    spine.set_visible(True)           # Ensure all spines are visible
    spine.set_edgecolor("black")      # Set spine color
    spine.set_linewidth(1.5)          # Set spine width]

xticks = [0,0.25, 0.5, 0.75,1.0]
ax[2].set_xticks(xticks)
ax[2].set_xlabel("Baseline")
ax[2].set_title('Tunability')
# ax[2].set_xlim(64,72)
# ax[2].set_ylim(0.15, 0.62)

# plt.legend()
# plt.show()
plt.savefig("images/triple_plot_final.pdf", bbox_inches="tight", dpi=300)

# %% colab={"base_uri": "https://localhost:8080/"} id="HDCcSRIrbprO" outputId="7c98bc3b-0e21-482a-8681-fcd49070de98"
# SUMMARY OF GAINS REFERENCED IN THE MAIN BODY
target_risk = 0.16

idx_acc_rdp = np.argmin( np.abs(y23 - target_risk))
idx_acc_pld = np.argmin( np.abs(y24 - target_risk))
accuracy_rdp = x23[idx_acc_rdp]
accuracy_pld = x24[idx_acc_pld]

idx_risk_rdp = np.argmin( np.abs(y33 - target_risk))
idx_risk_pld = np.argmin( np.abs(y34 - target_risk))
sigma_rdp = x33[idx_risk_rdp]
sigma_pld = x34[idx_risk_pld]

print(f'RDP sigma is {sigma_rdp:0.4}')
print(f'f-DP sigma is {sigma_pld:0.4}')
print(f'Percent improvement: {(sigma_rdp - sigma_pld)/sigma_rdp*100:0.3}%')


print(f'RDP accuracy is {accuracy_rdp:0.3}%')
print(f'f-DP accuracy is {accuracy_pld:0.3}%')
print(f'pp improvement: {accuracy_pld - accuracy_rdp:0.3}%')


# %% [markdown]
# ## Query Answering

# %%
@dataclasses.dataclass
class Privacy:
    epsilon: float
    delta: float

class ExactAccountant:
    """
    Implements k-folds homogeneous composition from Kairouz, et al
    Theorem 3.4
    https://arxiv.org/pdf/1311.0776.pdf

    Based on an OpenDP adaptive composition accountant.
    """
    def __init__(self, pld, tol=1e-5/2):
        self.pld = pld
        self.k = 0
        
    def self_compose(self, k=1):
        self.k += k
        return self
        
    def reset(self):
        self.k = 0
        return self

    def get_composed_epsilon(self, delta):
        tol = delta / 2
        epsilon = self.pld.get_epsilon_for_delta(delta)

        if self.k == 0:
            return (0.0, 0.0)

        basic = self.k * epsilon
        optimal_left_side = ((np.exp(epsilon) - 1) * epsilon * self.k)/(np.exp(epsilon) + 1)
        optimal_a = optimal_left_side + epsilon * np.sqrt(2 * self.k * np.log(epsilon + (np.sqrt(self.k*epsilon*epsilon)/tol)))
        optimal_b = optimal_left_side + epsilon * np.sqrt(2 * self.k * (1/tol))
        delta = 1 - (1 - delta) ** self.k
        delta = delta * (1 - delta) + tol
        return min(basic, optimal_a, optimal_b), delta

    def get_epsilon_for_delta(self, delta, grid_size=1000):
        
        def solve_for_initial_delta(target_delta):
            def f(delta):
                composed_delta = 1 - (1 - delta) ** self.k
                composed_delta = composed_delta * (1 - composed_delta) + composed_delta / 2
                return composed_delta - target_delta
        
            # Initial guess
            delta_bound = 1.0
            while self.pld.get_epsilon_for_delta(delta_bound) <= 0.:
                delta_bound /= 2
            initial_guess = delta_bound / 2
           
            try:
                # Solve the equation using
                solution = optimize.root_scalar(f, x0=initial_guess, method='newton')
        
                if solution.converged and 0 <= solution.root <= 1:
                    return solution.root
                else:
                    raise ValueError("Solution not in [0, 1] range or failed to converge")
            
            except Exception as e:
                print(f"Error in solving: {str(e)}")
                print("Attempting to find solution via grid search...")
                
                # If the solver method fails, try to find solution via grid search
                deltas = np.linspace(0, delta_bound, grid_size)
                values = [f(d) for d in deltas]
                closest_index = np.argmin(np.abs(values))
                
                if abs(values[closest_index]) < 2 / grid_size:  # tolerance
                    return deltas[closest_index]
                else:
                    raise ValueError("No solution found in [0, 1] range")

        delta_prime = solve_for_initial_delta(delta)
        epsilon, delta_approx = self.get_composed_epsilon(delta=delta_prime)
        delta_err = np.abs(delta_approx - delta)
        if delta_err > 1e-4:
            print(f"Estimated initial delta far apart: {delta_err=}, {delta_approx=}, {delta=}")
        
        return epsilon


# %%
ks = range(1, 16)
noise_param = 5.0
delta = 1e-5
alphas = np.array([.01, .1])

pld = privacy_loss_distribution.from_laplace_mechanism(noise_param)
epsilon = pld.get_epsilon_for_delta(delta)

data_chunks = []

for k in ks:
    composed_pld = pld.self_compose(k)
    exact_worst_adv = riskcal.conversions.get_advantage_from_pld(composed_pld)
    exact_adv = 1 - riskcal.conversions.get_beta_from_pld(composed_pld, alphas=alphas) - alphas
    
    smartnoise_epsilon = (
        ExactAccountant(pld, tol=delta / 2)
        .self_compose(k)
        .get_epsilon_for_delta(delta)
    )
    smartnoise_worst_adv = riskcal.conversions.get_advantage_for_epsilon_delta(
        epsilon=smartnoise_epsilon, delta=delta)
    smartnoise_adv = 1 - riskcal.conversions.get_beta_for_epsilon_delta(
        epsilon=smartnoise_epsilon, delta=delta, alpha=alphas) - alphas
    
    data_chunks.append(dict(
        k=k,
        adv=exact_worst_adv,
        baseline="Worst-case",
        method="fdp",
    ))
    data_chunks.append(dict(
        k=k,
        adv=smartnoise_worst_adv,
        baseline="Worst-case",
        method="standard",
    ))

    for baseline, exact_adv_val, smartnoise_adv_val in zip(alphas, exact_adv, smartnoise_adv):
        data_chunks.append(dict(
            k=k,
            adv=exact_adv_val,
            baseline=baseline,
            method="fdp",
        ))
        data_chunks.append(dict(
            k=k,
            adv=smartnoise_adv_val,
            baseline=baseline,
            method="standard",
        ))

# %%
renaming_dict = {
    "fdp": r"via $f$-DP",
    "standard": r"Standard, via $(\varepsilon, \delta)$-DP",
    "baseline": "Baseline",
    "method": "Method",
    "k": "Number of queries",
    "adv": "Risk (advantage)",
}

# %%
import matplotlib.ticker as ticker

order = [renaming_dict[k] for k in ["standard", "fdp"]]

g = sns.relplot(
    data=(
        pd.DataFrame(data_chunks)
        .rename(columns=renaming_dict)
        .replace(renaming_dict)
    ),
    x=renaming_dict["k"],
    y=renaming_dict["adv"],
    col=renaming_dict["baseline"],
    hue=renaming_dict["method"],
    hue_order=order,
    kind="line",
    aspect=0.9,
    col_order=list(alphas) + ["Worst-case"],
)

for ax in g.axes[0]:
    ax.minorticks_on()
    ax.set_xticks([1, 5, 10, 15])
    ax.set_xlim(min(ks), max(ks) + 0.05)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(which="minor", axis="x", color='black', alpha=.05, linestyle="-", linewidth=1,)
    ax.set_ylim(0, 0.8)
    # ax.hlines(0.09, min(ks), max(ks), linestyle="--", color="lightgrey")

    for position, spine in ax.spines.items():
        spine.set_visible(True)           # Ensure all spines are visible
        spine.set_edgecolor("black")      # Set spine color
        spine.set_linewidth(1.5)          # Set spine width

plt.savefig("images/query_reid_adv.pdf", bbox_inches="tight", dpi=300)
