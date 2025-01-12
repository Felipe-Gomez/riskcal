from dataclasses import dataclass
from typing import Union
import numpy as np
from dp_accounting.pld.privacy_loss_distribution import PrivacyLossDistribution


"""
This module provides functions that compute a tradeoff curve T(P,Q)
given a PLRVs instance (Probability Loss Random Variables). We rely 
on the notation from:

  - Kulynych et al. (https://arxiv.org/pdf/2407.02191) 
  - Dong et al. (https://arxiv.org/pdf/1905.02383)

Below, we summarize the main concepts and notation needed to 
understand how these functions work.

----------------------------------------------------------------------
DOMINATING PAIRS AND TRADEOFF FUNCTIONS
----------------------------------------------------------------------

Given a mechanism M, we say (P, Q) is a discrete-valued dominating 
pair if, for all 0 <= alpha <= 1,

    T(P,Q)(alpha) <= T(M(D), M(D'))(alpha),

where T denotes the tradeoff function. (In this code, T(P,Q) is often 
called f.)

We define random variables X and Y via:

    Y = log[P(o) / Q(o)]   with o ~ P,
    X = log[P(o') / Q(o')] with o' ~ Q.

In cases where (P, Q) have disjoint support, X can have a point mass 
at -∞, and Y can have a point mass at +∞. Their domains are:

    Domain_X = {-∞} ∪ {x_0, x_1, ..., x_{k-1}}
             = {x_{-1}} ∪ {x_0, ..., x_{k-1}},

    Domain_Y = {y_0, y_1, ..., y_{l-1}} ∪ {+∞}
             = {y_0, ..., y_{l-1}} ∪ {y_l}.

----------------------------------------------------------------------
GOOGLE DP_ACCOUNTING LIBRARY AND DISCRETIZATION
----------------------------------------------------------------------

In practice, all PLRVs passed to this module come from the 
`dp_accounting` library developed by Google. That library discretizes 
the privacy loss random variables so that

    x_i = Δ * (x_0 + i), 0 <= i <= k - 1
    y_j = Δ * (y_0 + j), 0 <= j <= l - 1,

where the scalar Δ is the discretization parameter. 
As the functions in this module are scale invarient to Δ 
(see, e.g. Algorithm 1 in Kulynych et al.), we let Δ=1 throughout 
this module. As a result, the finite domain of X and Y takes on 
equally spaced integer points, plus possible point masses at 
-∞ (for X) and +∞ (for Y).

----------------------------------------------------------------------
PIECEWISE LINEARITY AND ALPHA_BAR
----------------------------------------------------------------------

We let f = T(P,Q). In some cases, f(alpha) != f^{-1}(alpha). This 
discrepancy often arises under Poisson subsampling, where one order 
corresponds to remove-adjacent datasets (D → D') and the other to 
add-adjacent datasets (D' → D). When this occurs, we must apply a 
symmetrization step to obtain the symmetric tradeoff curve that 
matches the add/remove neighboring relation. This is precisely the 
process described in Definition F.1 of Dong et al. (arXiv:1905.02383).

To carry out this symmetrization, we need to find the smallest value 
of alpha for which -1 lies in the subdifferential of f. We denote 
this value alpha_bar.

It can be shown that f is piecewise linear with breakpoints at
(Pr[X > x_i], Pr[Y <= x_i]) for -1 ≤ i ≤ k - 1. The linear segment 
to the right of breakpoint i has slope -e^(x_i). At each breakpoint
i < k - 1, the subdifferential is the interval
[-e^(x_{i+1}), -e^(x_i)].

Let m be such that x_m = 0 (i.e., m = -x_0). Then -1 appears in 
subdifferentials m and m - 1:

    [-e^(x_{m+1}), -e^(x_m)] 
    [-e^(x_m), -e^(x_{m-1})]

Noting that alpha decreases as the index increases, the smallest 
alpha at which -1 appears in the subdifferential corresponds to i = m. 
This yields:

    alpha_bar = Pr[X > x_m] = Pr[X > 0],
    f(alpha_bar) = Pr[Y <= x_m] = Pr[Y <= 0].

Once alpha_bar is determined, implementing the symmetrization process 
(Definition F.1 of Dong et al.) is straightforward.


----------------------------------------------------------------------
IMPLEMENTATIONS
----------------------------------------------------------------------

The file defines and uses:

  - `_tradeoff_function(plrvs, alphas)`: 
      Computes T(P,Q)(alpha) (the FNR at each FPR alpha).
  
  - `_inverse_tradeoff_function(plrvs, alphas)`:
      Computes T(Q,P)(alpha), effectively the 'inverse' curve.

  - `get_beta(plrvs, alphas)`:
      Main entry point. Checks whether (P,Q) is symmetric; if not,
      computes a symmetrized tradeoff curve using alpha_bar, as above.

All of the above functions assume a PLRVs (defined below) instance as 
input and rely on NumPy vectorization for efficient computation.
"""


@dataclass
class PLRVs:
    """
    Privacy loss random variables. 
    """

    y0: int
    pmf_Y: np.ndarray
    infinity_mass_Y: float
    
    x0: int
    pmf_X: np.ndarray
    minus_infinity_mass_X: float
    
    is_symmetric: bool

def _tradeoff_function(
    plrvs: PLRVs,
    alphas: np.ndarray,
) -> np.ndarray:
    '''
    
    Computes the False Negative Rates (FNR) for a set of 
    False Positive Rates (FPR), given the PLRVs X and Y. 
    
    This function corresponds to Algorithm 1 in 
    Kulynych et al. For each value of FPR, we find the 
    Neyman-Pearson threshold in Domain_X, along with
    the Neyman-Pearson coin flip probability, then compute 
    the corresponding FNR in Domain_Y.
    
    Args:

        plrvs:
            A PLRVs instance with attributes:
            - pmf_X, pmf_Y: Probability mass functions for X, Y 
              over finite domains.
            - x0, y0: Offsets for the domain points.

        alphas: np.ndarray
            Ascending array of FPR values for which to compute FNR.

    Returns:
        betas: np.ndarray
            Array of FNR values corresponding to the input alphas.
    '''
    
    # Reverse alphas for implementation convenience
    alphas = alphas[::-1]
    
    # Preallocate beta array for output
    betas = np.empty_like(alphas)
    
    # Length of domains of X and Y (excluding masses at +- ∞)
    k = len(plrvs.pmf_X)
    l = len(plrvs.pmf_Y)
    
    # ----------------------------------------------------------------------
    # Step 1: Construct complement CDF of X so that ccdf_X[i] = Pr[X > x_{i-1}],
    #         for 0 <= i <= k, 
    #         where ccdf_X[0] = Pr[X > x_{-1}] = Pr[X > -∞] = 1 - Pr[X = -∞].
    #         The input pmf is over x_0, x_1, ..., x_{k-1}. 
    # ----------------------------------------------------------------------
    
    ccdf_X = np.empty(k+1)
    
    # Accumulate from the right: cumsum of reversed pmf gives us
    # the tail distribution. Then reverse again to get the usual order.
    ccdf_X[:-1] = np.cumsum(plrvs.pmf_X[::-1])[::-1]
    
    # Pr[X > x_{k-1}] = 0, so we add that as the last element
    ccdf_X[-1] = 0
    
    # ----------------------------------------------------------------------
    # Step 2: Construct cdf_Y so that cdf_Y[i] = Pr[Y <= y_i],
    #         for 0 <= i <= l - 1. Excludes the mass at +∞. 
    #         The input pmf is over y_0, y_1, ..., y_{l-1}. 
    # ----------------------------------------------------------------------
    
    cdf_Y = np.cumsum(plrvs.pmf_Y)

    # ----------------------------------------------------------------------
    # Step 3: Compute t array, where t[i] represents the index in X's domain
    #         of the NP threshold for FPR alpha[i].
    #         t ∈ [-1, k-1], with t = -1 meaning the threshold is -∞.
    # ----------------------------------------------------------------------
    
    # Search in ccdf_X[::-1] for the largest index where ccdf_X[index] <= alpha. 
    # Note that reversing ccdf_X aligns with reversed alphas.
    t = len(plrvs.pmf_X) - np.searchsorted(ccdf_X[::-1], alphas, side='right')
    
    # ----------------------------------------------------------------------
    # Step 4: Compute j array for Y, which aligns threshold x_t to y_j.
    #         x_0 + t = y_0 + j
    # ----------------------------------------------------------------------
    
    j = plrvs.x0 + t - plrvs.y0
    
    # ----------------------------------------------------------------------
    # Step 5: Identify the four cases. See individual comments for details.
    # ----------------------------------------------------------------------
    
    # Case 1: t = -1 --> NP threshold = -infinity --> beta = 0
    index_case1 = np.searchsorted(t, -1, side = 'right') # all t behind index_case1 are <= -1
    
    # Case 2: j <= -1 --> NP threshold < y_0 --> beta = 0
    index_case2 = np.searchsorted(j, -1, side = 'right') # all j behind index_case2 are <= -1
    
    # handle Case 1 and 2 jointly.
    index_case12 = max(index_case1, index_case2)
    betas[:index_case12] = 0 
    
    # Case 4: j > l - 1--> threshold > y_{l-1} --> beta = 1 - Pr[Y = inf] 
    index_case4 = np.searchsorted(j, l - 1, side = 'right') # all j behind index_case4 are <= l - 1
    betas[index_case4:] = 1 - plrvs.infinity_mass_Y
    
    # Case 3: y_0 <= threshold <= y_{l-1}
    slice_case3 = np.index_exp[index_case12:index_case4] # indices of the thresholds in Case 3
    t_case3 = t[slice_case3] # grab Case 3 threshold indices for Domain_X
    j_case3 = j[slice_case3] # grab Case 3 threshold indices for Domain_Y
    
    # Note: use cdf_X[t_case3 + 1] because we want Pr[X <= x_{t_case3}]
    gammas = (alphas[slice_case3] - ccdf_X[t_case3 + 1]) / plrvs.pmf_X[t_case3]
    betas[slice_case3] = cdf_Y[j_case3] - gammas * plrvs.pmf_Y[j_case3]
    
    # reverse betas to align with input ascending alphas 
    return betas[::-1]

def _inverse_tradeoff_function(
    plrvs: PLRVs,
    betas: np.ndarray,
) -> np.ndarray:
    '''
    Computes the False Positive Rates (FPR) for a set of 
    False Negative Rates (FNR), given the PLRVs X and Y. 
    
    This function corresponds to the inverse of Algorithm 1 in 
    Kulynych et al. For each value of FNR,  we find the 
    Neyman-Pearson threshold in Domain_Y, along with
    the Neyman-Pearson coin flip probability, then compute 
    the corresponding FNR in Domain_Y.
    
    Args:
        plrvs:
            A PLRVs instance with attributes:
            - pmf_X, pmf_Y: Probability mass functions for X, Y 
              over finite domains.
            - x0, y0: Offsets for the domain points.
            - is_symmetric: Boolean indicating whether T(P,Q) = T(Q,P).

        betas: np.ndarray
            Ascending array of FNR values for which to compute FPR.

    Returns:
        alphas: np.ndarray
            Array of FPR values corresponding to the input betas.
    '''
    
    # Preallocate alpha array for output
    alphas = np.empty_like(betas)
    
    # Length of domains of X and Y (excluding masses at +- inf)
    k = len(plrvs.pmf_X)
    l = len(plrvs.pmf_Y)
    
    # ----------------------------------------------------------------------
    # Step 1: Construct CDF of Y so that cdf_Y[i] = Pr[Y <= y_i],
    #         for 0 <= i <= l, 
    #         where cdf_Y[l] = Pr[Y <= y_l] = Pr[Y <= ∞] = 1.
    #         The input pmf is over y_0, y_1, ..., y_{l-1}. 
    # ----------------------------------------------------------------------
    cdf_Y = np.empty(l + 1)
    cdf_Y[:-1] =  np.cumsum(plrvs.pmf_Y)
    cdf_Y[-1] = 1

    # ----------------------------------------------------------------------
    # Step 2: Construct compliment cdf of X so that ccdf_X[i] = Pr[X > x_i],
    #         for 0 <= i <= k - 1.
    # ----------------------------------------------------------------------
    ccdf_X = np.cumsum(plrvs.pmf_X[::-1])[::-1]
    ccdf_X = np.roll(ccdf_X, -1)
    ccdf_X[-1] = 0

    # ----------------------------------------------------------------------
    # Step 3: Compute t array, where t[i] represents the index in Y's domain
    #         of the NP threshold for FNR beta[i].
    #         t ∈ [0, l], with t = l meaning the threshold is ∞.
    # ----------------------------------------------------------------------
    
    # Search in cdf_Y for the first index where cdf_Y[index] >= beta.
    t = np.searchsorted(cdf_Y, betas, side='left')
    
    # ----------------------------------------------------------------------
    # Step 4: Compute j array for X, which aligns threshold y_t to x_j.
    #         y_0 + t = x_0 + j
    # ----------------------------------------------------------------------
    j = plrvs.y0 + t - plrvs.x0
    
    # ----------------------------------------------------------------------
    # Step 5: Identify the four cases. See individual comments for details.
    # ----------------------------------------------------------------------
    
    # Case 1: j < 0 --> NP threshold < x_0 --> alpha = 1 - Pr[X = -inf] 
    index_case1 = np.searchsorted(j, -1, side = 'right') # all j behind index_case1 are <= -1
    alphas[:index_case1] = 1 - plrvs.minus_infinity_mass_X
    
    # Case 3: j > k - 1 --> threshold > x_k --> alpha = 0
    index_case3 = np.searchsorted(j, k - 1, side = 'right') # all j behind index_case3 are <= k - 1
    
    # Case 4: t = l --> threshold = infinity --> alpha = 0
    index_case4 = np.searchsorted(t, l - 1, side = 'right') # all t behind index_case4 are <= l - 1
    
    # handle Case 3 and 4 jointly.
    index_case34 = min(index_case3, index_case4)
    alphas[index_case34:] = 0 
    
    # Case 2: x_0 <= threshold <= x_{l-1}
    slice_case2 = np.index_exp[index_case1:index_case34] # indices of the thresholds in Case 3
    t_case2 = t[slice_case2] # grab Case 2 threshold indices for Domain_Y
    j_case2 = j[slice_case2] # grab Case 2 threshold indices for Domain_X
    gammas = (cdf_Y[t_case2] - betas[slice_case2]) / plrvs.pmf_Y[t_case2]
    alphas[slice_case2] = ccdf_X[j_case2] + gammas * plrvs.pmf_X[j_case2]
    
    # Done! Return alphas
    return alphas


def _ensure_array(x):
    is_scalar = isinstance(x, (int, float))
    if is_scalar:
        return np.asarray([x]), is_scalar
    return np.asarray(x), False


def get_beta(
    plrvs: PLRVs,
    alphas: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Computes the FNR values corresponding to the input FPR values (alphas).
    This function applies the following logic:
    
        1. Ensures the input alphas are in a NumPy array (float or 1D array).
           If they are not sorted, sort them in ascending order.
        
        2. Check whether the PLRVs object is symmetric. If so, call
           the direct tradeoff function T(P,Q) (i.e., _tradeoff_function).
        
        3. If not symmetric, apply the symmetrization logic of 
           Definition F.1 from Dong et al. (see file docstring for details)
           
           a. Compute alpha_bar = Pr[X > 0] and f_alpha_bar = Pr[Y <= 0].
           
           b. Compare alpha_bar and f_alpha_bar:
           
              - If alpha_bar <= f_alpha_bar, the symmetrized tradeoff curve 
                includes a region in which the curve is linear and equal to
                alpha_bar + f_alpha_bar - alpha. Outside that region,
                the curve follows _tradeoff_function on one side and
                _inverse_tradeoff_function on the other.
              
              - Otherwise, the symmetrized curve at each alpha is given by
                the maximum of T(P,Q)(alpha) and T(Q,P)(alpha).
        
        4. Returns the FNR values (betas) in the same shape (scalar or array) 
           as the input alphas.
    
    Args:
        plrvs:
            A PLRVs instance with attributes:
            - pmf_X, pmf_Y: Probability mass functions for X, Y 
              over finite domains.
            - x0, y0: Offsets for the domain points.
            - is_symmetric: Boolean indicating whether T(P,Q) = T(Q,P).
        
        alphas:
            A float or array of floats representing the False Positive Rates 
            for which we need to compute the False Negative Rates.

    Returns:
        A float (if alphas was a single float) or a NumPy array (if alphas 
        was an array) of False Negative Rates corresponding to the input 
        alphas.
    """
    
    # Convert alphas to array; check if input was scalar
    alphas, is_scalar = _ensure_array(alphas) 
    
    # Ensure ascending order
    is_sorted = np.all(np.diff(alphas) >= 0)
    if not is_sorted:
        sort_idx = np.argsort(alphas)
        alphas = alphas[sort_idx]
        undo_sort_indices = np.argsort(sort_idx)
    
    # Symmetric means T(P,Q) = T(Q,P). No need to symmetrize
    if plrvs.is_symmetric:
        output =  _tradeoff_function(plrvs, alphas)
    
    # See module docstring for explanation
    else:
        
        # ----------------------------------------------------------------------
        # Step a: Compute alpha_bar, f_alpha_bar
        #         alpha_bar = Pr[X > 0] = sum of pmf_X for all x > 0
        #         f_alpha_bar = Pr[Y <= 0] = sum of pmf_Y for all y <= 0
        #      
        #         since pmf_X[i] = Pr[X = x_0 + i], it follows that
        #         pmf_X[-x0 + 1] = Pr[X = 1], hence we sum from there onwards.
        #         Similar logic holds for pmf_Y. 
        # ----------------------------------------------------------------------
        
        alpha_bar = np.sum(plrvs.pmf_X[-plrvs.x0 + 1:])
        f_alpha_bar = np.sum(plrvs.pmf_Y[:-plrvs.y0 + 1])
        
        # ----------------------------------------------------------------------
        # Step b: Apply Definition F.1
        # ----------------------------------------------------------------------
        if alpha_bar <= f_alpha_bar:
            
            # Region partitioning input alphas based on alpha_bar, f_alpha_bar
            alpha_bar_index = np.searchsorted(alphas, alpha_bar, side='right') # all alphas behind index are <= alpha_bar
            f_alpha_bar_index = np.searchsorted(alphas, f_alpha_bar, side='right') # all alphas behind index are <= f_alpha_bar
            
            output = np.empty_like(alphas)
            
            # 1) Evaluate the symmeterized tradeoff curve for alpha < alpha_bar
            output[:alpha_bar_index] = _tradeoff_function(plrvs, alphas[:alpha_bar_index])
            
            # 2) Evaluate the symmeterized tradeoff curve in linear region alpha_bar <= alpha <= f_alpha_bar
            output[alpha_bar_index:f_alpha_bar_index] = alpha_bar + f_alpha_bar - alphas[alpha_bar_index:f_alpha_bar_index]
            
            # 3) Evaluate the symmeterized tradeoff curve for f_alpha_bar < alpha
            output[f_alpha_bar_index:] = _inverse_tradeoff_function(plrvs, alphas[f_alpha_bar_index:])

        else:
            # When alpha_bar > f_alpha_bar, symmetrization = maximum of
            # T(P,Q)(alpha) and T(Q,P)(alpha).
            tradeoff_arr = _tradeoff_function(plrvs, alphas)
            inverse_tradeoff_arr = _inverse_tradeoff_function(plrvs, alphas)
            output = np.maximum( tradeoff_arr, inverse_tradeoff_arr )
    
    # Output item if input alphas was a scalar
    if is_scalar:
        return output.item()
    
    # If input array alphas were not sorted, undo the sort
    if not is_sorted:
        output = output[undo_sort_indices]
        
    # return numpy array
    return output