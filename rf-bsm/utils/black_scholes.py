import numpy as np
import scipy.stats as si


n = si.norm.pdf
N = si.norm.cdf


# See Paul Wilmot Introduces Quantitative Finance pg 177 - 181 for derivations and context.
def d(i, A, B, r, sigma, T):
    '''
    Helper function for Black-Scholes style formulas
    '''
    d_2 = (np.log(A / B) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if i == 2:
        return d_2
    else:
        return d_2 + (sigma * np.sqrt(T))


# black scholes analytic pricer.
def black_scholes_put(S, K, T, r, sigma):
    """
    PUT Option:

    S: spot price
    K: strike price
    T: time to maturity
    r: interest rate
    sigma: volatility of underlying asset

    """
    d1 = d(1, S, K, r, sigma, T)
    d2 = d(2, S, K, r, sigma, T)

    # check conditions.
    natm = (S != K)   # not at the money
    natx = (T != 0)   # not at expiry
    condition = natm | natx

    # avoid numerical instability at expiry at the money '0/0'
    put = np.where(condition, (K * np.exp(-r * T) * N(-d2) - S * N(-d1)), 0)
    return put
