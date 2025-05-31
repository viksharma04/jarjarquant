# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport log, erf, sqrt

# Declare accepted NumPy types for best performance
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cpdef np.ndarray[DTYPE_t, ndim=1] compute_trend_indicator(
    np.ndarray[DTYPE_t, ndim=1] close,
    np.ndarray[DTYPE_t, ndim=1] legendre,
    int lookback,
    int atr_length,
    np.ndarray[DTYPE_t, ndim=1] atr):
    """
    Computes a trend indicator based on a regression against Legendre polynomials.
    
    Parameters
    ----------
    close : 1D np.ndarray[float64]
        Array of closing prices.
    legendre : 1D np.ndarray[float64]
        Array of Legendre polynomial coefficients with shape (3, 1).
    lookback : int
        Lookback window for the trend regression.
    atr_length : int
        Lookback window used in ATR normalization (should exceed lookback).
    atr : 1D np.ndarray[float64]
        Array of ATR values (precomputed) corresponding to each price.
        
    Returns
    -------
    np.ndarray[float64]
        Indicator values (same length as close); undefined values (first indices)
        are set to 0.0.
    """
    cdef int n = close.shape[0]
    cdef int front_bad = lookback - 1
    if atr_length > front_bad:
        front_bad = atr_length
    if front_bad > n:
        front_bad = n

    # Allocate the output array
    cdef np.ndarray[DTYPE_t, ndim=1] output = np.empty(n, dtype=DTYPE)
    
    cdef int icase, k, window_start
    cdef double price, mean, dot_prod, rsq, yss, diff, pred, diff2, denom, k_factor, rsq_err

    # Set the undefined (initial) values to 0.0
    for icase in range(front_bad):
        output[icase] = 0.0

    # Loop over the valid cases
    for icase in range(front_bad, n):
        # Determine window start index (assume icase >= lookback-1)
        window_start = icase - lookback + 1
        mean = 0.0
        dot_prod = 0.0
        # Compute dot product and mean of log(prices) over the lookback window
        for k in range(lookback):
            price = log(close[window_start + k])
            mean += price
            dot_prod += price * legendre[k]

        mean /= lookback

        # Choose scaling factor k: normally lookback-1, but if lookback==2 use 2
        k_factor = lookback - 1
        if lookback == 2:
            k_factor = 2.0

        # Normalize by ATR (adding a tiny constant to avoid division by zero)
        denom = atr[icase] * k_factor
        output[icase] = dot_prod * 2.0 / (denom + 1e-60)

        # Compute R-square for the regression fit
        yss = 0.0
        rsq_err = 0.0
        for k in range(lookback):
            price = log(close[window_start + k])

            diff = price - mean
            yss += diff * diff
            
            pred = dot_prod * legendre[k]
            
            diff2 = diff - pred
            rsq_err += diff2 * diff2
            
        rsq = 1.0 - rsq_err / (yss + 1e-60)
        if rsq < 0.0:
            rsq = 0.0
        output[icase] *= rsq

    return output
