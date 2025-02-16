# bar_permute_cy.pyx

# Import NumPy for runtime support and its C API for type declarations.
import numpy as np
cimport numpy as cnp

def permute_cython(cnp.ndarray[cnp.double_t, ndim=2] basis_prices, 
                   cnp.ndarray[cnp.double_t, ndim=3] rel_prices):
    """
    Parameters:
      basis_prices: NumPy array with shape (n_markets, 4)
                    [columns: Open, High, Low, Close]
      rel_prices: NumPy array with shape (n_markets, n_steps, 4)
                  [columns: rel_open, rel_high, rel_low, rel_close]
                  
    Returns:
      permuted: NumPy array with shape (n_markets, n_steps+1, 4)
                containing the reconstructed OHLC data.
    """
    cdef int n_markets = basis_prices.shape[0]
    cdef int n_steps = rel_prices.shape[1]
    # Allocate output array: one extra row per market for the initial basis price.
    cdef cnp.ndarray[cnp.double_t, ndim=3] permuted = np.empty((n_markets, n_steps + 1, 4), dtype=np.double)
    
    cdef int i, j

    for i in range(n_markets):
        # Set the initial basis prices for market i.
        permuted[i, 0, 0] = basis_prices[i, 0]  # Open
        permuted[i, 0, 1] = basis_prices[i, 1]  # High
        permuted[i, 0, 2] = basis_prices[i, 2]  # Low
        permuted[i, 0, 3] = basis_prices[i, 3]  # Close
        
        for j in range(n_steps):
            # Compute new open from the previous close and the relative open change.
            permuted[i, j + 1, 0] = permuted[i, j, 3] + rel_prices[i, j, 0]
            # New high is new open plus relative high change.
            permuted[i, j + 1, 1] = permuted[i, j + 1, 0] + rel_prices[i, j, 1]
            # New low is new open plus relative low change.
            permuted[i, j + 1, 2] = permuted[i, j + 1, 0] + rel_prices[i, j, 2]
            # New close is new open plus relative close change.
            permuted[i, j + 1, 3] = permuted[i, j + 1, 0] + rel_prices[i, j, 3]
            
    return permuted

def permute_cython_single(cnp.ndarray[cnp.double_t, ndim=2] basis_prices, 
                          cnp.ndarray[cnp.double_t, ndim=3] rel_prices):
    """
    Parameters:
      basis_prices: NumPy array with shape (n_markets, 1)
                    [column: Close]
      rel_prices: NumPy array with shape (n_markets, n_steps, 1)
                  [column: rel_close]
                  
    Returns:
      permuted: NumPy array with shape (n_markets, n_steps+1, 1)
                containing the reconstructed Close data.
    """
    cdef int n_markets = basis_prices.shape[0]
    cdef int n_steps = rel_prices.shape[1]
    # Allocate output array: one extra row per market for the initial basis price.
    cdef cnp.ndarray[cnp.double_t, ndim=3] permuted = np.empty((n_markets, n_steps + 1, 1), dtype=np.double)
    
    cdef int i, j

    for i in range(n_markets):
        # Set the initial basis prices for market i.
        permuted[i, 0, 0] = basis_prices[i, 0]  # Close
        
        for j in range(n_steps):
            # Compute new close from the previous close and the relative close change.
            permuted[i, j + 1, 0] = permuted[i, j, 0] + rel_prices[i, j, 0]
            
    return permuted
