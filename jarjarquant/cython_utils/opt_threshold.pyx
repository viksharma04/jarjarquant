# opt_threshold.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np

cdef double EPS = 1.e-30

cpdef tuple optimize_threshold_cython(np.ndarray[double, ndim=1] work_signal,
                                      np.ndarray[double, ndim=1] work_return,
                                      int min_kept):
    """
    Cython version of the cumulative update loop.
    Assumes work_signal and work_return are preprocessed (e.g., sorted, NaNs removed).
    Returns:
        best_high_index, best_low_index, best_high_pf, best_low_pf
    """
    cdef Py_ssize_t n = work_signal.shape[0]
    if n == 0:
        raise ValueError("Input arrays must have at least one element.")
    
    cdef Py_ssize_t i
    cdef double r

    # Value Accumulators (Sum of Returns)
    cdef double win_above = 0.0, lose_above = 0.0, win_below = 0.0, lose_below = 0.0

    # Count Accumulators (Number of Trades)
    cdef int count_win_above = 0, count_lose_above = 0, count_win_below = 0, count_lose_below = 0

    cdef double current_pf, best_high_pf, best_low_pf
    cdef double best_high_acc = 0.0, best_low_acc = 0.0
    cdef Py_ssize_t best_high_index = 0
    cdef Py_ssize_t best_low_index = n - 1

    # Compute initial accumulators for the entire dataset ("above" group)
    for i in range(n):
        r = work_return[i]
        if r > 0.0:
            win_above += r
            count_win_above += 1
        else:
            lose_above -= r  # subtract negative to add its magnitude
            count_lose_above += 1

    best_high_pf = win_above / (lose_above + EPS)
    if (count_win_above + count_lose_above) > 0:
        best_high_acc = <double>count_win_above / (count_win_above + count_lose_above)

    best_low_pf = -1.0
    best_low_acc = 0.0

    # Loop over possible thresholds, updating accumulators
    for i in range(n - 1):
        r = work_return[i]

        # Remove r from the "above" group
        if r > 0.0:
            win_above -= r
            count_win_above -= 1
        else:
            lose_above += r
            count_lose_above -= 1

        # Add r to the "below" group
        if r > 0.0:
            lose_below += r
            count_lose_below += 1 # Count as a loss for Short
        else:
            win_below -= r
            count_win_below += 1  # Count as a win for Short

        # Skip if the next signal is identical (to avoid redundant thresholds)
        if work_signal[i + 1] == work_signal[i]:
            continue

        # Check profit factor for "above" group if sufficient cases remain
        if (n - i - 1) >= min_kept:
            current_pf = win_above / (lose_above + EPS)
            if current_pf > best_high_pf:
                best_high_pf = current_pf
                best_high_index = i + 1

                # Snapshot the accuracy at this best PF
                if (count_win_above + count_lose_above) > 0:
                    best_high_acc = <double>count_win_above / (count_win_above + count_lose_above)
                else:
                    best_high_acc = 0.0

        # Check profit factor for "below" group if sufficient cases remain
        if (i + 1) >= min_kept:
            current_pf = win_below / (lose_below + EPS)
            if current_pf > best_low_pf:
                best_low_pf = current_pf
                best_low_index = i + 1

                # Snapshot the accuracy at this best PF
                if (count_win_below + count_lose_below) > 0:
                    best_low_acc = <double>count_win_below / (count_win_below + count_lose_below)
                else:
                    best_low_acc = 0.0

    return best_high_index, best_low_index, best_high_pf, best_low_pf, best_high_acc, best_low_acc
