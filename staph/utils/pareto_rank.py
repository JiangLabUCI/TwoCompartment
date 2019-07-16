import numpy as np
from numba import njit


@njit(cache=True)
def dominated(X, x):
    """dominated(X,x)

    Check if x is dominated (better/lower objective) by any row of X.

    Return 1 if dominated, 0 otherwise.

    Parameters
    ----------
    X : 2D numpy.ndarray of ints or floats
        Rows are elements and columns are objective values.
    x : 1D numpy.ndarray of ints or floats
        Element to flag as dominated or non-dominated.

    Returns
    -------
    dom_flag : int (0 or 1)
        1 if `x` is dominated by any element in `X`. 0 otherwise.
    """
    return np.any(np.sum(X < x, axis=1) == len(x))


@njit(cache=True)
def get_pareto_ranks(X):
    """get_pareto_ranks(X)

    Pareto-rank elements of X.

    Return list of ranks in same order as input.

    Parameters
    ----------
    X : 2D numpy.ndarray of ints or floats
        Rows are elements and columns are objective values.

    Returns
    -------
    ranks : list of int
        Ranks from 1 to however many there are.
    """
    N = X.shape[0]
    non_dom_mask = np.zeros(N)
    init_max = np.max(X) + 1
    ranks = np.zeros(N)
    rankval = 1
    while np.any(ranks == 0):
        for ind in range(N):
            domted = dominated(X, X[ind, :])
            if not (domted):
                non_dom_mask[ind] = 1
        non_dom_inds = np.where(non_dom_mask == 1)
        for ind in range(len(non_dom_inds)):
            X[non_dom_inds[ind], :] = init_max
            ranks[non_dom_inds[ind]] = rankval
            non_dom_mask[non_dom_inds[ind]] = 0
        rankval += 1

    return ranks
