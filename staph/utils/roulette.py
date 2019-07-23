import numpy as np
from typing import List
from numba import njit


@njit(cache=False)
def roulette(prop_array: List[float]) -> int:
    """Perform roulette selection.

    For a given array, perform roulette selection and return the selected index.

    Parameters
    ----------
    prop_array
        Array of propensities.

    Returns
    -------
    ind
        The index of the reaction to fire.
    """
    nprop = len(prop_array)
    prop_total = np.sum(prop_array)
    r = np.random.rand()
    roulette_sum = np.cumsum(prop_array) / prop_total
    for ind in range(nprop):
        if r < roulette_sum[ind]:
            break
    return ind


@njit(cache=False)
def simple_system(seed: int = 0) -> np.ndarray:
    """Run roulette after settind seed.

    Sets seed and calls roulette within a for loop with the expectation that
    each iteration of loop generates the same index. Indices collected and
    returned in `choices`.

    Paramters
    ---------
    seed
        Seed to set before calling roulette.

    Returns
    -------
    choices
        The array of choices picked for the given seed.
    
    Notes
    -----
    """
    n = 10
    x = np.ones(n)
    choices = np.zeros(2 * n)
    for ind in range(len(choices)):
        np.random.seed(seed)
        choices[ind] = roulette(x)
    return choices
