import numpy as np
import scipy.io as sio
import multiprocessing as mp
from timeit import default_timer as timer
from scipy.optimize import minimize
from functools import partial
from numba import njit
from typing import List, Tuple, Any, Union
from .data import get_singh_data, get_b1d2, calc_for_map
from .tau_twocomp import tau_twocomp_carrier
from .dev import compute_deviance


@njit(cache=True)
def get_best_thresh(final_loads: np.ndarray, low: int = 10, high: int = 20):
    """Brute force best threshold search.

    Return the threshold that provides the best deviance for the given 
    `final_loads` array.

    Parameters
    ----------
    final_loads
        The bacterial loads at the end of the simulations.
    low
        Lower limit of the threshold.
    high
        Upper limit of the threshold.
    
    Returns
    -------
    best_thresh
        The threshold that minimizes deviance.
    best_dev
        The lowest deviance found.

    """
    npts = final_loads.shape[0]
    thresh_array = np.arange(low, high)
    nthresh = thresh_array.shape[0]
    p_inf = np.zeros(npts)
    devs = np.zeros(thresh_array.shape[0])
    for ind1 in range(npts):
        for ind2 in range(nthresh):
            this_thresh = thresh_array[ind2]
            p_inf[ind1] = np.mean(final_loads[ind1, :] >= this_thresh)
            devs[ind2] += compute_deviance(p_inf=p_inf[ind1], dose_index=ind1)
    best_dev_index = np.argmin(devs)
    best_thresh = thresh_array[best_dev_index]
    best_dev = devs[best_dev_index]

    return best_thresh, best_dev

