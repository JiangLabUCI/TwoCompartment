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


@njit(cache=False)
def get_best_thresh(
    final_loads: np.ndarray, low: int = 10, high: int = 20
) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """Brute force best threshold search.

    Return the threshold that provides the best deviance for the given 
    `final_loads` array. Checks every value in [`low`, `high`).

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
    devs
        The list of (total) deviances for each threshold.
    all_devs
        The list of deviances for each point for each threshold.
    """
    npts = final_loads.shape[0]
    thresh_array = np.arange(low, high)
    nthresh = thresh_array.shape[0]
    p_inf = np.zeros(npts)
    devs = np.zeros(thresh_array.shape[0])
    all_devs = np.zeros((thresh_array.shape[0], npts))
    for ind1 in range(npts):
        for ind2 in range(nthresh):
            this_thresh = thresh_array[ind2]
            p_inf[ind1] = np.mean(final_loads[ind1, :] >= this_thresh)
            all_devs[ind2, ind1] = compute_deviance(p_inf=p_inf[ind1], dose_index=ind1)
            devs[ind2] += all_devs[ind2, ind1]
    best_dev_index = np.argmin(devs)
    best_thresh = thresh_array[best_dev_index]
    best_dev = devs[best_dev_index]

    return best_thresh, best_dev, devs, all_devs


def thresh_obj_wrapper(
    r1: float,
    r2: float,
    r3: float,
    Imax: float,
    npts: int,
    nrep: int,
    nstep: int,
    seed: int,
    pool: Any,
    obj_flag: bool,
) -> Union[float, Tuple[List, List, List, List]]:
    """Threshold wrapper objective function that returns total deviance.
    
    Uses the `get_b1d2` function to compute b1 and d2 from r3, Imax, b2 and d1.
    Calls `calc_devlist_carrier` to compute the objective function value.

    Parameters
    ----------
    r1
        Rate constant with units (/day).
    r2
        Rate constant with units (/day).
    r3
        Rate constant with units (cm^2/(bacteria * day)).
    Imax
        Carrying capacity with units (bacteria/cm^2).
    npts
        Number of dose values to evaluate objective at. Only evaluating at
        all 6 doses matters. Lower `npts` are for testing purposes.
    nstep
        Number of steps to run the simulation for.
    seed
        Seed of the random number generator.
    pool : multiprocessing_pool
        Pool object used to evaluate the objectives in parallel.
    obj_flag
        If `True`, return only objective value. Else return Tuple with
        devs, extflags, endts and statuses.

    Returns
    -------
    objval
        Objective value used for optimization, which is the deviance.
    """

    # Get data
    h0, _, _, _, A, H0 = get_singh_data()

    # Set b2 = d1 = 0. Compute b1 and d2 for simulation.
    b2 = 0
    d1 = 0
    b1, d2 = get_b1d2(b2=b2, d1=d1, r3=r3, r3Imax=r3 * Imax)
    rates = np.array([r1, r2, b1, b2, d1, d2])
    imax = Imax * A

    seeds = np.random.randint(low=0, high=1e5, size=nrep)
    p_inf = [0 for choice in range(npts)]
    devs = []
    extflags = []
    endts = []
    statuses = []
    final_loads = np.zeros([npts, nrep])

    # If any of supplied rates are negative.
    print("Rates are :  ", rates)
    if np.any(np.array(rates) < 0):
        return 3.2e3
    arg_list = []

    # Retrieve final populations
    for ind1 in range(npts):
        # Holders for the return values of each dose
        extflag = np.zeros(nrep)
        endt = np.zeros(nrep)
        status = np.zeros(nrep)

        # Assemble the argument list for multiprocessing.
        arg_list = []
        for ind2 in range(nrep):
            init_load = np.array([H0[ind1]], dtype=np.int32)
            arg_list.append((init_load, rates, imax, nstep, seeds[ind2], 6.0, False))
        # Run parallel simulation
        partial_func = partial(calc_for_map, func=tau_twocomp_carrier)
        results = pool.map(partial_func, arg_list)
        for ind2, r in enumerate(results):
            # print("ind2 : ", ind2)
            extflag[ind2] = r[0]
            endt[ind2] = r[1]
            status[ind2] = r[4]
            final_loads[ind1, ind2] = 0

        extflags.append(extflag)
        endts.append(endt)
        statuses.append(status)
        this_status = status
        print(
            f"Seed = {seed}, pinf = {p_inf[ind1]:.3f}, dev = {dev:.3f},  status histogram : ",
            np.histogram(
                this_status, bins=np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6]) - 0.1
            )[0],
        )
        if np.any(this_status == 0):
            print("Zero status detected, rates, dose =  ", rates, h0[ind1])

        # p_inf = prob (H(t) + I(t) > thresh)

    best_thresh = get_best_thresh(
        final_loads=final_loads, low=H0[5] + 1, high=np.int(Imax * A)
    )
    print(best_thresh)

    if obj_flag:
        objval = np.sum(devs)
        print("Objective is : ", objval)
        print("------------------------------------------")
        return objval
    else:
        return (devs, extflags, endts, statuses)


def thresh_minimizer(
    filename="results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat",
    npts: int = 2,
    nrep: int = 10,
    seed: int = 0,
    desol_ind: List[float] = [0],
    nstep: int = 200_000,
    method: str = "sweep",
    niter: int = 4,
    problem_type: int = 1,
    n_procs: int = 2,
):
    """Identify threshold that provides best fit.
    """

    print("Seed is : ", seed)
    print("Nstep : ", nstep)
    print("Number of points is : ", npts)
    ndesol = len(desol_ind)  # number of DE solutions to investigate
    data = sio.loadmat(filename)
    Xlist = data["solset"][0][0][2]
    Flist = data["solset"][0][0][3]
    sortF = np.sort(np.unique(Flist.flatten()))
    bFlist = np.zeros([ndesol])
    bXlist = np.zeros([ndesol, Xlist.shape[2]])
    for ind in range(ndesol):
        bFlist[ind] = sortF[desol_ind[ind]]
        [ind1, ind2] = np.where(Flist == bFlist[ind])
        Xt = Xlist[ind1, ind2, :]
        Xt2 = np.unique(Xt, axis=0)
        bXlist[ind, :] = np.power(10, Xt2.flatten())
    print("Best F values : ", bFlist)
    print("Best parameters : ", bXlist)
    optim_objs = []

    print("Creating pool with", n_procs, " processes\n")
    pool = mp.Pool(n_procs)
    print("pool = %s", pool)

    t0 = timer()
    for ind in range(ndesol):
        r1, r2, r3 = bXlist[ind, 0], bXlist[ind, 1], bXlist[ind, 2]
        modno = int(filename[filename.find("mod") - 1])
        if modno == 3:
            Imax = bXlist[ind, 3]
        elif modno == 6:
            Imax = bXlist[ind, 3] / r3
            print("bXlist, r3 and Imax are : ", bXlist[ind, 3], r3, Imax)
        print("r3 * Imax is : ", r3 * Imax)
        t1 = timer()
        min_obj = sweep_thresh(
            r1=r1, r2=r2, r3=r3, Imax=Imax, npts=npts, nrep=nstep, seed=seed, pool=pool
        )
        # min_obj = minimize(
        #     minimization_objective,
        #     initial_guess,
        #     args=(r1, r2, r3, Imax, npts, nrep, nstep, seed, pool, True),
        #     options={"maxfev": niter},
        #     method=method,
        # )
        optim_objs.append(min_obj)
        print(min_obj)
        t2 = timer()
        print("1 DE solution took : ", t2 - t1, "s")
