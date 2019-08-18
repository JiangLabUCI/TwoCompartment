import numpy as np
import multiprocessing as mp
from timeit import default_timer as timer
from functools import partial
from numba import njit
from typing import List, Tuple, Any, Union
from .data import get_singh_data, get_b1d2, calc_for_map
from .tau_twocomp import tau_twocomp_carrier
from .dev import compute_deviance, get_bF_bX, get_consts_bX, transform_x
from collections import Counter


@njit(cache=False)
def get_best_thresh(
    final_loads: np.ndarray,
) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """Best threshold search.

    Return the threshold that provides the best deviance for the given
    `final_loads` array. Checks the unique values in `final_loads` + 1.

    Parameters
    ----------
    final_loads
        The bacterial loads at the end of the simulations.

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
    thresh_array = np.sort(np.unique(final_loads)) + 1
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
    x: List[float],
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
    t_type: str = None,
) -> Union[float, Tuple[List, List, List, List]]:
    """Threshold wrapper objective function that returns total deviance.

    Uses the `get_b1d2` function to compute b1 and d2 from r3, Imax, b2 and d1.
    Calls `calc_devlist_carrier` to compute the objective function value.

    Parameters
    ----------
    x
        Input array to be optimized. Consists of rate constants b2 and d1.
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
    t_type
        Tranformation type to apply to b2.

    Returns
    -------
    objval
        Objective value used for optimization, which is the deviance.
    """

    # Get data
    h0, _, _, _, A, H0 = get_singh_data()

    # Return high objective values for negative rates.
    if (x[0] < 0) or (x[1] < 0):
        return 3000

    # Compute b1 and d2 for simulation.
    b2, d1 = transform_x(x, t_type=t_type)
    b1, d2 = get_b1d2(b2=b2, d1=d1, r3=r3, r3Imax=r3 * Imax)
    rates = np.array([r1, r2, b1, b2, d1, d2])
    imax = Imax * A

    seeds = np.random.randint(low=0, high=1e5, size=nrep)
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
            arg_list.append((init_load, rates, imax, nstep, seeds[ind2], 6.0, True))
        # Run parallel simulation
        partial_func = partial(calc_for_map, func=tau_twocomp_carrier)
        results = pool.map(partial_func, arg_list)
        for ind2, r in enumerate(results):
            # print("ind2 : ", ind2)
            extflag[ind2] = r[0]
            endt[ind2] = r[1]
            status[ind2] = r[4]
            final_loads[ind1, ind2] = np.sum(r[2][-1, :])

        extflags.append(extflag)
        endts.append(endt)
        statuses.append(status)
        this_status = status
        print(f"Seed = {seed}, status counts : ", Counter(this_status))
        if np.any(this_status == 0):
            print("Zero status detected, rates, dose =  ", rates, h0[ind1])

        # p_inf = prob (H(t) + I(t) > thresh)

    best_thresh, best_dev, devs, all_devs = get_best_thresh(final_loads=final_loads)
    best_dev_index = np.argmin(devs)
    p_res = np.zeros((npts))
    for ind1 in range(npts):
        p_res[ind1] = np.mean(final_loads[ind1, :] >= best_thresh)
    print(f"Best thresh is : {best_thresh:.8e}")
    print(f"Which gives best dev of : {best_dev:.4f}")
    print(f"Which is sum of :  {all_devs[best_dev_index, :]}")
    print(f"p_res is : {p_res}")

    if obj_flag:
        objval = best_dev
        print("Objective is : ", objval)
        print("------------------------------------------")
        return objval
    else:
        return (
            extflags,
            endts,
            statuses,
            best_thresh,
            best_dev,
            all_devs[best_dev_index, :].flatten(),
        )


def thresh_brute_min(
    filename="results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat",
    npts: int = 2,
    nrep: int = 10,
    seed: int = 0,
    desol_ind: List[float] = [0],
    nstep: int = 200_000,
    n_procs: int = 2,
    lims: dict = {"d1l": 0, "d1u": 5, "b2l": 0, "b2u": 5},
    nb2: int = 2,
    nd1: int = 2,
    t_type: str = None,
):
    """Identify best fit threshold.

    Brute search for b2 and d1 and identify the best fit threshold.

    Parameters
    ----------
    filename
        DE solution file name
    npts
        Number of doses to calculate deviance at
    nrep
        Number of simulations per (b1,d2) pair to calculate deviance
    seed
        Seed of the `NumPy` random generator, different from the seed
        of `numba`
    desol_ind
        Indices of the DE solutions to evaluate deviance
    nstep
        Maximum number of steps to run each simulation
    n_procs
        Number of parallel processes to evaluate the objective at.
    lims
        Dictionary of upper and lower limits of b2 and d1.
    nb2
        Number of b2 solutions to go over.
    nd1
        Number of d1 solutions to go over.
    t_type
        Tranformation type to apply to b2.
    """

    print("Seed is : ", seed)
    print("Nstep : ", nstep)
    print("Number of points is : ", npts)
    ndesol = len(desol_ind)  # number of DE solutions to investigate
    bFlist, bXlist = get_bF_bX(filename=filename, desol_ind=desol_ind)

    print("Best F values : ", bFlist)
    print("Best parameters : ", bXlist)
    # The suffix u denotes untransformed variables.
    b2listu = np.linspace(lims["b2l"], lims["b2u"], nb2)
    d1listu = np.linspace(lims["d1l"], lims["d1u"], nd1)
    all_statuses = []
    optim_objs = np.zeros(nb2 * nd1)
    optim_thresh = np.zeros(nb2 * nd1)
    all_devs = np.zeros((nb2 * nd1, npts))

    print("Creating pool with", n_procs, " processes\n")
    pool = mp.Pool(n_procs)
    print("pool = %s", pool)

    t0 = timer()
    for ind in range(ndesol):
        r1, r2, r3, Imax, modno = get_consts_bX(
            bXlist=bXlist, ind=ind, filename=filename
        )
        for ind1, b2u in enumerate(b2listu):
            for ind2, d1u in enumerate(d1listu):
                print(f"Starting loop for b2u={b2u:.2e}, d1u={d1u:.2f}")
                linear_ind = ind1 * nd1 + ind2
                retval = thresh_obj_wrapper(
                    x=[b2u, d1u],
                    r1=r1,
                    r2=r2,
                    r3=r3,
                    Imax=Imax,
                    npts=npts,
                    nrep=nrep,
                    nstep=nstep,
                    seed=seed,
                    pool=pool,
                    obj_flag=False,
                )
                all_statuses.append(retval[2])
                optim_thresh[linear_ind] = retval[3]
                optim_objs[linear_ind] = retval[4]
                all_devs[linear_ind, :] = retval[5]
    print("optim objs is : ", optim_objs)
    print(f"Best fit is : {np.min(optim_objs)}")
    t1 = timer()
    print("1 DE solution took : ", t1 - t0, "s")

    ind1 = 0
    ind2 = filename.find("_DE")
    jname = filename[ind1:ind2]
    solstr = "to".join([str(min(desol_ind)), str(max(desol_ind))])
    op_filename = (
        jname
        + "_"
        + str(nrep)
        + "rep"
        + str(seed)
        + "se"
        + "_"
        + solstr
        + "b2d1_1o5_cpu.npz"
    )
    print("Output filename : ", op_filename)

    with open(op_filename, "wb") as f:
        np.savez(
            f,
            seed=seed,
            desol_ind=desol_ind,
            bXlist=bXlist,
            bFlist=bFlist,
            nstep=nstep,
            optim_objs=optim_objs,
            modno=modno,
            t_type=t_type,
            lims=lims,
            b2listu=b2listu,
            d1listu=d1listu,
            all_statuses=all_statuses,
            all_devs=all_devs,
            nb2=nb2,
            nd1=nd1,
        )
