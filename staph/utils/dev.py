import numpy as np
import scipy.io as sio
import multiprocessing as mp
from timeit import default_timer as timer
from scipy.optimize import minimize
from functools import partial
from numba import njit
from .data import get_singh_data, get_b1d2, calc_for_map
from .tau_twocomp import tau_twocomp_carrier
from typing import List, Tuple, Any, Union


@njit(cache=True)
def compute_deviance(p_inf: float, dose_index: int) -> float:
    """Compute deviance from Singh data.

    For a given infection probability and dose index, compute the deviance
    from Singh data.

    Parameters
    ----------
    p_inf
        Infection probability to compute deviance of.
    dose_index
        The index of the dose from the Singh data. Integer in [0, 5].

    Returns
    -------
    dev
        Deviance of the given infection probability for that dose index.

    Notes
    -----
    The formula for deviance is:
    -2 * (norig[dose_index] * np.log(p_inf * ntot / norig[dose_index])
    + (ntot - norig[dose_index])
    * np.log((1 - p_inf) * ntot / (ntot - norig[dose_index])))
    
    For the case when all 20 people show response, second term drops off and
    deviance is : -2 * norig[dose_index] * np.log(p_inf) = 
    -2 * ntot * np.log(p_inf).

    When p_inf is predicted to be 1, set it to 0.99 and use the main formula.
    This condition is checked only for those cases where the number of observed
    responses is less than 20.

    When p_inf is predicted to be 0, set it to 0.01 and use the main formula.
    Also add 1000 to the deviance for some forcing. This will not affect final
    results much because this condition (p_inf = 0) will not be met for good
    solutions obtained in the later stages of optimization.
    """

    # Import data
    _, norig, ntot, _, _, _ = get_singh_data()

    if norig[dose_index] == ntot:
        dev = -2 * ntot * np.log(p_inf)
    elif p_inf == 1:
        p_inf = 0.99
        dev = -2 * (
            norig[dose_index] * np.log(p_inf * ntot / norig[dose_index])
            + (ntot - norig[dose_index])
            * np.log((1 - p_inf) * ntot / (ntot - norig[dose_index]))
        )
        p_inf = 1
    elif p_inf == 0:
        p_inf = 0.01
        dev = -2 * (
            norig[dose_index] * np.log(p_inf * ntot / norig[dose_index])
            + (ntot - norig[dose_index])
            * np.log((1 - p_inf) * ntot / (ntot - norig[dose_index]))
        )
        dev = dev
        p_inf = 0
    else:
        dev = -2 * (
            norig[dose_index] * np.log(p_inf * ntot / norig[dose_index])
            + (ntot - norig[dose_index])
            * np.log((1 - p_inf) * ntot / (ntot - norig[dose_index]))
        )

    return dev


def carrier_obj_wrapper(
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
) -> Union[float, Tuple[List, List, List, List]]:
    """Wrapper objective function that returns total deviance.
    
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
    b2, d1 = transform_x(x)
    b1, d2 = get_b1d2(b2=b2, d1=d1, r3=r3, r3Imax=r3 * Imax)
    rates = np.array([r1, r2, b1, b2, d1, d2])
    np.random.seed(seed)
    seeds = np.random.randint(low=0, high=1e5, size=nrep)
    p_inf = [0 for choice in range(npts)]
    devs = []
    extflags = []
    endts = []
    statuses = []

    # If any of supplied rates are negative.
    print("Rates are :  ", rates)
    if np.any(np.array(rates) < 0):
        return 3.2e3
    arg_list = []
    for choice in range(npts):
        # Holders for the return values of each dose
        extflag = np.zeros(nrep)
        endt = np.zeros(nrep)
        status = np.zeros(nrep)

        # Assemble the argument list for multiprocessing.
        arg_list = []
        for ind1 in range(nrep):
            init_load = np.array([H0[choice]], dtype=np.int32)
            arg_list.append(
                (init_load, rates, Imax * A, nstep, seeds[ind1], 6.0, False)
            )
        # Run parallel simulation
        partial_func = partial(calc_for_map, func=tau_twocomp_carrier)
        results = pool.map(partial_func, arg_list)
        for ind1, r in enumerate(results):
            # print("ind1 : ", ind1)
            extflag[ind1] = r[0]
            endt[ind1] = r[1]
            status[ind1] = r[4]
        # p_inf = prob (I(t) > Imax)
        # Population exceeds Imax if I(t) > Imax (status == 3) or
        # if I(t) overflows.
        # 3 : Succesful completion, terminated when I(t) > Imax.
        # 4 : Succesful completion, curI overflow.
        p_inf[choice] = np.mean((status == 3) + (status == 4) + (status == 5))
        dev = compute_deviance(p_inf=p_inf[choice], dose_index=choice)
        devs.append(dev)

        # Early exit criterion, exit if objective has exceeded preset
        # threshold. Higher doses will have p_inf = 1, resulting in
        # even higher objectives.

        if np.sum(devs) > 150:
            print("Stopping early.")
            objval = 150
            return objval

        extflags.append(extflag)
        endts.append(endt)
        statuses.append(status)
        this_status = status
        print(
            f"Seed = {seed}, pinf = {p_inf[choice]:.3f}, dev = {dev:.3f},  status histogram : ",
            np.histogram(
                this_status, bins=np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6]) - 0.1
            )[0],
        )
        if np.any(this_status == 0):
            print("Zero status detected, rates, dose =  ", rates, h0[choice])

    if obj_flag:
        objval = np.sum(devs)
        print("Objective is : ", objval)
        print("------------------------------------------")
        return objval
    else:
        return (devs, extflags, endts, statuses)


def transform_x(x: List[float]) -> Tuple[float, float]:
    """Transform `x` to `b2` and `d1`.

    Transform `x` to `b2` and `d1`, used by `carrier_obj_wrapper`.

    Parameters
    ----------
    x
        The vector containing values used for transformation.
    
    Returns
    -------
    b2 : float
        Second order birth rate of twocomp model (stochastic,
        units = 1 / (bacteria * day)).
    d1 : float
        First order death rate of twocomp model. (stochastic,
        units = 1 / day)

    Notes
    -----
    Use this function to centralize the transform to avoid errors.
    """
    b2 = 10 ** (-x[0])
    d1 = x[1]
    return b2, d1


def compute_devs_min(
    filename="results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat",
    npts: int = 2,
    nrep: int = 10,
    seed: int = 0,
    desol_ind: List[int] = [0],
    nstep: int = 200_000,
    method: str = "Powell",
    niter: int = 4,
    problem_type: int = 1,
    n_procs: int = 2,
):
    """Optimize for deviances of the DEMC solutions.

    Runs the optimization and saves the output.

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
    method
        Name of the optimizer to use
    niter
        Maximum number of iterations of the minimizer
    problem_type
        If 1, optimize for b2 and d1. If 2, optimize only for d1 with b2 = 0.
    n_procs
        Number of parallel processes to evaluate the objective at.
    """

    print("Seed is : ", seed)
    print("Nstep : ", nstep)
    print("Number of points is : ", npts)
    ndesol = len(desol_ind)  # number of DE solutions to investigate
    bFlist, bXlist = get_bF_bX(filename=filename, desol_ind=desol_ind)
    print("Best F values : ", bFlist)
    print("Best parameters : ", bXlist)
    optim_objs = []

    if problem_type == 1:
        minimization_objective = carrier_obj_wrapper
        initial_guess = (2.5, 30)
        print("Initial_guess is : ", initial_guess)

    print("Creating pool with", n_procs, " processes\n")
    pool = mp.Pool(n_procs)
    print("pool = %s", pool)

    t0 = timer()
    for ind in range(ndesol):
        r1, r2, r3, Imax, modno = get_consts_bX(bXlist=bXlist, ind=ind, filename=filename)
        t1 = timer()
        min_obj = minimize(
            minimization_objective,
            initial_guess,
            args=(r1, r2, r3, Imax, npts, nrep, nstep, seed, pool, True),
            options={"maxfev": niter},
            method=method,
        )
        optim_objs.append(min_obj)
        print(min_obj)
        t2 = timer()
        print("1 DE solution took : ", t2 - t1, "s")
    print("Totally took : ", t2 - t0, "s")
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
        + str(niter)
        + "ite"
        + "_"
        + method
        + "_"
        + solstr
        + "b2d1_1o3_cpu.npz"
    )
    print("Output filename : ", op_filename)

    with open(op_filename, "wb") as f:
        np.savez(
            f,
            seed=seed,
            desol_ind=desol_ind,
            bXlist=bXlist,
            bFlist=bFlist,
            niter=niter,
            method=method,
            nstep=nstep,
            optim_objs=optim_objs,
            modno=modno,
        )


def get_consts_bX(
    bXlist: np.ndarray, ind: int, filename: str
) -> Tuple[float, float, float, float, int]:
    """Pre-process bX and return kinetic constants.

    From bXlist, extract the kinetic constants r1, r2, r3 and Imax.

    Parameters
    ----------
    bXlist
        The array of solutions that provide the best objective values.
    ind
        The index of the solution to return constants for.
    filename
        The name of the file from which the indices were extracted.
    
    Returns
    -------
    r1
        Rate constant with units (/day).
    r2
        Rate constant with units (/day).
    r3
        Rate constant with units (cm^2/(bacteria * day)).
    Imax
        Carrying capacity with units (bacteria/cm^2).
    modno
        Model number. 3 means Imax was predicted in DEMC. 
        6 means r3Imax was predicted in DEMC.

    Notes
    -----
    If the filename has `3mod` in it, Imax was predicted as the 4th element.
    If it has `6mod` in it, r3Imax was predicted as the 4th element and Imax
    has to be computed from its value.
    """
    r1, r2, r3 = bXlist[ind, 0], bXlist[ind, 1], bXlist[ind, 2]
    modno = int(filename[filename.find("mod") - 1])
    if modno == 3:
        Imax = bXlist[ind, 3]
    elif modno == 6:
        Imax = bXlist[ind, 3] / r3
        print("bXlist, r3 and Imax are : ", bXlist[ind, 3], r3, Imax)
    print("r3 * Imax is : ", r3 * Imax)

    return r1, r2, r3, Imax, modno


def get_bF_bX(
    filename: str = "results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat",
    desol_ind: List[int] = [0],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the best F and X values from DEMC solutions.
    
    Read the solutions present in `filename` and extract the best objective
    values and the solutions giving best objective values to return.

    Parameters
    ----------
    filename
        The file containing the DEMC solutions.
    desol_ind
        The indexes of the DE solutions to return objective and solutions at.
    
    Returns
    -------
    bFlist
        The best objective values.
    bXlist
        The solutions that provide the best objective values.
    """
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
    return bFlist, bXlist
