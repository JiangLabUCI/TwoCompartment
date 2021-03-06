import numpy as np
import scipy.io as sio
import multiprocessing as mp
from timeit import default_timer as timer
from scipy.optimize import minimize, differential_evolution
from functools import partial
from numba import njit
from .data import get_singh_data, get_b1d2, calc_for_map
from .tau_twocomp import tau_twocomp_carrier
from typing import List, Tuple, Any, Union
from collections import Counter


@njit(cache=False)
def compute_deviance(p_inf: float, dose_index: int) -> float:
    """Smoothed deviance from Singh data.

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

    See Also
    --------
    compute_deviance_hform : Compute deviance of Singh data.

    Notes
    -----
    This version is smoother than that from [1]_.

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

    References
    ----------
    .. [1] Haas, C. N., Rose, J. B., & Gerba, C. P. (2014). Quantitative
    Microbial Risk Assessment. Quantitative Microbial Risk Assessment:
    Second Edition (Vol. 9781118145). Hoboken, New Jersey: John Wiley & Sons,
    Inc. https://doi.org/10.1002/9781118910030
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


def compute_deviance_hform(p_inf: float, dose_index: int) -> float:
    """Non-smoothed deviance from Singh data.

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

    See Also
    --------
    compute_deviance : Compute deviance of Singh data.

    Notes
    -----
    The formula for deviance is from [1]_, Chapter 8 (Page 315, Figure 8.14:
    this has the code to compute deviance). Small modification: add small
    positive to fpred as well.

    References
    ----------
    .. [1] Haas, C. N., Rose, J. B., & Gerba, C. P. (2014). Quantitative
    Microbial Risk Assessment. Quantitative Microbial Risk Assessment:
    Second Edition (Vol. 9781118145). Hoboken, New Jersey: John Wiley & Sons,
    Inc. https://doi.org/10.1002/9781118910030
    """

    # Import data
    _, norig, ntot, _, _, _ = get_singh_data()

    dev = 0
    fpred = p_inf
    fobs = norig[dose_index] / ntot
    Y1 = norig[dose_index] * np.log((fpred + 1e-15) / (fobs + 1e-15))
    Y2 = (ntot - norig[dose_index]) * np.log((1 - fpred + 1e-15) / (1 - fobs + 1e-15))
    dev = -2 * (Y1 + Y2)
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
    t_type: str,
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
    imax = Imax * A
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
            arg_list.append((init_load, rates, imax, nstep, seeds[ind1], 6.0, False))
        # Run parallel simulation
        partial_func = partial(calc_for_map, func=tau_twocomp_carrier)
        results = pool.map(partial_func, arg_list)
        for ind1, r in enumerate(results):
            # print("ind1 : ", ind1)
            extflag[ind1] = r[0]
            endt[ind1] = r[1]
            status[ind1] = r[4]
        p_inf[choice] = status_to_pinf(status)
        dev = compute_deviance(p_inf=p_inf[choice], dose_index=choice)
        devs.append(dev)

        # Early exit criterion, exit if objective has exceeded preset
        # threshold. Higher doses will have p_inf = 1, resulting in
        # even higher objectives.

        if obj_flag and np.sum(devs) > 150:
            print("Stopping early.")
            objval = 150
            return objval

        extflags.append(extflag)
        endts.append(endt)
        statuses.append(status)
        this_status = status
        print(
            f"Seed = {seed}, pinf = {p_inf[choice]:.3f}, dev = {dev:.3f},  status counts : ",
            Counter(this_status),
        )
        if np.any(this_status == 0):
            print("Zero status detected, rates, dose =  ", rates, h0[choice])

    objval = np.sum(devs)
    print("Objective is : ", objval)
    print("------------------------------------------")
    if obj_flag:
        return objval
    else:
        return (devs, extflags, endts, statuses)


def status_to_pinf(status: List[int]) -> float:
    """Calculate pinf from status.

    Parameters
    ----------
    status
        List of simulation statuses.

    Return
    ------
    pinf
        Response probability.

    Notes
    -----
    p_inf = prob (I(t) > Imax) + prob(H(t) > Imax)
    Population exceeds Imax if I(t) > Imax (status == 3) or
    if I(t) overflows or H(t) overflows.
    3 : Succesful completion, terminated when I(t) > Imax.
    4 : Succesful completion, curI overflow.
    5 : Succesful completion, curH overflow.
    """
    pinf = np.mean((status == 3) + (status == 4) + (status == 5))
    return pinf


def transform_x(x: List[float], t_type: Union[None, str]) -> Tuple[float, float]:
    """Transform `x` to `b2` and `d1`.

    Transform `x` to `b2` and `d1`, used by `carrier_obj_wrapper`.

    Parameters
    ----------
    x
        The vector containing values used for transformation.
    t_type
        Specify the kind of transform. Leave blank or specify as "log".

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
    if t_type == "log":
        b2 = 10 ** (-x[0])
    elif t_type is None:
        b2 = x[0]
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
    t_type: str = None,
    initial_guess: Tuple[float, float] = (2.5, 30),
    **kwargs,
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
    t_type
        Tranformation type to apply to b2.
    initial_guess
        Initial guess for the optimizer.
    **kwargs
        Keyword arguments.
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
        print("Initial_guess is : ", initial_guess)

    print("Creating pool with", n_procs, " processes\n")
    pool = mp.Pool(n_procs)
    print("pool = %s", pool)

    t0 = timer()
    for ind in range(ndesol):
        r1, r2, r3, Imax, modno = get_consts_bX(
            bXlist=bXlist, ind=ind, filename=filename
        )
        t1 = timer()
        this_args = (r1, r2, r3, Imax, npts, nrep, nstep, seed, pool, True, t_type)
        if method == "Powell":
            min_obj = minimize(
                minimization_objective,
                initial_guess,
                args=this_args,
                options={"maxfev": niter},
                method=method,
            )
            opt_str = method
        elif method == "diffev":
            pop = kwargs["pop"]
            if kwargs["use_initial"]:
                init = np.zeros((pop, 2))
                np.random.seed(seed)
                init[:, 0] = (
                    initial_guess[0]
                    + (np.random.random(pop) - 0.5) * initial_guess[0] * 0.2
                )
                init[:, 1] = (
                    initial_guess[1]
                    + (np.random.random(pop) - 0.5) * initial_guess[1] * 0.2
                )
                print(init)
                opt_str = "de" + str(pop) + "p_init"
            else:
                init = "latinhypercube"
                opt_str = "de" + str(pop) + "p_lhs"
            min_obj = differential_evolution(
                minimization_objective,
                bounds=kwargs["bounds"],
                args=this_args,
                maxiter=niter,
                popsize=pop,
                polish=False,
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
        + opt_str
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
            t_type=t_type,
            initial_guess=initial_guess,
        )


def get_consts_bX(
    bXlist: np.ndarray, ind: int, filename: str, verbose: int = 1
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
    verbose
        Verbosity of output. 0(no output) or 1(more output).

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
        if verbose:
            print("bXlist, r3 and Imax are : ", bXlist[ind, :], r3, Imax)
    if verbose:
        print("r3 * Imax is : ", r3 * Imax)

    return r1, r2, r3, Imax, modno


def get_bF_bX(
    filename: str = "results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat",
    desol_ind: List[int] = [0],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the best F and X values from DEMC solutions.

    Read the solutions present in `filename` and return the best objective
    values and corresponding best-fit parameters.

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
        The best-fit parameters that provide the best objective values.
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


def compute_devs_brute(
    filename="results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat",
    npts: int = 2,
    nrep: int = 10,
    seed: int = 0,
    desol_ind: List[int] = [0],
    nstep: int = 200_000,
    problem_type: int = 1,
    n_procs: int = 2,
    lims: dict = {"d1l": 0, "d1u": 5, "b2l": 0, "b2u": 5},
    nb2: int = 2,
    nd1: int = 2,
    t_type: str = None,
):
    """Optimize for deviances of the DEMC solutions.

    Runs the optimization and saves the outputs for brute force optimization.

    Parameters
    ----------
    filename
        DE solution file name.
    npts
        Number of doses to calculate deviance at.
    nrep
        Number of simulations per (b1,d2) pair to calculate deviance.
    seed
        Seed of the `NumPy` random generator, different from the seed
        of `numba`.
    desol_ind
        Indices of the DE solutions to evaluate deviance.
    nstep
        Maximum number of steps to run each simulation.
    problem_type
        If 1, optimize for b2 and d1. If 2, optimize only for d1 with b2 = 0.
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

    all_devs = []
    all_statuses = []
    tot_devs = []

    if problem_type == 1:
        minimization_objective = carrier_obj_wrapper

    print("Creating pool with", n_procs, " processes\n")
    pool = mp.Pool(n_procs)
    print("pool = %s", pool)

    t0 = timer()
    for ind1 in range(ndesol):
        r1, r2, r3, Imax, modno = get_consts_bX(
            bXlist=bXlist, ind=ind1, filename=filename
        )
        t1 = timer()
        for b2u in b2listu:
            for d1u in d1listu:
                print(f"Starting loop for b2u={b2u:.2e}, d1u={d1u:.2f}")
                devs, _, _, statuses = minimization_objective(
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
                    t_type=t_type,
                )
                all_devs.append(devs)
                all_statuses.append(statuses)
                tot_devs.append(np.sum(devs))
        print(np.min(tot_devs))
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
        + "_"
        + solstr
        + "b2d1_1o4_cpu.npz"
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
            modno=modno,
            all_devs=all_devs,
            tot_devs=tot_devs,
            all_statuses=all_statuses,
            b2listu=b2listu,
            d1listu=d1listu,
            nb2=nb2,
            nd1=nd1,
            t_type=t_type,
        )
