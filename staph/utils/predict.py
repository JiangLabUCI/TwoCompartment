import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Callable
from scipy import interpolate
from .dev import carrier_obj_wrapper, status_to_pinf
from .data import calc_for_map, get_b1d2, get_singh_data, get_bedrail_data
from .tau_twocomp import tau_twocomp_carrier
from .tau_twocomp_rmf import tau_twocomp_carrier_rmf


def predict_fit(
    filename: str = "results/rank_1_solutions.csv",
    n_cores: int = 2,
    nrep: int = 10,
    nstep: int = 200,
    rank_1_sol_inds: List[int] = [0, 5],
    seed: int = 0,
    doselist: np.ndarray = np.arange(0, 1),
    hyp: str = "base",
    inoc_time: str = "base",
    pop_array_flag: bool = False,
) -> [None, List[np.ndarray]]:
    """Predict outcome probabilities.

    This function predicts the outcome probabilties at user provided 
    list of doses for rank 1 solutions in the `filename` file.
    Predictions are run in parallel.

    Parameters
    ----------
    filename
        The names of binary file with solutions from optimization.
    n_cores
        The number of cores to run predictions on.
    nrep
        The number of repetitions to run the predictions for.
    nstep
        The maximum number of simulation steps to run the algorithm for.
    rank_1_sol_inds
        The indices of the rank 1 solutions.
    seed
        Seed of the random number generator.
    doselist
        The list of doses to predict outcome probabilities at.
    hyp
        One of "base", "r1s" or "rmf".
    inoc_time
        One of "base", "imm" or "24h".
    pop_array_flag
        Whether or not to return population arrays.

    Notes
    -----

    The three cases under which predictions will be made are given by `hyp`.
    - "base" means no changes to constants.
    - "r1s" means r1* hypothesis, r1 is updated.
    - "rmf" means rmf hypotehsis, rmf is introduced.

    inoc_time
    - "base" is assuming skin was wiped with alcohol before inoculation.
    - "imm" is assuming skin was control-soap-washed before inoculation.
    - "24h" is assuming skin was control-soap-washed 24h before inoculation.

    The predicted constants are stored in "results/pred_consts.csv". It has
    the following columns:
        Dataset - 1 (control soap, 24h before inoculation)
                - 2 (control soap, immediate inoculation)
        Parameterset - 1-6 (Index of the rank 1 solution + 1)
        Hyp - 1 (r1 and r2 change)
              2 (r1 changes)
              3 (r2 changes)
              4 (r3 changes)
              5 (r3Imax changes)
              6 (rmf is introduced)
        Parameter - Value of parameter fitted according to hypothesis.
        SSE - Sum of squared error of the fit.
        AIC - Akaike information criterion of the fit.
        AICc - Corrected Akaike information criterion of the fit.
        BIC - Bayesian information criterion of the fit.
    Hyp 1 does not have the parameters stored and does not appear in the file.

    """

    assert hyp in ["base", "r1s", "rmf"]
    if hyp == "base":
        assert inoc_time == "base"
    else:
        assert inoc_time in ["imm", "24h"]

    df = pd.read_csv(filename)
    pdf = pd.read_csv("results/pred_consts.csv")
    _, _, _, _, A, _ = get_singh_data()

    if inoc_time == "imm":
        datasetno = 1
    elif inoc_time == "24h":
        datasetno = 2
    if inoc_time is not "base":
        pdf = pdf[pdf.Dataset == datasetno]
    pool = mp.Pool(n_cores)

    print("Parameters are : ")
    print(df)

    nrank1sols = len(rank_1_sol_inds)
    ndose = len(doselist)
    pinf = np.zeros([nrank1sols, ndose])
    pcar = np.zeros([nrank1sols, ndose])
    ps = np.zeros([nrank1sols, ndose])
    if pop_array_flag:
        pop_array = []
        t_array = []
    for ind1, r1sind in enumerate(rank_1_sol_inds):
        np.random.seed(seed)
        seeds = np.random.randint(low=0, high=1e5, size=nrep)
        rates, simfunc, Imax, = get_rates_simfunc(
            df=df, pdf=pdf, r1sind=r1sind, hyp=hyp
        )

        for ind2 in range(ndose):
            arg_list = []
            extflag = np.zeros(nrep)
            status = np.zeros(nrep)
            for ind3 in range(nrep):
                init_load = np.array([doselist[ind2]], dtype=np.int32)
                arg_list.append(
                    (init_load, rates, Imax * A, nstep, seeds[ind3], 6.0, True)
                )
            # Run parallel simulation
            partial_func = partial(calc_for_map, func=simfunc)
            results = pool.map(partial_func, arg_list)
            for ind3, r in enumerate(results):
                extflag[ind3] = r[0]
                status[ind3] = r[4]
                if pop_array_flag:
                    pop_array.append(r[2])
                    t_array.append(r[3])

            pinf[ind1, ind2] = status_to_pinf(status)
            ps[ind1, ind2] = np.mean(extflag)
            pcar[ind1, ind2] = 1 - (pinf[ind1, ind2] + ps[ind1, ind2])

    de_str = ""
    for r1sind in rank_1_sol_inds:
        de_str += str(r1sind)
    with open(
        "results/preds"
        + hyp
        + inoc_time
        + str(sum(doselist))
        + "dl"
        + de_str
        + "r1_"
        + str(nrep)
        + "rep"
        + ".npz",
        "wb",
    ) as f:
        np.savez(
            f,
            df=df,
            pinf=pinf,
            pcar=pcar,
            ps=ps,
            rank_1_sol_inds=rank_1_sol_inds,
            doselist=doselist,
        )
    if pop_array_flag:
        return pop_array, t_array


def get_rates_simfunc(
    df: pd.DataFrame, pdf: pd.DataFrame, r1sind: int, hyp: str
) -> Tuple[List[float], callable, float]:
    """Get rates and simulation function.

    Use the hypothesis to get the corresponding rates and simulation functions.
    Intended for use by `predict_fit`.

    Parameters
    ----------
    df
        Dataframe with rank 1 solutions.
    pdf
        Dataframe with fitted constants.
    r1sind
        Index of rank 1 solution.
    hyp
        Hypothesis.
    
    Returns
    -------
    rates
        Rates for stochastic simulation.
    simfunc
        Model for stochastic simulation.
    Imax
        Imax value used for threshold of stochastic simulation.

    See Also
    --------
    predict_fit : Predict outcome probabilities.

    """
    r1 = df.r1[r1sind]
    r2 = df.r2[r1sind]
    r3 = df.r3[r1sind]
    r3Imax = df["r3*Imax"][r1sind]
    d1 = df.d1[r1sind]
    b2 = df.b2[r1sind]
    Imax = r3Imax / r3
    b1, d2 = get_b1d2(b2=b2, d1=d1, r3=r3, r3Imax=r3 * Imax)
    if hyp == "r1s":
        r1_star = float(
            pdf[(pdf.Hyp == 2) & (pdf.Parameterset == r1sind + 1)].Parameter
        )
        rates = np.array([r1_star, r2, b1, b2, d1, d2])
        simfunc = tau_twocomp_carrier
    elif hyp == "rmf":
        rmf = float(pdf[(pdf.Hyp == 6) & (pdf.Parameterset == r1sind + 1)].Parameter)
        rates = np.array([r1, r2, b1, b2, d1, d2, rmf])
        simfunc = tau_twocomp_carrier_rmf
    elif hyp == "base":
        rates = np.array([r1, r2, b1, b2, d1, d2])
        simfunc = tau_twocomp_carrier

    return rates, simfunc, Imax


def get_rates(hyp: str = "r1*") -> Tuple[List[np.float32], np.float32]:
    """Get the rates for r1* or rmf hypotheses.

    Parameters
    ----------
    hyp
        Hypothesis the rates are needed for, either "r1*" or "rmf".

    Returns
    -------
    rates
        The list of rates used for simulation.

    Notes
    -----
    Information is read from "pred_consts.txt". 
    If `hyp` is "r1*", six rate constants are returned in a list 
    (r1, r2, b1, b2, d1 and d2).This is then used for simulation with the 
    `tau_twocomp_carrier` function.

    If `hyp` is "rmf" seven rate constants are returned in a list 
    (r1, r2, b1, b2, d1, d2 and rmf). This is then used for simulation with 
    the `tau_twocomp_carrier_rmf` function.

    """
    dsno = 1
    parno = 6
    filename = "results/pred_consts.csv"
    data = pd.read_csv(filename)
    if hyp == "rmf":
        hypno = 6
        rmf = data[
            (data.Dataset == dsno) & (data.Parameterset == parno) & (data.Hyp == hypno)
        ].Parameter.values[0]
    elif hyp == "r1*":
        hypno = 2
        r1 = float(
            data[
                (data.Dataset == dsno)
                & (data.Parameterset == parno)
                & (data.Hyp == hypno)
            ].Parameter
        )

    data = pd.read_csv("results/rank_1_solutions.csv")
    data = data.iloc[parno - 1]
    r2 = data.r2
    r3 = data.r3
    r3Imax = data["r3*Imax"]
    b2 = data.b2
    d1 = data.d1
    b1, d2 = get_b1d2(b2=b2, d1=d1, r3=r3, r3Imax=r3Imax)
    if hyp == "rmf":
        r1 = data.r1
        rates = np.array([r1, r2, b1, b2, d1, d2, rmf])
    elif hyp == "r1*":
        rates = np.array([r1, r2, b1, b2, d1, d2])
    Imax = r3Imax / r3
    return rates, Imax


def sim_multi(
    simfunc: Callable,
    rates: List[np.float32],
    dose_intervals: List,
    dose_loads: List,
    Imax: float,
    nstep: int = 200_000,
    seed: int = 0,
    t_max: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], int, int]:
    """Simulate multiple inoculations.

    Simulate multiple inoculations with specified intervals and loads.

    Parameters
    ----------
    simfunc
        Simulation function for the simulations.
    rates
        Rates corresponding to the simulation function.
    dose_intervals
        Time intervals between inoculations.
    dose_loads
        Bacterial load at inoculation time.
    Imax
        Imax value to be used for simulation.
    nstep
        Number of steps to execute the simulation for.
    seed
        Seed for random number generator.
    t_max
        Maximum time to simulate for.
    
    Returns
    -------
    pop
        Time course of populations.
    t
        Time course.
    t_array
        List of time courses, directly from simulation.
    explosion
        Flag, = 1 if population explodes, = 0 otherwise.
    extinction
        Flag, = 1 if population goes extinct, = 0 otherwise.
    
    Notes
    -----
    Extinction is set by `status_to_pinf`. If pinf = 1, extinction = 1.
    If pinf = 0, extinction = 0.
    """
    n = len(dose_intervals)
    dose_intervals = np.hstack(
        [dose_intervals, np.max([0, t_max - np.sum(dose_intervals)])]
    )
    pop_array = [0 for ind in range(n)]
    t_array = [0 for ind in range(n)]
    np.random.seed(seed)
    seeds = np.random.randint(0, 1e5, size=n)
    explosion = 0
    for ind in range(n):
        init_load = np.array([dose_loads[ind], 0])
        if ind != 0:
            init_load += pop_array[ind - 1][:, -1]
        t_final = dose_intervals[ind + 1]
        _, endt, pop_array[ind], t_array[ind], this_status = simfunc(
            init_load=init_load,
            rates=rates,
            Imax=np.int32(Imax),
            nstep=nstep,
            seed=seeds[ind],
            t_max=t_final,
            store_flag=True,
        )

        if endt >= t_final:
            # Simulation overshooting
            # Interpolate intermediate population
            inds_to_keep = np.where(t_array[ind] <= t_final)
            i = np.max(inds_to_keep)
            t_i = t_array[ind][i]
            t_ip1 = t_array[ind][i + 1]
            p_i = pop_array[ind][:, i]
            p_ip1 = pop_array[ind][:, i + 1]
            f = interpolate.interp1d(x=[t_i, t_ip1], y=[p_i, p_ip1], axis=0)
            p_interp = f(t_final)
            p_interp = np.array([np.int(p_interp[0]), np.int(p_interp[1])])
            # Drop extra points
            t_array[ind] = t_array[ind][inds_to_keep]
            pop_array[ind] = pop_array[ind][:, inds_to_keep].reshape(2, -1)
            # Append exposure time point and interpolated population
            t_array[ind] = np.hstack([t_array[ind], t_final])
            pop_array[ind] = np.hstack([pop_array[ind], p_interp.reshape(2, -1)])
        # Dropping extra points may result in undershooting
        endt = np.max(t_array[ind])
        if endt < t_final:
            # Simulation undershooting, due to extinction
            # Append zeros to population
            t_array[ind] = np.hstack([t_array[ind], t_final])
            pop_array[ind] = np.hstack([pop_array[ind], np.array([[0], [0]])])
        if ind == 0:
            pop = pop_array[ind]
            t = t_array[ind]
        else:
            pop = np.hstack([pop, pop_array[ind]])
            t = np.hstack([t, np.max(t) + t_array[ind]])
        # explosion = 1 if pinf = 1, explosion = 0 if pinf = 0
        explosion = status_to_pinf(np.array([this_status]))
        if explosion:
            pop = pop[:, :-1]
            t = t[:-1]
            break
    # Right shift the simulations
    t = np.hstack([0, dose_intervals[0], dose_intervals[0] + t])
    pop = np.hstack([np.array([[0, 0], [0, 0]]), pop])
    if np.sum(pop[:, -1]):
        extinction = 0
    else:
        extinction = 1
    return pop, t, t_array, explosion, extinction, this_status


def predict_bedrail(
    n_cores: int = 2,
    nrep: int = 10,
    nstep: int = 200,
    seed: int = 0,
    hyp: str = "r1*",
    sex: str = "F",
    pop_flag: bool = True,
    t_max: float = 6.0,
    n_to_save: int = 10,
) -> None:
    """Predicts outcome probabilities for bedrail case study.

    This function predicts the outcome probabilties and population time courses
    for the bedrail case study.

    Parameters
    ----------
    n_cores
        The number of cores to run predictions on.
    nrep
        The number of repetitions to run the predictions for.
    nstep
        The maximum number of simulation steps to run the algorithm for.
    seed
        Initial of the random number generator.
    hyp
        Hypothesis, either "r1*" or "rmf".
    sex
        For hand size, is "F" or "M".
    pop_flag
        If `True`, save the population. If `False`, don't save population.
    t_max
        Maximum time to run the simulations for.
    n_to_save
        Number of time courses of each outcome type to save.
    """
    np.random.seed(seed)
    seeds = np.random.randint(low=0, high=1e5, size=nrep)
    tref = np.linspace(0, t_max, 20)
    sstat = np.zeros((nrep, tref.shape[0]))
    pool = mp.Pool(n_cores)

    rates, Imax = get_rates(hyp)
    if hyp == "r1*":
        simfunc = tau_twocomp_carrier
    elif hyp == "rmf":
        simfunc = tau_twocomp_carrier_rmf
    dose_intervals, dose_loads = get_bedrail_data(nrep, tmax=t_max)
    pop = [0 for x in range(nrep)]
    popH = [0 for x in range(nrep)]
    popI = [0 for x in range(nrep)]
    t = [0 for x in range(nrep)]
    explosion = [0 for x in range(nrep)]
    extinction = [0 for x in range(nrep)]
    status = [0 for x in range(nrep)]
    arg_list = []
    for ind1 in range(nrep):
        # Assemble the argument list for multiprocessing.
        arg_list.append(
            (
                simfunc,
                rates,
                dose_intervals[ind1],
                dose_loads[ind1],
                Imax,
                nstep,
                seeds[ind1],
            )
        )
    partial_func = partial(calc_for_map, func=sim_multi)
    results = pool.map(partial_func, arg_list)
    for ind2, r in enumerate(results):
        pop[ind2] = r[0]
        popH[ind2] = r[0][0, :]
        popI[ind2] = r[0][1, :]
        t[ind2] = r[1]
        explosion[ind2] = r[3]
        extinction[ind2] = r[4]
        status[ind2] = r[5]
        sstat[ind2, :] = get_stat_time_course(
            tsim=r[1], pop=np.sum(pop[ind2], axis=0), tref=tref, thresh=Imax
        )

    pres, pcar, ps = stat_ocprob(stat=sstat)

    # Output file name
    output_name = (
        "results/pred_"
        + str(nrep)
        + "rep"
        + str(nstep)
        + "nst"
        + hyp.replace("*", "")
        + "hyp"
        + sex
        + str(int(t_max))
        + "_multi.npz"
    )

    if not pop_flag:
        extcount, expcount, carcount = 0, 0, 0
        tempH, tempI, tempt = [], [], []
        new_exp = [0 for x in range(n_to_save * 3)]
        new_ext = [0 for x in range(n_to_save * 3)]
        for ind in range(nrep):
            if extinction[ind] and (extcount < n_to_save):
                tempH.append(popH[ind])
                tempI.append(popI[ind])
                tempt.append(t[ind])
                new_ext[extcount + expcount + carcount] = 1
                extcount += 1
            if explosion[ind] and (expcount < n_to_save):
                tempH.append(popH[ind])
                tempI.append(popI[ind])
                tempt.append(t[ind])
                new_exp[extcount + expcount + carcount] = 1
                expcount += 1
            if carcount < n_to_save:
                tempH.append(popH[ind])
                tempI.append(popI[ind])
                tempt.append(t[ind])
                carcount += 1
            if (
                (extcount >= n_to_save)
                and (expcount >= n_to_save)
                and (carcount >= n_to_save)
            ):
                break
        popH = tempH
        popI = tempI
        t = tempt
        print(f"Extinction : ps = {ps[-1]}, saved = {extcount}")
        print(f"Explosion : pres = {pres[-1]}, saved = {expcount}")
        print(f"Carrier : pcar = {pcar[-1]}, saved = {carcount}")
    else:
        new_exp = explosion
        new_ext = extinction
    with open(output_name, "wb") as f:
        np.savez(
            f,
            t=t,
            pres=pres,
            ps=ps,
            pcar=pcar,
            popH=popH,
            popI=popI,
            nstep=nstep,
            nrep=nrep,
            sex=sex,
            seed=seed,
            explosion=explosion,
            extinction=extinction,
            tref=tref,
            pop_flag=pop_flag,
            status=status,
            t_max=t_max,
            new_exp=new_exp,
            new_ext=new_ext,
        )

    print("Output file name : ", output_name)


def get_stat_time_course(
    tsim: np.ndarray, pop: np.ndarray, tref: np.ndarray, thresh: int
) -> np.ndarray:
    """Return ssimulation status.

    Return the simulation status for a time series at given reference times.

    Parameters
    ----------
    tsim
        Simulation times.
    pop
        Population values at tsim.
    tref
        Reference times.
    thresh
        Threshold for deciding simulation status.
    
    Returns
    -------
    stat
        Simulation statuses.
    
    Notes
    -----
    stat values can be:
    -1 : Never set, error.
    1 : Population extinct/never set.
    2 : Population in (0, threshold).
    3 : Population >= threshold.
    """
    nref = len(tref)
    stat = np.ones(nref) * -1
    for ind in range(nref):
        this_t = tref[ind]
        sim_ind = np.max(np.where(tsim <= this_t))
        this_pop = pop[sim_ind]
        if this_pop == 0:
            stat[ind] = 1
        elif this_pop < thresh:
            stat[ind] = 2
        elif this_pop >= thresh:
            stat[ind] = 3

    return stat


def stat_ocprob(stat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get outcome probabilities.

    Get outcome probabilities from simulation status.

    Parameters
    ----------
    stat
        Status matrix (nrep, nt).
    
    Returns
    -------
    pres
        Response probability (nt,).
    pcar
        Carrier probability (nt,).
    ps
        Unaffected probability (nt,).
    """
    pres = np.mean(stat == 3, axis=0)
    pcar = np.mean(stat == 2, axis=0)
    ps = np.mean(stat == 1, axis=0)

    return pres, pcar, ps
