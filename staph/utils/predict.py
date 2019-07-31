import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from typing import List, Tuple
from .dev import carrier_obj_wrapper, status_to_pinf
from .data import calc_for_map, get_b1d2, get_singh_data
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

