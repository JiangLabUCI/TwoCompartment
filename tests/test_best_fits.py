import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp
from ..staph.utils.tau_twocomp import tau_twocomp_carrier
from ..staph.utils.data import get_b1d2, get_singh_data, calc_for_map
from ..staph.utils.dev import status_to_pinf
from ..staph.analysis.igate_ntest import igate
from collections import Counter


def test_best_fits():
    # Open rank 1 solutions, use parameters and see if Fst is recovered

    # Define some parameters
    npts = 6
    nrep = 1000
    nstep = 200_000
    seed = 0
    n_procs = 8
    tol = 1e-7
    ptol = 6e-3
    np.random.seed(seed)
    seeds = np.random.randint(low=0, high=1e5, size=nrep)
    pool = mp.Pool(n_procs)

    # Get data
    _, _, _, _, A, H0 = get_singh_data()

    # Get rates
    # Constructed from 6021324_1000rep0se100ite_Powell_xxtoxxb2d1_1o3_cpu.npz
    # using `landscape.py`
    # So verify against `ntest.o7721941.xx`
    data = pd.read_csv("results/rank_1_solutions.csv")
    print(data)
    for r1_ind in range(6):
        r1 = data.r1[r1_ind]
        r2 = data.r2[r1_ind]
        r3 = data.r3[r1_ind]
        r3Imax = data["r3*Imax"].iloc[r1_ind]
        Imax = r3Imax / r3
        b2 = data.b2[r1_ind]
        d1 = data.d1[r1_ind]
        b1, d2 = get_b1d2(b2=b2, d1=d1, r3=r3, r3Imax=r3Imax)
        rates = np.array([r1, r2, b1, b2, d1, d2])
        desol_ind = data.desol_inds[r1_ind]
        fname = "results/ops/ntest.o7721941." + str(desol_ind + 1)
        print("fname : ", fname)
        x = igate(filenames=[fname], option1=4)
        # compare deviance, b2 and d1
        assert abs(data.Fst[r1_ind] - x[0]) < tol
        assert abs(b2 - x[1]) < tol
        assert abs(d1 - x[2]) < tol

        for ind1 in range(npts):
            init_load = np.array([H0[ind1]], dtype=np.int32)
            arg_list = []
            extflag = np.zeros(nrep)
            endt = np.zeros(nrep)
            status = np.zeros(nrep)

            for ind2 in range(nrep):
                arg_list.append(
                    (init_load, rates, Imax * A, nstep, seeds[ind2], 6.0, True)
                )
            # Run parallel simulation
            partial_func = partial(calc_for_map, func=tau_twocomp_carrier)
            results = pool.map(partial_func, arg_list)
            for ind2, r in enumerate(results):
                extflag[ind2] = r[0]
                endt[ind2] = r[1]
                status[ind2] = r[4]
                # Check population non-negative
                assert np.all(r[3] >= 0)
            print(Counter(status))
            this_pinf = status_to_pinf(status)
            # compare pinf
            assert abs(this_pinf - x[3][ind1]) < ptol
            print(f"In file {x[3][ind1]}, in this run {this_pinf}")
