import numpy as np
import sys
import time
from staph.utils.dev import compute_devs_min as cdmin
from staph.utils.dev import compute_devs_brute as cdbrute
from staph.utils.predict import predict_fit

if __name__ == "__main__":
    choice = np.int32(sys.argv[1])
    seed = 0
    if choice == 1:
        nrep = 1000
        # For wide bounds
        fname = "results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat"
        ncores = np.int32(sys.argv[2])
        ind = np.int32(sys.argv[3])
        assert ind >= 1
        desol_ind = np.arange(ind - 1, ind)
        nstep = 200000
        npts = 2
        niter = 5
        cdmin(
            filename=fname,
            npts=npts,
            nrep=nrep,
            seed=seed,
            desol_ind=desol_ind,
            nstep=nstep,
            method="Powell",
            niter=niter,
            problem_type=1,
            n_procs=ncores,
            initial_guess=(1e-3, 160),
        )
    elif choice == 2:
        nrep = 1000
        # For wide bounds
        fname = "results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat"
        ncores = np.int32(sys.argv[2])
        ind = np.int32(sys.argv[3])
        assert ind >= 1
        desol_ind = np.arange(ind - 1, ind)
        nstep = 200000
        npts = 6
        cdbrute(
            filename=fname,
            npts=npts,
            nrep=nrep,
            seed=seed,
            desol_ind=desol_ind,
            nstep=nstep,
            problem_type=1,
            n_procs=ncores,
            lims={"d1l": 130, "d1u": 170, "b2l": 1.2, "b2u": 3.6},
            nb2=16,
            nd1=10,
            t_type="log",
        )
    elif choice == 3:
        nrep = 1000
        # For wide bounds
        fname = "results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat"
        ncores = np.int32(sys.argv[2])
        ind = np.int32(sys.argv[3])
        assert ind >= 1
        desol_ind = np.arange(ind - 1, ind)
        nstep = 200000
        npts = 1
        niter = 2
        cdmin(
            filename=fname,
            npts=npts,
            nrep=nrep,
            seed=seed,
            desol_ind=desol_ind,
            nstep=nstep,
            method="diffev",
            niter=niter,
            problem_type=1,
            n_procs=ncores,
            initial_guess=(1.1, 0.7),
            bounds=[[0.5, 2], [0.5, 6]],
            pop=5,
            use_initial=True,
        )
    elif choice == 4:
        ncores = np.int32(sys.argv[2])
        predict_fit(
            nrep=1000,
            nstep=200000,
            hyp="base",
            inoc_time="base",
            rank_1_sol_inds=[0],
            doselist=np.int32(np.power(10, np.arange(1, 6.3, 0.3))),
            n_cores=ncores,
        )
