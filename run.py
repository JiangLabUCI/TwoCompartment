import numpy as np
import sys
import time
from staph.utils.dev import compute_devs_min as cdmin
from staph.utils.dev import compute_devs_brute as cdbrute


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
            lims={"d1l": 1.5, "d1u": 3.5, "b2l": 5, "b2u": 30},
            nb2=5,
            nd1=5,
        )
