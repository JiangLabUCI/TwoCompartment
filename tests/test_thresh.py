import numpy as np
from ..staph.utils.dev_thresh import get_best_thresh, thresh_brute_min
from ..staph.utils.dev_thresh import r_to_load
from ..staph.utils.data import get_singh_data
from ..staph.analysis.f2 import get_filename
from ..staph.utils.tau_twocomp import tau_twocomp_carrier


def test_gb_thresh():
    thresh = 1000

    # Construct fake final loads
    _, norig, ntot, _, _, _ = get_singh_data()
    p = np.array(norig) / ntot
    npts = 6
    nrep = 20
    fake_loads = np.zeros([npts, nrep])

    for ind1 in range(npts):
        switch_index = np.int(p[ind1] * nrep)
        print(f"switch index {switch_index}")
        fake_loads[ind1, :switch_index] = thresh + 1
        fake_loads[ind1, switch_index:] = thresh - 1
    # print(fake_loads)
    best_thresh, best_dev, devs, _ = get_best_thresh(fake_loads)
    assert devs.shape[0] == 2
    assert best_thresh == thresh
    assert np.abs(best_dev) < 1e-12

    np.random.seed(0)
    for ind1 in range(npts):
        switch_index = np.int(p[ind1] * nrep)
        n1 = len(np.arange(switch_index))
        n2 = nrep - n1
        fake_loads[ind1, :switch_index] = thresh + 1 * np.random.randint(2, 10, n1)
        fake_loads[ind1, switch_index:] = thresh - 1 * np.random.randint(2, 10, n2)
    # print(fake_loads)
    best_thresh, best_dev, devs, _ = get_best_thresh(fake_loads)
    assert devs.shape[0] == len(np.unique(fake_loads))
    assert best_thresh == 999
    assert np.abs(best_dev) < 1e-12


def test_thresh_minimizer():
    thresh_brute_min(nstep=100)


def test_get_filenames():
    assert get_filename(1) == "ntest.o8381995.1"
    assert get_filename(60) == "ntest.o8376290.60"
    assert get_filename(160) is None


def test_r_to_load():
    init_load = np.array([100], dtype=np.int32)
    rates = np.array([1.0, 0, 0, 0, 0, 0])  # r1, r2, b1, b2, d1, d2
    imax = 2000
    nstep = 200
    seed = 0
    t_max = 10
    store_flag = True
    r = tau_twocomp_carrier(
        init_load=init_load,
        rates=rates,
        imax=imax,
        nstep=nstep,
        seed=seed,
        t_max=t_max,
        store_flag=store_flag,
    )
    assert r[2][-1, 0] == 0
    assert r[2][-1, 1] == 0
    assert r_to_load(r) == 0
