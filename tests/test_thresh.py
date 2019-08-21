import numpy as np
from ..staph.utils.dev_thresh import get_best_thresh, thresh_brute_min
from ..staph.utils.data import get_singh_data


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
