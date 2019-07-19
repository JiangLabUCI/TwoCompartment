import numpy as np
from ..staph.utils.dev_thresh import get_best_thresh
from ..staph.utils.data import get_singh_data


def test_gb_thresh():
    thresh = 1000

    # Construct fake final loads
    _, norig, ntot, _, _, _ = get_singh_data()
    p = np.array(norig) / ntot
    npts = 6
    nrep = 20
    low = 899
    high = 1005
    fake_loads = np.zeros([npts, nrep])
    for ind1 in range(npts):
        switch_index = np.int(p[ind1] * nrep)
        print(f"switch index {switch_index}")
        fake_loads[ind1, :switch_index] = thresh + 1
        fake_loads[ind1, switch_index:] = thresh - 1
    # print(fake_loads)
    best_thresh, best_dev, devs, _ = get_best_thresh(fake_loads, low=low, high=high)
    assert devs.shape[0] == high - low
    assert best_thresh == thresh
    assert np.abs(best_dev) < 1e-12

