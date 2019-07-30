import numpy as np
from ..staph.utils.det_models import *


def test_dm_derivatives():
    p = {}
    p["k1"], p["k2"], p["k3"], p["Nmax"] = 0, 1, 1, 1
    assert rh_growth_model(0, 0, p) == 0
    assert rh_growth_model(0, 1, p) == 0
    assert rh_growth_model(0, 2, p) == -2

    p["k1"], p["k2"], p["k3"], p["Nmax"] = 1, 0, 1, 1
    assert rh_growth_model(0, 1, p) == -1
    assert rh_growth_model(0, 2, p) == -4

    p["k1"], p["k2"], p["k3"], p["Nmax"] = 1, 1, 0, 1
    assert rh_growth_model(0, 1, p) == -1
    assert rh_growth_model(0, 2, p) == -2

    p["k1"], p["k2"], p["k3"], p["Nmax"] = 1, 1, 1, 0
    assert rh_growth_model(0, 1, p) == -2
    assert rh_growth_model(0, 2, p) == -6

    del p
    p = {}
    p["r1"], p["r2"], p["r3"], p["r3Imax"] = 1, 1, 1, 1
    assert np.all(twocomp_model(np.random.random(), [0, 0], p) == [0, 0])
    assert np.all(twocomp_model(np.random.random(), [1, 0], p) == [-2, 1])
    assert np.all(twocomp_model(np.random.random(), [0, 1], p) == [0, 0])
    assert np.all(twocomp_model(np.random.random(), [1, 1], p) == [-2, 1])

    p["rmf"] = 0
    assert np.all(twocomp_rmf_model(np.random.random(), [0, 0], p) == [0, 0])
    assert np.all(twocomp_rmf_model(np.random.random(), [1, 0], p) == [-2, 1])
    assert np.all(twocomp_rmf_model(np.random.random(), [0, 1], p) == [0, 0])
    assert np.all(twocomp_rmf_model(np.random.random(), [1, 1], p) == [-2, 1])

    p["r1"], p["r2"], p["r3"], p["r3Imax"] = np.random.random(), np.random.random(), np.random.random(), np.random.random()
    y = [np.random.random(), np.random.random()]
    assert np.all(twocomp_model(np.random.random(), y, p) == twocomp_rmf_model(np.random.random(), y, p))

