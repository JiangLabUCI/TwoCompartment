import numpy as np
from ..staph.utils.det_models import *
from ..staph.utils.tau_twocomp import tau_twocomp_carrier
from ..staph.utils.tau_twocomp_rmf import tau_twocomp_carrier_rmf


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

    p["r1"], p["r2"], p["r3"], p["r3Imax"] = (
        np.random.random(),
        np.random.random(),
        np.random.random(),
        np.random.random(),
    )
    y = [np.random.random(), np.random.random()]
    assert np.all(
        twocomp_model(np.random.random(), y, p)
        == twocomp_rmf_model(np.random.random(), y, p)
    )


def test_sm():
    # Only r1 fires
    r1, r2, b1, b2, d1, d2 = 1, 0, 0, 0, 0, 0
    rates = np.array([r1, r2, b1, b2, d1, d2])
    init_load = np.array([100], dtype=np.int32)
    # Not until extinction
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=10, seed=0
    )
    assert extflag == 0
    assert status == -1
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind - 1] > pop_array[0, ind]
        assert pop_array[1, ind] == 0
    # Until extinction
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=200, seed=0
    )
    assert extflag == 1
    assert status == 1
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        # Population declines
        assert pop_array[0, ind - 1] > pop_array[0, ind]
        assert pop_array[1, ind] == 0

    # Only r2 fires --> conservation
    r1, r2, b1, b2, d1, d2 = 0, 1, 0, 0, 0, 0
    rates = np.array([r1, r2, b1, b2, d1, d2])
    n0 = 100
    init_load = np.array([n0], dtype=np.int32)
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=200, seed=0
    )
    assert extflag == 0
    assert status == 2
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind] + pop_array[1, ind] == n0

    # r2 and d1 fire --> extinction
    r1, r2, b1, b2, d1, d2 = 0, 1, 0, 0, 1, 0
    rates = np.array([r1, r2, b1, b2, d1, d2])
    n0 = 15
    init_load = np.array([n0], dtype=np.int32)
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=200, seed=0
    )
    assert extflag == 1
    assert status == 1
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind - 1] >= pop_array[0, ind]
        if pop_array[0, ind - 1] == pop_array[0, ind]:
            assert pop_array[1, ind - 1] > pop_array[1, ind]

    # r2 and d1,d2 fire --> extinction
    r1, r2, b1, b2, d1, d2 = 0, 1, 0, 0, 1, 1
    rates = np.array([r1, r2, b1, b2, d1, d2])
    n0 = 15
    init_load = np.array([n0], dtype=np.int32)
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=200, seed=0
    )
    assert extflag == 1
    assert status == 1
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind - 1] >= pop_array[0, ind]
        if pop_array[0, ind - 1] == pop_array[0, ind]:
            assert pop_array[1, ind - 1] > pop_array[1, ind]


def test_sm_explosion():
    # r2 and b1 fire --> explosion
    r1, r2, b1, b2, d1, d2 = 0, 1, 1, 0, 0, 0
    rates = np.array([r1, r2, b1, b2, d1, d2])
    n0 = 15
    init_load = np.array([n0], dtype=np.int32)
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=1500, seed=0, t_max=300
    )
    assert extflag == 0
    assert status == 3
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind - 1] >= pop_array[0, ind]
        assert pop_array[1, ind - 1] <= pop_array[1, ind]

    # r2 and b2 fire --> explosion
    r1, r2, b1, b2, d1, d2 = 0, 1, 0, 1, 0, 0
    rates = np.array([r1, r2, b1, b2, d1, d2])
    n0 = 15
    init_load = np.array([n0], dtype=np.int32)
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=1500, seed=0, t_max=300
    )
    assert extflag == 0
    assert status == 3
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind - 1] >= pop_array[0, ind]
        assert pop_array[1, ind - 1] <= pop_array[1, ind]


def test_sm_rmf():
    # Only r1 fires
    r1, r2, b1, b2, d1, d2, rmf = 1, 0, 0, 0, 0, 0, 0
    rates = np.array([r1, r2, b1, b2, d1, d2, rmf])
    init_load = np.array([100], dtype=np.int32)
    # Not until extinction
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier_rmf(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=10, seed=0
    )
    assert extflag == 0
    assert status == -1
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind - 1] > pop_array[0, ind]
        assert pop_array[1, ind] == 0
    # Until extinction
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier_rmf(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=200, seed=0
    )
    assert extflag == 1
    assert status == 1
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        # Population declines
        assert pop_array[0, ind - 1] > pop_array[0, ind]
        assert pop_array[1, ind] == 0

    # Only r2 fires --> conservation
    r1, r2, b1, b2, d1, d2, rmf = 0, 1, 0, 0, 0, 0, 0
    rates = np.array([r1, r2, b1, b2, d1, d2, rmf])
    n0 = 100
    init_load = np.array([n0], dtype=np.int32)
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier_rmf(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=200, seed=0
    )
    assert extflag == 0
    assert status == 2
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind] + pop_array[1, ind] == n0

    # r2 and d1 fire --> extinction
    r1, r2, b1, b2, d1, d2, rmf = 0, 1, 0, 0, 1, 0, 0
    rates = np.array([r1, r2, b1, b2, d1, d2, rmf])
    n0 = 15
    init_load = np.array([n0], dtype=np.int32)
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier_rmf(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=200, seed=0
    )
    assert extflag == 1
    assert status == 1
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind - 1] >= pop_array[0, ind]
        if pop_array[0, ind - 1] == pop_array[0, ind]:
            assert pop_array[1, ind - 1] > pop_array[1, ind]

    # r2 and d1,d2 fire --> extinction
    r1, r2, b1, b2, d1, d2, rmf = 0, 1, 0, 0, 1, 1, 0
    rates = np.array([r1, r2, b1, b2, d1, d2, rmf])
    n0 = 15
    init_load = np.array([n0], dtype=np.int32)
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier_rmf(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=200, seed=0
    )
    assert extflag == 1
    assert status == 1
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind - 1] >= pop_array[0, ind]
        if pop_array[0, ind - 1] == pop_array[0, ind]:
            assert pop_array[1, ind - 1] > pop_array[1, ind]

    # r2 and rmf fire --> extinction
    r1, r2, b1, b2, d1, d2, rmf = 0, 1, 0, 0, 0, 0, 1
    rates = np.array([r1, r2, b1, b2, d1, d2, rmf])
    n0 = 15
    init_load = np.array([n0], dtype=np.int32)
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier_rmf(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=200, seed=0
    )
    assert extflag == 1
    assert status == 1
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind - 1] >= pop_array[0, ind]
        if pop_array[0, ind - 1] == pop_array[0, ind]:
            assert pop_array[1, ind - 1] > pop_array[1, ind]


def test_sm_rmf_explosion():
    # r2 and b1 fire --> explosion
    r1, r2, b1, b2, d1, d2, rmf = 0, 1, 1, 0, 0, 0, 0
    rates = np.array([r1, r2, b1, b2, d1, d2, rmf])
    n0 = 15
    init_load = np.array([n0], dtype=np.int32)
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier_rmf(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=1500, seed=0, t_max=300
    )
    assert extflag == 0
    assert status == 3
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind - 1] >= pop_array[0, ind]
        assert pop_array[1, ind - 1] <= pop_array[1, ind]

    # r2 and b2 fire --> explosion
    r1, r2, b1, b2, d1, d2, rmf = 0, 1, 0, 1, 0, 0, 0
    rates = np.array([r1, r2, b1, b2, d1, d2, rmf])
    n0 = 15
    init_load = np.array([n0], dtype=np.int32)
    (extflag, t, pop_array, t_array, status) = tau_twocomp_carrier_rmf(
        init_load=init_load, rates=rates, imax=float(1e7), nstep=1500, seed=0, t_max=300
    )
    assert extflag == 0
    assert status == 3
    assert np.all(pop_array >= 0)
    for ind in range(1, len(t_array)):
        assert t_array[ind - 1] <= t_array[ind]
        assert pop_array[0, ind - 1] >= pop_array[0, ind]
        assert pop_array[1, ind - 1] <= pop_array[1, ind]
