import numpy as np
from ..staph.utils.dev import compute_deviance, carrier_obj_wrapper
from ..staph.utils.dev import compute_deviance_hform, get_bF_bX
from ..staph.utils.dev import compute_devs_min as cdmin
from ..staph.utils.dev import compute_devs_brute as cdbrute
from ..staph.utils.dev import transform_x, status_to_pinf, get_consts_bX
from ..staph.utils.data import get_singh_data


def test_compute_deviance():
    _, norig, ntot, _, _, _ = get_singh_data()
    for ind in range(len(norig)):
        compute_deviance(norig[ind] / ntot, ind) == 0
        compute_deviance_hform(norig[ind] / ntot, ind) == 0

    assert compute_deviance(0.5, 5) == -2 * 20 * np.log(0.5)
    assert compute_deviance(0, 0) == -2 * (
        4 * np.log(0.01 * ntot / 4)
        + (ntot - 4) * np.log((1 - 0.01) * ntot / (ntot - 4))
    )
    assert compute_deviance(1, 0) == -2 * (
        4 * np.log(0.99 * ntot / 4)
        + (ntot - 4) * np.log((1 - 0.99) * ntot / (ntot - 4))
    )


def test_stp():
    status = np.zeros(10)
    assert status_to_pinf(status) == 0
    status = np.ones(10) * 3
    assert status_to_pinf(status) == 1.0
    status = np.ones(10) * 4
    assert status_to_pinf(status) == 1.0
    status = np.ones(10) * 5
    assert status_to_pinf(status) == 1.0
    status = np.array([0, 1, 2, 3, 4, 5])
    assert status_to_pinf(status) == 0.5
    status = np.array([6, 7, 8, 3, 4, 5])
    assert status_to_pinf(status) == 0.5


def test_get_bF_bX():
    filename = "results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat"
    bFlist, bXlist = get_bF_bX(filename=filename, desol_ind=[0])
    tol = 1e-12
    assert abs(bXlist[0, 0] - 1.941837212691128) < tol
    assert abs(bXlist[0, 1] - 0.014735846051069906) < tol
    assert abs(bXlist[0, 2] - 2.7098279779742947e-07) < tol
    assert abs(bXlist[0, 3] - 3.1908412905384753) < tol
    for ind in range(len(bFlist) - 1):
        assert bFlist[ind] <= bFlist[ind + 1]


def test_get_consts_bX():
    filename = "results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat"
    bFlist, bXlist = get_bF_bX(filename=filename, desol_ind=[0])
    r1, r2, r3, Imax, modno = get_consts_bX(
        bXlist, ind=[0], filename=filename, verbose=0
    )
    tol = 1e-12
    assert modno == 6
    assert abs(r1 - 1.941837212691128) < tol
    assert abs(r2 - 0.014735846051069906) < tol
    assert abs(r3 - 2.7098279779742947e-07) < tol
    assert abs(Imax - 3.1908412905384753 / 2.7098279779742947e-07) < tol
    bFlist, bXlist = get_bF_bX(filename=filename, desol_ind=[74])
    r1, r2, r3, Imax, _ = get_consts_bX(bXlist, ind=[0], filename=filename, verbose=0)
    assert abs(r1 - 1.6948745105357776) < tol
    assert abs(r2 - 0.010170711716411323) < tol
    assert abs(r3 - 3.9085594907079334e-07) < tol
    filename = filename.replace("6", "3")
    r1, r2, r3, Imax, _ = get_consts_bX(bXlist, ind=[0], filename=filename, verbose=0)
    assert abs(r1 - 1.6948745105357776) < tol
    assert abs(r2 - 0.010170711716411323) < tol
    assert abs(r3 - 3.9085594907079334e-07) < tol
    assert abs(Imax - 3.400300160074247) < tol


def test_minimize():
    x = carrier_obj_wrapper([-1, 2], 1.0, 1.0, 1.0, 1.0, 2, 100, 100, 0, 2, True, None)
    assert x == 3000
    x = carrier_obj_wrapper([1, -2], 1.0, 1.0, 1.0, 1.0, 2, 100, 100, 0, 2, True, None)
    assert x == 3000


def test_cdmin():
    cdmin()


def test_cdbrute():
    cdbrute()


def test_transform():
    x = np.array([1.0, 1.0])
    assert (transform_x(x, t_type=None) == x).all()
    xlogged = np.array([0.1, 1.0])
    assert (transform_x(x, t_type="log") == xlogged).all()
