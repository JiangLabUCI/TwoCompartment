import numpy as np
from ..staph.utils.dev import compute_deviance, carrier_obj_wrapper
from ..staph.utils.dev import compute_deviance_hform
from ..staph.utils.dev import compute_devs_min as cdmin
from ..staph.utils.dev import compute_devs_brute as cdbrute
from ..staph.utils.dev import transform_x
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
