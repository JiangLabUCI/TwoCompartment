import numpy as np
from ..staph.utils.tau_twocomp import get_propensity as prop
from ..staph.utils.tau_twocomp_rmf import get_propensity as proprmf


def test_prop():
    # 0 : H -r1-> \\phi
    # 1 : H -r2-> I
    # 2 : I -b1-> I + I
    # 3 : I + I -b2-> I + I + I
    # 4 : I -d1-> \\phi
    # 5 : I + I -d2-> I
    tol = 1e-7
    rates = np.random.rand(6)

    p = prop(rates, 0, 0)
    assert (p == 0).all()

    p = prop(rates, 1, 0)
    assert p[2] == p[3] == p[4] == p[5] == 0
    assert abs(p[0] - rates[0]) < tol
    assert abs(p[1] - rates[1]) < tol

    p = prop(rates, 0, 1)
    assert p[0] == p[1] == p[3] == p[5] == 0
    assert abs(p[2] - rates[2]) < tol
    assert abs(p[4] - rates[4]) < tol

    p = prop(rates, 0, 3)
    assert (p[3] - rates[3] * 3) < tol
    assert (p[5] - rates[5] * 3) < tol


def test_prop_rmf():
    # 0 : H -r1-> \\phi
    # 1 : H -r2-> I
    # 2 : I -b1-> I + I
    # 3 : I + I -b2-> I + I + I
    # 4 : I -d1-> \\phi
    # 5 : I + I -d2-> I
    # 6 : H -rmf-> \\phi
    # 7 : I -rmf-> \\phi
    tol = 1e-7
    rates = np.random.rand(8)

    p = proprmf(rates, 0, 0)
    assert (p == 0).all()

    p = proprmf(rates, 1, 0)
    assert p[2] == p[3] == p[4] == p[5] == p[7] == 0
    assert abs(p[0] - rates[0]) < tol
    assert abs(p[1] - rates[1]) < tol
    assert abs(p[6] - rates[6]) < tol

    p = proprmf(rates, 0, 1)
    assert p[0] == p[1] == p[3] == p[5] == p[6] == 0
    assert abs(p[2] - rates[2]) < tol
    assert abs(p[4] - rates[4]) < tol
    assert abs(p[7] - rates[6]) < tol

    p = proprmf(rates, 0, 3)
    assert (p[3] - rates[3] * 3) < tol
    assert (p[5] - rates[5] * 3) < tol
