import numpy as np
from ..staph.utils.data import get_singh_data, get_b1d2, get_fcs_fcr, get_k_ec


def test_singh():
    h0, norig, ntot, tiny, A, H0 = get_singh_data()

    assert A == 3.0
    assert H0.dtype == int
    for ind in range(len(h0)):
        assert H0[ind] == np.int32(h0[ind] * A)
    assert (np.array(norig) <= ntot).all()
    assert tiny < 1


def test_b1d2():
    b1, d2 = get_b1d2(b2=0, d1=0, r3=0, r3Imax=0)
    assert b1 == 0
    assert d2 == 0

    b1, d2 = get_b1d2(b2=0, d1=0, r3=3, r3Imax=5)
    assert b1 == 5
    assert d2 == 2

    b1, d2 = get_b1d2(b2=3, d1=5, r3=0, r3Imax=0)
    assert b1 == 5
    assert d2 == 3


def test_fcs_fcr():
    for strain in [1, 2]:
        fcs, fcr = get_fcs_fcr(C=0, strain=strain)
        assert fcs == 0
        assert fcr == 0

        ks, EC50s, kr, EC50r = get_k_ec(strain=strain)
        fcs, fcr = get_fcs_fcr(C=EC50s, strain=strain)
        assert fcs == ks / 2
        fcs, fcr = get_fcs_fcr(C=EC50r, strain=strain)
        assert fcr == kr / 2
