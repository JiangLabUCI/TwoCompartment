import numpy as np
from ..staph.utils.data import get_singh_data, get_b1d2, get_fcs_fcr, get_k_ec
from ..staph.utils.data import get_kinetic_data_params, get_soap_data, cochran_armitage


def test_singh():
    h0, norig, ntot, tiny, A, H0 = get_singh_data()

    assert A == 3.0
    assert H0.dtype == int
    for ind in range(len(h0)):
        assert H0[ind] == np.int32(h0[ind] * A)
    assert (np.array(norig) <= ntot).all()
    assert tiny < 1


def test_cochran_armitage():
    d = [10, 1000, 10000, 100_000, 1_000_000, 100_000_000]
    p = [0, 0, 9, 6, 20, 2]
    n = [0, 3, 11, 7, 21, 2]
    tol = 1e-3
    zca, pbar, xbar = cochran_armitage(d, p, n)
    assert abs(zca - 3.039) < tol
    assert abs(pbar - 0.841) < tol
    assert abs(xbar - 12.036) < tol


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

    b1, d2 = get_b1d2(b2=3, d1=5, r3=1, r3Imax=0, A=2)
    assert d2 == 4


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


def test_kinetic():
    p = get_kinetic_data_params()
    for ind in range(3):
        this_t = p["t" + str(ind + 1)]
        for each_t in this_t:
            assert each_t in [0, 1, 2, 3, 4, 6]


def test_soap():
    p24 = get_soap_data(dsno=1)
    pimm = get_soap_data(dsno=2)
    assert p24["y0"] > pimm["y0"]
    for each in range(len(p24["y"])):
        assert p24["y"][each] > pimm["y"][each]
    assert p24["sse_rh"] < p24["sse_r1"]
    assert p24["sse_rh"] < p24["sse_rmf"]
    assert p24["aicc_rh"] > p24["aicc_r1"]
    assert p24["aicc_rh"] > p24["aicc_rmf"]
    assert pimm["sse_rh"] < pimm["sse_r1"]
    assert pimm["sse_rh"] < pimm["sse_rmf"]
    assert pimm["aicc_rh"] > pimm["aicc_r1"]
    assert pimm["aicc_rh"] > pimm["aicc_rmf"]
