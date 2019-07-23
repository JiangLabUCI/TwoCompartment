import numpy as np
from ..staph.utils.roulette import roulette as rou
from ..staph.utils.roulette import simple_system as simp


def test_roulette_zeros():
    # Test if correct choice is returned for pre-determined systems
    n = 10
    for ind in range(n):
        temp = np.zeros(n)
        temp[ind] = n
        assert rou(temp) == ind
    temp = np.zeros(4)
    temp[0] = 1
    temp[3] = 1
    for ind in range(n):
        assert rou(temp) in [0, 3]


def test_roulette_rep():
    # Test if roulette is reproducible.
    choices = simp(0)
    assert (choices == choices[0]).all()


def test_proportion():
    # Test if roulette returns the right proportions
    n1 = 10
    n2 = 1
    nrep = 10000
    temp = np.array([n1, n2, 0])
    choice = np.zeros(nrep)
    for ind in range(nrep):
        choice[ind] = rou(temp)
    p1 = len(choice[choice == 0]) / nrep
    p2 = len(choice[choice == 1]) / nrep
    assert np.abs(p1 - n1 / (n1 + n2)) < 1e-2
    assert np.abs(p2 - n2 / (n1 + n2)) < 1e-2

    temp = np.array([0, n1, n2])
    choice = np.zeros(nrep)
    for ind in range(nrep):
        choice[ind] = rou(temp)
    p1 = len(choice[choice == 1]) / nrep
    p2 = len(choice[choice == 2]) / nrep
    assert np.abs(p1 - n1 / (n1 + n2)) < 1e-2
    assert np.abs(p2 - n2 / (n1 + n2)) < 1e-2

    temp = np.array([n1, 0, n2])
    choice = np.zeros(nrep)
    for ind in range(nrep):
        choice[ind] = rou(temp)
    p1 = len(choice[choice == 0]) / nrep
    p2 = len(choice[choice == 2]) / nrep
    assert np.abs(p1 - n1 / (n1 + n2)) < 1e-2
    assert np.abs(p2 - n2 / (n1 + n2)) < 1e-2
