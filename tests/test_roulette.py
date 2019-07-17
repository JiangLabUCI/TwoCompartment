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
