import numpy as np
from ..staph.utils.pareto_rank import get_pareto_ranks


def test_1d():
    X = np.array([2, 3, 4, 1])
    X.shape = (4, 1)
    ranks = get_pareto_ranks(X)
    assert np.all(ranks == [2, 3, 4, 1])


def test_2d():
    X = np.array([[1, 1], [2, 3], [0.5, 1.5], [1.5, 0.5]])
    ranks = get_pareto_ranks(X)
    assert np.all(ranks == [1, 2, 1, 1])
