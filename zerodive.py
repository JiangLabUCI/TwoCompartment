import numpy as np
from staph.utils.tau_twocomp import tau_twocomp_carrier
from staph.utils.data import get_singh_data

npts = 6
nrep = 1000
nstep = 200_000

np.random.seed(0)
seeds = np.random.randint(low=0, high=1e5, size=nrep)

Imax = 10_720_594.783_688_36
bXlist = [3.122_086_50e00, 5.331_199_67e-02, 2.653_476_38e-07, 2.844_684_50e00]
rates = [
    3.122_086_503_498_812_3,
    0.053_311_996_740_339_76,
    20.063_904_757_661_415,
    0.009_187_672_705_231_771,
    17.219_220_257_083_26,
    0.009_187_849_603_656_904,
]  # [r1, r2, b1, b2, d1, d2]


h0, _, _, _, A, H0 = get_singh_data()

init_load = np.array([H0[3]], dtype=np.int32)
tau_twocomp_carrier(init_load, rates, Imax * A, nstep, seeds[836], 6.0, False)
