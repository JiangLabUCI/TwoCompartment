import numpy as np
import matplotlib.pyplot as plt
from ..staph.utils.data import get_bedrail_data, get_occurence_dist
from ..staph.utils.tau_twocomp import tau_twocomp_carrier
from ..staph.utils.tau_twocomp_rmf import tau_twocomp_carrier_rmf

# from ..staph.analysis.predict_carrier import sim_multi, get_rates


def test_bedrail_data():
    n = 3
    times, loads = get_bedrail_data(n)
    times2, loads2 = get_bedrail_data(n)
    for ind in range(n):
        assert np.all(times[ind].shape == loads[ind].shape)
        assert np.all(loads[ind].dtype == np.int32)
        assert np.all(times[ind] == times2[ind])
        assert np.all(loads[ind] == loads2[ind])


def test_occ():
    x = get_occurence_dist(10000)
    print(f"Mean = {np.mean(x):.2f}")
    print(f"Std = {np.std(x):.2f}")
    print(f"Max = {np.max(x):.2f}")
    print(f"Min = {np.min(x):.2f}")


def test_contact_freq():
    lam = 3
    inter_times = np.random.exponential(scale=1 / lam, size=100)
    assert (sum(inter_times) / 100 - lam) < 1e-2


# def test_sim_multi():
#     rates, Imax = get_rates("r1*")
#     dose_intervals = [0.1, 2.0, 3.0]
#     print(type(Imax))
#     pop, t, t_array, explosion, extinction = sim_multi(
#         tau_twocomp_carrier,
#         rates=rates,
#         dose_intervals=dose_intervals,
#         dose_loads=[100_000, 200_000, 200_000],
#         Imax=Imax,
#     )

#     tdiff = t[1:] - t[:-1]
#     # plt.plot(t, pop[0, :])
#     # plt.plot(t, pop[1, :])
#     # plt.show()
#     assert (tdiff >= 0).all()
#     for ind in range(len(t_array)):
#         assert np.abs((np.max(t_array[ind]) - dose_intervals[ind])) < 1e-5
#     assert explosion == 0
#     assert extinction == 0

#     rates, Imax = get_rates("rmf")
#     dose_intervals = [0.1, 2.0, 3.0]
#     print(type(Imax))
#     pop, t, t_array, explosion, extinction = sim_multi(
#         tau_twocomp_carrier_rmf,
#         rates=rates,
#         dose_intervals=dose_intervals,
#         dose_loads=[100_000, 200_000, 200_000],
#         Imax=Imax,
#     )

#     tdiff = t[1:] - t[:-1]
#     # plt.plot(t, pop[0, :])
#     # plt.plot(t, pop[1, :])
#     # plt.show()
#     assert (tdiff >= 0).all()
#     for ind in range(len(t_array)):
#         assert np.abs((np.max(t_array[ind]) - dose_intervals[ind])) < 1e-5
#     assert explosion == 0
#     assert extinction == 1
