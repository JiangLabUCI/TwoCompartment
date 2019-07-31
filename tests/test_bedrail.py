import numpy as np
import matplotlib.pyplot as plt
from ..staph.utils.data import get_bedrail_data, get_occurence_dist
from ..staph.utils.tau_twocomp import tau_twocomp_carrier
from ..staph.utils.tau_twocomp_rmf import tau_twocomp_carrier_rmf
from ..staph.utils.predict import get_rates, sim_multi, get_stat_time_course


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


def test_get_rates():
    rates_rmf, Imax_rmf = get_rates(hyp="rmf")
    rates_r1, Imax_r1 = get_rates(hyp="r1*")
    assert len(rates_rmf) == 7
    assert len(rates_r1) == 6
    assert rates_rmf[1] == rates_r1[1]  # r2
    assert rates_rmf[2] == rates_r1[2]  # b1
    assert rates_rmf[3] == rates_r1[3]  # b2
    assert rates_rmf[4] == rates_r1[4]  # d1
    assert rates_rmf[5] == rates_r1[5]  # d2
    assert Imax_rmf == Imax_r1


def test_sim_multi_r1():
    rates, Imax = get_rates("r1*")
    dose_intervals = [0.1, 2.0, 3.0]
    pop, t, t_array, explosion, extinction = sim_multi(
        tau_twocomp_carrier,
        rates=rates,
        dose_intervals=dose_intervals,
        dose_loads=[100_000, 200_000, 200_000],
        Imax=Imax,
    )
    tdiff = t[1:] - t[:-1]
    # plt.plot(t, pop[0, :])
    # plt.plot(t, pop[1, :])
    # for ind in range(len(t) - 1):
    #     if t[ind] == t[ind + 1]:
    #         plt.plot(t[ind + 1], pop[1, ind + 1], "ro")
    # plt.show()
    assert t.shape[0] == pop.shape[1]
    assert (tdiff >= 0).all()
    for ind in range(len(t_array)):
        if ind == len(t_array) - 1:
            t_final = np.max(t_array[ind])
        else:
            t_final = dose_intervals[ind + 1]
        assert np.abs((np.max(t_array[ind]) - t_final)) < 1e-5
    assert explosion == 1
    assert extinction == 0


def test_sim_multi_rmf():
    rates, Imax = get_rates("rmf")
    dose_intervals = [0.1, 2.0, 2.0]
    t_max = 10
    pop, t, t_array, explosion, extinction = sim_multi(
        tau_twocomp_carrier_rmf,
        rates=rates,
        dose_intervals=dose_intervals,
        dose_loads=[100_000, 200_000, 200_000],
        Imax=Imax,
        t_max=t_max,
    )

    tdiff = t[1:] - t[:-1]
    # plt.plot(t, pop[0, :])
    # plt.plot(t, pop[1, :])
    # for ind in range(len(t) - 1):
    #     if t[ind] == t[ind + 1]:
    #         plt.plot(t[ind + 1], pop[0, ind + 1], "ro")
    # plt.show()
    assert t.shape[0] == pop.shape[1]
    assert (tdiff >= 0).all()
    for ind in range(len(t_array)):
        if ind == len(t_array) - 1:
            t_final = t_max - np.sum(dose_intervals[1:])
        else:
            t_final = dose_intervals[ind + 1]
        assert np.abs((np.max(t_array[ind]) - t_final)) < 1e-5
    assert explosion == 0
    assert extinction == 1


def test_stat_time_course():
    tsim = np.linspace(0, 3, 20)
    tref = np.linspace(0, 5, 10)
    pop = np.zeros(len(tsim))
    sstat = get_stat_time_course(tsim=tsim, pop=pop, tref=tref, thresh=10)
    assert (sstat == 1).all()
    assert len(sstat) == len(tref)

    pop = np.ones(len(tsim)) * 5
    sstat = get_stat_time_course(tsim=tsim, pop=pop, tref=tref, thresh=10)
    assert (sstat == 2).all()
    assert len(sstat) == len(tref)
    sstat = get_stat_time_course(tsim=tsim, pop=pop, tref=tref, thresh=2)
    assert (sstat == 3).all()
    assert len(sstat) == len(tref)

    tsim = np.linspace(0, 3, 12)
    tref = np.linspace(0, 3, 6)
    pop = np.array([0, 0, 5, 5, 15, 15, 0, 0, 5, 5, 15, 15])
    sstat = get_stat_time_course(tsim=tsim, pop=pop, tref=tref, thresh=10)
    assert (sstat == [1, 2, 3, 1, 2, 3]).all()

    tsim = np.linspace(0, 5, 11)
    tref = np.linspace(0, 10, 6) + 0.1
    pop = np.array([0, 0, 5, 5, 15, 20, 25, 30, 35, 40, 40])
    # print("")
    # print(list(zip(tsim, pop)))
    # print(tref)
    sstat = get_stat_time_course(tsim=tsim, pop=pop, tref=tref, thresh=10)
    assert (sstat == [1, 3, 3, 3, 3, 3]).all()

    pop = np.array([0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0])
    sstat = get_stat_time_course(tsim=tsim, pop=pop, tref=tref, thresh=10)
    assert (sstat == [1, 1, 1, 1, 1, 1]).all()
