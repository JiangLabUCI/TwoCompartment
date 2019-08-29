import numpy as np
import pytest
import matplotlib.pyplot as plt
from ..staph.utils.data import get_bedrail_data, get_occurence_dist
from ..staph.utils.tau_twocomp import tau_twocomp_carrier
from ..staph.utils.tau_twocomp_rmf import tau_twocomp_carrier_rmf
from ..staph.utils.predict import sim_multi, get_stat_time_course
from ..staph.utils.predict import stat_ocprob, get_rates_simfunc


def test_get_rates_simfunc():
    with pytest.raises(AssertionError):
        get_rates_simfunc(r1sind=1, hyp="base", inoc_time="imm")
    with pytest.raises(AssertionError):
        get_rates_simfunc(r1sind=1, hyp="base45", inoc_time="imm")
    rates, simfunc, _, thresh = get_rates_simfunc(
        r1sind=0, hyp="base", inoc_time="base"
    )
    r1 = 1.941_837_212_691_128
    assert simfunc == tau_twocomp_carrier
    assert abs(rates[0] - r1) < 1e-5
    assert abs(thresh - 7_339_343.0) < 1e2

    rates2, simfunc, _, _ = get_rates_simfunc(r1sind=0, hyp="r1s", inoc_time="24h")
    assert simfunc == tau_twocomp_carrier
    assert rates[0] < rates2[0]
    assert abs(rates[0] - rates2[0]) > 1
    for ind in range(1, len(rates)):
        assert abs(rates[ind] - rates2[ind]) < 1e-5

    rates3, simfunc, _, _ = get_rates_simfunc(r1sind=0, hyp="rmf", inoc_time="24h")
    assert simfunc == tau_twocomp_carrier_rmf
    assert abs(rates[0] - rates3[0]) < 1e-5
    for ind in range(1, len(rates)):
        assert abs(rates[ind] - rates3[ind]) < 1e-5
    assert len(rates3) == 7


def test_bedrail_data():
    n = 3
    times, loads, _ = get_bedrail_data(n)
    times2, loads2, _ = get_bedrail_data(n)
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


def test_sim_multi_r1():
    rates, simfunc, Imax, _ = get_rates_simfunc(r1sind=0, hyp="r1s", inoc_time="24h")
    dose_intervals = [0.1, 2.0, 3.0]
    t_max = 6.0
    pop, t, t_array, explosion, extinction, _ = sim_multi(
        simfunc,
        rates=rates,
        dose_intervals=dose_intervals,
        dose_loads=[1000, 200_000, 200_000],
        Imax=Imax,
        t_max=t_max,
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
    assert np.max(t) <= t_max
    assert explosion == 0
    assert extinction == 0
    for ind in range(len(t_array)):
        if ind == len(t_array) - 1:
            t_final = t_max - np.sum(dose_intervals)
        else:
            t_final = dose_intervals[ind + 1]
        assert np.abs((np.max(t_array[ind]) - t_final)) < 1e-5


def test_sim_multi_rmf():
    rates, simfunc, Imax, _ = get_rates_simfunc(r1sind=0, hyp="rmf", inoc_time="24h")
    dose_intervals = np.array([0.3, 2.0, 2.0])
    dose_loads = np.array([100_000, 200_000, 200_000])
    t_max = 10
    pop, t, t_array, explosion, extinction, _ = sim_multi(
        simfunc,
        rates=rates,
        dose_intervals=dose_intervals,
        dose_loads=dose_loads,
        Imax=Imax,
        t_max=t_max,
        nstep=200_000,
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
    assert np.max(t) == t_max
    assert explosion == 0
    assert extinction == 1
    for ind in range(len(t_array)):
        if ind == len(t_array) - 1:
            t_final = t_max - np.sum(dose_intervals)
        else:
            t_final = dose_intervals[ind + 1]
        assert np.abs((np.max(t_array[ind]) - t_final)) < 1e-5


def test_sim_multi_overshoot():
    # Reproduces bug which counts a carrier as an extinction.

    seed = 0
    nrep = 100
    nstep = 200_000

    np.random.seed(seed)
    seeds = np.random.randint(low=0, high=1e5, size=nrep)
    rates, simfunc, Imax, _ = get_rates_simfunc(r1sind=0, hyp="rmf", inoc_time="24h")
    dose_intervals, dose_loads, _ = get_bedrail_data(nrep, tmax=6.0)
    ind1 = 59
    print(dose_intervals[ind1].dtype, dose_loads[ind1].dtype)
    _, t, _, explosion, extinction, _ = sim_multi(
        simfunc, rates, dose_intervals[ind1], dose_loads[ind1], Imax, nstep, seeds[ind1]
    )
    # plt.plot(t, pop[0, :])
    # plt.plot(t, pop[1, :])
    # for ind in range(len(t) - 1):
    #     if t[ind] == t[ind + 1]:
    #         plt.plot(t[ind + 1], pop[0, ind + 1], "ro")
    # plt.show()
    assert (np.max(t) - 6) < 1e-3
    assert extinction == 0
    assert explosion == 0


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


def test_stat_ocprob():
    x = np.array([[1, 2, 3, 1], [1, 2, 3, 2], [1, 2, 3, 3]])
    pres, pcar, ps = stat_ocprob(x)
    assert (pres == np.array([0, 0, 1, 1 / 3])).all()
    assert (pcar == np.array([0, 1, 0, 1 / 3])).all()
    assert (ps == np.array([1, 0, 0, 1 / 3])).all()
