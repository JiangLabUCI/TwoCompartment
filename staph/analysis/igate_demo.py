import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib import cm


def get_intervals(t, pop, t_req):
    """Get intervals for plotting.

    Get the lower and upper limits of variation at required times points.
    These are calculated as one standard deviation below and above the mean.

    Parameters
    ----------
    t
        The list of simulation times for each iteration.
    pop
        The list of total population size (adapted + unadapted) for each
        iteration.
    t_req
        Time points at which the intervals are required.

    Returns
    -------
    lower_int
        Lower limit of variation.
    upper_int
        Upper limit of variation.
    """
    nrep = len(t)
    pop_req = np.zeros([nrep, len(t_req)])
    for ind in range(nrep):
        # print(ind)
        interp_fn = interp1d(t[ind], pop[ind])
        try:
            this_pop_req = interp_fn(t_req)
        except ValueError:
            print(t[ind], t_req)
        pop_req[ind, :] = this_pop_req
    upper_int = np.zeros(len(t_req))
    lower_int = np.zeros_like(upper_int)

    for ind in range(len(t_req)):
        this_mean = np.mean(pop_req[:, ind])
        this_sd = np.std(pop_req[:, ind])
        upper_int[ind] = this_mean + this_sd
        lower_int[ind] = this_mean - this_sd

    return lower_int, upper_int


def igate(option1: int = 2, disp: bool = True):
    """
    Model demo plot.

    Demonstrate the effect of b2 and d1 by plotting how much simulations vary
    over the timecourse for determinstic case, b2 = d1 = 0, b2 = 0 and d1 = 0.

    Parameters
    ----------
    option1
        If 1, plot the time series itself. If 2, plot 1 SD of the time series.
    disp
        If `True`, display the plot.
    """
    filename = "results/demo.npz"
    with np.load(filename, allow_pickle=True) as data:
        t = data["t"]
        pop = data["pop"]
        t_det = data["sol_det_t"]
        pop_det = data["sol_det_y"]
    nrep = int(len(t) / 3)

    alpha_val = 1.0
    cols = plt.get_cmap("Set2")
    if option1 == 1:

        for ind in range(nrep):
            this_pop = pop[nrep + ind]
            this_pop[this_pop < 0] = np.max(this_pop)
            h1, = plt.step(
                t[nrep + ind], np.log10(this_pop), color=cols(0), alpha=alpha_val
            )

        for ind in range(nrep):
            this_pop = pop[nrep * 2 + ind]
            this_pop[this_pop < 0] = np.max(this_pop)
            h2, = plt.step(
                t[nrep * 2 + ind], np.log10(this_pop), color=cols(1), alpha=alpha_val
            )

        for ind in range(nrep):
            pop[ind][pop[ind] < 0] = np.max(pop[ind])
            h3, = plt.step(t[ind], np.log10(pop[ind]), color=cols(2), alpha=alpha_val)

        h4, = plt.plot(t_det, np.log10(pop_det), "k", linewidth=2)
        ordered_handles = [h4, h3, h2, h1]
    elif option1 == 2:
        old_t_det = t_det
        t_det = t_det[t_det < 3.5]

        ind1 = nrep
        ind2 = 2 * nrep - 1
        lower2, upper2 = get_intervals(t[ind1:ind2], pop[ind1:ind2], t_det)
        lower2[lower2 <= 0] = 1
        h1 = plt.fill_between(t_det, np.log10(lower2), np.log10(upper2))  # , alpha=0.5)

        ind1 = 2 * nrep
        ind2 = 3 * nrep - 1
        lower3, upper3 = get_intervals(t[ind1:ind2], pop[ind1:ind2], t_det)
        lower3[lower3 <= 0] = 1
        h2 = plt.fill_between(t_det, np.log10(lower3), np.log10(upper3))  # , alpha=0.5)

        lower1, upper1 = get_intervals(t[:nrep], pop[:nrep], t_det)
        h3 = plt.fill_between(t_det, np.log10(lower1), np.log10(upper1))  # , alpha=0.5)

        h4, = plt.plot(old_t_det, np.log10(pop_det), "k", linewidth=2)

        ordered_handles = [h4, h3, h2, h1]
    plt.legend(ordered_handles, ["Deterministic", "$b_2=d_1=0$", "$b_2=0$", "$d_1=0$"])

    plt.xlim([0, 3])
    plt.ylim([2.5, 6])
    plt.xticks(ticks=[0, 1, 2, 3])
    plt.yticks(ticks=[3, 4, 5, 6])
    plt.xlabel("Time (days)")
    plt.ylabel("$\log_{10}$(bacterial load)")

    if disp == True:
        plt.show()
