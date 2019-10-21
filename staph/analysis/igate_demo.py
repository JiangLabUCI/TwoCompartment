import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def get_intervals(t, pop, t_req):
    # TODO: document this func
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


def igate(option1: int = 2):
    """
    Model demo plot.

    # TODO: document this func
    """
    filename = "results/demo.npz"
    with np.load(filename, allow_pickle=True) as data:
        t = data["t"]
        pop = data["pop"]
        t_det = data["sol_det_t"]
        pop_det = data["sol_det_y"]
    nrep = int(len(t) / 3)

    alpha_val = 0.5

    if option1 == 1:
        for ind in range(nrep):
            pop[ind][pop[ind] < 0] = np.max(pop[ind])
            plt.step(t[ind], np.log10(pop[ind]), color="xkcd:blue", alpha=alpha_val)

        for ind in range(nrep):
            this_pop = pop[nrep * 2 + ind]
            this_pop[this_pop < 0] = np.max(this_pop)
            plt.step(
                t[nrep * 2 + ind],
                np.log10(this_pop),
                "--",
                color="xkcd:red",
                alpha=alpha_val,
            )

        for ind in range(nrep):
            this_pop = pop[nrep + ind]
            this_pop[this_pop < 0] = np.max(this_pop)
            plt.step(
                t[nrep + ind],
                np.log10(this_pop),
                color="xkcd:green",
                alpha=alpha_val / 2,
            )
        plt.plot(t_det, np.log10(pop_det), "k", linewidth=2)
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

        plt.legend(
            [h4, h3, h2, h1], ["Deterministic", "$b_2=d_1=0$", "$b_2=0$", "$d_1=0$"]
        )

    plt.xlim([0, 3])
    plt.ylim([2.5, 6])
    plt.show()
