import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from typing import List, Dict


def make_phase_plot(
    nquivs: int = 15, logflag: bool = False, disp: bool = True, save: bool = False
):
    """Make phase plot.

    Plot a phase diagram of the two-compartment model.

    Parameters
    ----------
    nquivs
        Number of quivers in the x and y directions.
    logflag
        If `True`, make x and y on the log scales. Defaults to `False`.
    disp
        If `True`, display the plot. Defaults to `True`.
    save
        If `True`, save the figure as a PDF. Defaults to `False`.
    """
    # Read data and parameters
    df = pd.read_csv("results/rank_1_solutions.csv")
    index = 4
    p = {}
    p["r1"] = df.r1[index]
    p["r2"] = df.r2[index]
    p["r3"] = df.r3[index]
    p["r3Imax"] = df["r3*Imax"][index]

    # Plot parameters
    lw = 4
    cols = ["#1b9e77", "#d95f02", "#7570b3"]  # colorbrewer 1
    plt.figure(figsize=(4.5, 4))
    if save == True:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 14
        plt.rcParams["lines.linewidth"] = lw
        plt.rcParams["axes.linewidth"] = lw
        plt.rcParams["lines.markersize"] = 10

    # Make quiver data
    if logflag:
        y1 = 10 ** np.linspace(0, 7, 20)
        y2 = 10 ** np.linspace(0, 7, 20)
    else:
        y1 = np.linspace(0.0, 1e7, nquivs)
        y2 = np.linspace(0.0, 1e7, nquivs)
    Y1, Y2 = np.meshgrid(y1, y2)
    t = 0
    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
    NI, NJ = Y1.shape
    for i in range(NI):
        for j in range(NJ):
            if logflag:
                x = Y1[i, j]
                y = Y2[i, j]
            else:
                x = Y1[i, j]
                y = Y2[i, j]
            yprime = twocomp_derivatives([x, y], t, p)
            u[i, j] = yprime[0]
            v[i, j] = yprime[1]

    # Plot quivers
    if logflag:
        Q = plt.quiver(
            np.log10(Y1),
            np.log10(Y2),
            np.log10(u + np.min(u) + 1),
            np.log10(v + np.min(v) + 1),
            color="0.5",
        )
    else:
        Q = plt.quiver(Y1, Y2, u, v, color="0.5", linewidth=lw)

    plt.plot(0, 0, "o", label="Critical point", color=cols[1])
    plt.plot(0, p["r3Imax"] / p["r3"], "o", color=cols[1])

    # Starting points
    y10s = [1e7]
    y20s = [0.2e7]

    for ind in range(len(y10s)):
        # y10 = ind * (10 ** 6)
        # print(y10)
        tspan = np.linspace(0, 6, 20)
        y0 = [y10s[ind], y20s[ind]]
        ys = odeint(lambda Y, t: twocomp_derivatives(Y, t, p), y0, tspan)
        if logflag:
            plt.plot(np.log10(ys[:, 0]), np.log10(ys[:, 1]), "-", color=cols[0])  # path
            plt.plot(
                np.log10([ys[0, 0]]),
                np.log10([ys[0, 1]]),
                "o",
                label="Start",
                color=cols[0],
            )  # start
            plt.plot(
                np.log10([ys[-1, 0]]),
                np.log10([ys[-1, 1]]),
                ">",
                label="End",
                color=cols[0],
            )  # end
        else:
            plt.plot(ys[:, 0], ys[:, 1], "-", label="Trajectory", color=cols[0])  # path
            plt.plot([ys[0, 0]], [ys[0, 1]], "o", label="Start", color=cols[0])  # start
            plt.plot(
                [ys[-1, 0]], [ys[-1, 1]], "bx", mew=3, label="End", color=cols[0]
            )  # end
        # print(ys)

    plt.xlabel("S1")
    plt.ylabel("S2")
    plt.legend(frameon=True, loc="upper right")
    plt.xticks(ticks=[0, 0.5e7, 1e7])
    plt.yticks(ticks=[0, 0.5e7, 1e7])
    # plt.xlim([-2, 8])
    # plt.ylim([-4, 4])
    if disp == True:
        plt.show()
    if save == True:
        plt.tight_layout()
        plt.savefig("results/imgs/phase_plot.pdf", transparent=True)


def twocomp_derivatives(Y: List[float], t: float, p: Dict) -> List[float]:
    """Return 2C derivatives.

    Parameters
    ----------
    Y
        Model states.
    t
        Time point in the simulation.
    p
        Parameters of the 2C model.
    
    Returns
    -------
    ders
        The derivatives of the 2C model.
    """
    H, I = Y
    hprime = -p["r1"] * H - p["r2"] * H
    iprime = p["r2"] * H + p["r3Imax"] * I - p["r3"] * I * I
    ders = [hprime, iprime]
    return ders

