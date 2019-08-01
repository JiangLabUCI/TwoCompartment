import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Polygon
from typing import List, Dict
from ..utils.data import get_kinetic_data_params, get_singh_data
from ..utils.rh_data import get_rh_fit_data
from ..utils.det_models import rh_growth_model, twocomp_model


def f2():
    """Assemble figure 2.

    Assemble figure by calling the appropriate functions.
    """
    part_cols = ["#4daf4a", "#ff7f00", "#e41a1c"]
    col_mo = ["#984ea3", "#ff7f00"]
    sol_inds = [0, 5]
    annotation_args = {"va": "bottom", "weight": "bold", "fontsize": "12"}

    plt.figure(figsize=(9, 8))
    plt.subplot(221)
    growth_obj(col_mo, solinds=sol_inds)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    plt.text(x1 - 0.15 * (x2 - x1), y2, "A", annotation_args)

    plt.subplot(222)
    dr_obj(col_mo, solinds=sol_inds)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    plt.text(x1 - 0.15 * (x2 - x1), y2, "B", annotation_args)

    ax = plt.subplot(223)
    filename = "results/predsbasebase2523823dl5r1_1000rep.npz"
    with np.load(filename) as data:
        dose = data["doselist"]
        pinf = data["pinf"]
        pcar = data["pcar"]
        ps = data["ps"]
    partition_plot(dose, pinf[0,], pcar[0,], ps[0,], ax, cols=part_cols, log=True)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    plt.text(x1 - 0.15 * (x2 - x1), y2, "C", annotation_args)

    plt.subplot(224)
    pareto_plot(col_mo, solinds=sol_inds)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    plt.text(x1 - 0.15 * (x2 - x1), y2, "D", annotation_args)

    plt.savefig("results/figs/f2.pdf", bbox_inches="tight")
    plt.show()


def dr_obj(col, solinds=[0]):
    """Plot dose-response data and fits.

    Plots the dose-response data of SA, best fit RH model and two rank 1
    solutions of the 2C model.

    Parameters
    ----------
    col
        Colors of the 2C solutions.
    solinds
        Indices of the rank 1 solutions to plot.
    """
    # Get requisite data
    h0, norig, ntot, tiny, A, H0 = get_singh_data()
    _, dev_rh, int_dose, k = get_rh_fit_data()
    pinf_rh = 1 - np.exp(-int_dose / k)
    df = pd.read_csv("results/rank_1_solutions.csv")
    print(df)
    # Empty container for 2c response probabilities
    pinfs = np.zeros((len(solinds), 6))

    # Get infection probabilities from output files
    for ind1, this_sol_ind in enumerate(solinds):
        fname = "results/ops/ntest.o7721941." + str(df["desol_inds"][this_sol_ind] + 1)
        with open(fname) as f:
            d = f.read()
        d = d.split("\n")
        qstr = "Objective is :  " + str(df.Fst[this_sol_ind])[:-4]
        for ind2, line in enumerate(d):
            if line.startswith(qstr):
                roi = d[ind2 - 6 : ind2]
                break
        pinf = []
        for ind2, this_roi in enumerate(roi):
            temp = this_roi.replace(",", "").split()
            pinf.append(float(temp[5]))
        pinfs[ind1] = pinf

    plt.plot(np.log10(H0), np.array(norig) / 20, "ko", label="Data")
    this_label = f"RH (dev = {round(dev_rh, 2)})"
    plt.plot(np.log10(H0), pinf_rh, "--", label=this_label, color="grey")
    for ind1 in range(len(solinds)):
        this_ind = solinds[ind1]
        this_pinf = pinfs[ind1, :]
        this_dev = df.Fst[this_ind]
        this_label = f"2C (dev = {round(this_dev,2):.2f})"
        plt.plot(np.log10(H0), this_pinf, label=this_label, color=col[ind1])
    plt.legend(loc="lower right")
    plt.xlabel("$Log_{10}$(dose)")
    plt.ylabel("$P_{response}$")


def growth_obj(col: List[str], solinds: List[int] = [0]):
    """Plot growth data and fits.

    Plots the kinetic/growth data of SA, best fit RH model and two rank 1
    solutions of the 2C model.

    Parameters
    ----------
    col
        Colors of the 2C solutions.
    solinds
        Indices of the rank 1 solutions to plot.
    """
    fname = "results//rank_1_solutions.csv"
    df = pd.read_csv(fname)

    p = get_kinetic_data_params()
    sse_rh = get_rh_fit_data()
    sse_rh = sse_rh[0]

    # RH model
    # Solve integration problem
    solrh = solve_ivp(lambda t, y: rh_growth_model(t, y, p), [0, 6], p["initial_rh"])

    # Make plots
    for ind in range(3):
        if ind == 1:
            plt.plot(
                solrh.t,
                np.log10(solrh.y[ind, :].transpose()),
                color="grey",
                linestyle="--",
                label=f"RH (SSE = {round(sse_rh,2)})",
            )
        else:
            plt.plot(
                solrh.t,
                np.log10(solrh.y[ind, :].transpose()),
                color="grey",
                linestyle="--",
            )

    # 2C model
    # For each solution index
    for ind1 in range(len(solinds)):
        this_sol_ind = solinds[ind1]
        p["r1"] = df.r1[this_sol_ind]
        p["r2"] = df.r2[this_sol_ind]
        p["r3"] = df.r3[this_sol_ind]
        p["r3Imax"] = df["r3*Imax"][this_sol_ind]
        sse_2c = df.Fde[this_sol_ind]

        # Solve integration problems
        twoc_t = []
        twoc_y = []
        for ind2 in range(3):
            sol2c = solve_ivp(
                lambda t, y: twocomp_model(t, y, p),
                [0, 6],
                [p["initial_rh"][ind2], 0],
                dense_output=True,
            )
            twoc_t.append(sol2c.t)
            twoc_y.append(np.log10(sum(sol2c.y, 1)))

            if ind2 == 1:
                plt.plot(
                    twoc_t[ind2],
                    twoc_y[ind2],
                    color=col[ind1],
                    label=f"2C (SSE = {round(sse_2c,2)})",
                )
            else:
                plt.plot(twoc_t[ind2], twoc_y[ind2], color=col[ind1])

    # Data

    plt.plot(p["t1"], p["y1"], "ko", label="Data")
    plt.plot(p["t2"], p["y2"], "ko")
    plt.plot(p["t3"], p["y3"], "ko")
    plt.ylim([0, 8])
    plt.xlabel("Time (days)")
    plt.ylabel("Staph. density (CFU/cm$^2$)")

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 0, 1, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])


def partition_plot(
    x: np.ndarray, pinf: np.ndarray, pcar: np.ndarray, ps: np.ndarray, ax, **kwargs
):
    """Plot outcome probabilities.

    Make a partition plot of the outcome probabilities.

    Parameters
    ----------
    x
        List of x. x can be dose or time.
    pinf
        List of infection probabilities.
    pcar
        List of carrier probabilities.
    ps
        List of unaffected probabilities.
    ax
        Axis to plot on.

    Notes
    -----
    """

    if "cols" not in kwargs.keys():
        cols = ["xkcd:green", "xkcd:orange", "xkcd:red"]
    else:
        cols = kwargs["cols"]

    if "log" in kwargs.keys():
        x = np.log10(x)

    vertices = [[0, 0], [0, 1], *zip(x, pinf + pcar + ps), [x[-1], 0]]
    area = Polygon(vertices, color=cols[0], label="Unaffected")
    ax.add_patch(area)
    vertices = [[0, 0], *zip(x, pinf + pcar), [x[-1], 0]]
    area = Polygon(vertices, color=cols[1], label="Carrier")
    ax.add_patch(area)
    vertices = [[0, 0], *zip(x, pinf), [x[-1], 0]]
    area = Polygon(vertices, color=cols[2], label="Ill")
    ax.add_patch(area)
    plt.xlim([0, x[-1]])
    plt.legend(loc="lower right")
    if not ("xlab" in kwargs.keys() and kwargs["xlab"] is False):
        plt.xlabel(r"$\log_{10}$(dose)")
    if not ("ylab" in kwargs.keys() and kwargs["ylab"] is False):
        plt.ylabel("Probability")


def pareto_plot(col, solinds=[0]):
    """Pareto front plot.

    Plot all the solutions and the pareto front.

    Parameters
    ----------
    col
        Colors of the 2C solutions.
    solinds
        Indices of the rank 1 solutions to plot.
    """
    fname = "results/all_solutions.csv"
    df_all = pd.read_csv(fname)
    fname = "results/rank_1_solutions.csv"
    df_r1 = pd.read_csv(fname)

    df_nr1 = df_all[df_all.ranks != 1]
    plt.plot(df_r1.Fde, df_r1.Fst, "o", color="k", label="Rank 1 solutions")
    plt.plot(df_r1.Fde, df_r1.Fst, "-", color="k", label="Pareto front")
    for ind in range(len(solinds)):
        plt.plot(df_r1.Fde[solinds[ind]], df_r1.Fst[solinds[ind]], "o", color=col[ind])
    plt.plot(df_nr1.Fde, df_nr1.Fst, ".", color="k", label="Rank >1 solutions")
    plt.legend()
    plt.xlabel("Growth objective")
    plt.ylabel("Dose-response objective")
