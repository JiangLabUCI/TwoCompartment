from os import listdir
from typing import List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from scipy.integrate import solve_ivp

from ..analysis.igate_ntest import igate
from ..utils.data import get_kinetic_data_params, get_singh_data
from ..utils.det_models import rh_growth_model, twocomp_model
from ..utils.dev import compute_deviance
from ..utils.rh_data import get_rh_fit_data


def f2(display: bool = False):
    """Assemble figure 2.

    Assemble figure by calling the appropriate functions.

    Parameters
    ----------
    display
        If `True`, display the plot.
    """
    # part_cols = ["#4daf4a", "#ff7f00", "#e41a1c"]
    part_cols = ["#70a89f", "#fdb462", "#fb8072"]  # colorbrewer 1
    col_mo = ["#1b9e77", "#d95f02"]
    sol_inds = [0, 4]
    annotation_args = {"va": "bottom", "weight": "bold", "fontsize": "12"}

    plt.figure(figsize=(9, 8))
    plt.subplot(221)
    pareto_plot(col_mo, solinds=sol_inds)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    plt.text(x1 - 0.15 * (x2 - x1), y2, "a", annotation_args)

    ax = plt.subplot(222)
    fname = "results//rank_1_solutions.csv"
    df = pd.read_csv(fname)
    growth_obj(df, col_mo, ax, solinds=sol_inds)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    plt.text(x1 - 0.15 * (x2 - x1), y2, "b", annotation_args)

    ax = plt.subplot(223)
    dr_obj(col_mo, solinds=sol_inds)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    plt.text(x1 - 0.15 * (x2 - x1), y2, "c", annotation_args)

    ax = plt.subplot(224)
    filename = "results/predsbasebase2523823dl" + str(sol_inds[1])
    filename += "r1_1000rep.npz"
    with np.load(filename) as data:
        dose = data["doselist"]
        pinf = data["pinf"]
        pcar = data["pcar"]
        ps = data["ps"]
    partition_plot(dose, pinf[0,], pcar[0,], ps[0,], ax, cols=part_cols, log=True)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    plt.text(x1 - 0.15 * (x2 - x1), y2, "d", annotation_args)

    plt.tight_layout()
    plt.savefig("results/figs/f2.pdf")
    if display:
        plt.show()


def dr_obj(col, solinds=[0], ax=None):
    """Plot dose-response data and fits.

    Plots the dose-response data of SA, best fit RH model and two rank 1
    solutions of the 2C model.

    Parameters
    ----------
    col
        Colors of the 2C solutions.
    solinds
        Indices of the rank 1 solutions to plot.
    ax
        The axis object to plot the growth objective on.
    """

    if ax is None:
        ax = plt.gca()

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
        fname = "results/ops/"
        fname += get_filename(df["desol_inds"][this_sol_ind] + 1)
        print(f"Fname is : {fname}")
        with open(fname) as f:
            d = f.read()
        d = d.split("\n")
        qstr = f"Which gives best dev of : {df.Fst[this_sol_ind]:.4f}"
        for ind2, line in enumerate(d):
            if line.startswith(qstr):
                roi = d[ind2 + 2]
                break
        print(ind1, qstr)
        assert roi.startswith("p_res is : ")
        roi = roi[12:]
        roi = roi.replace("]", "").split()
        print(roi)
        pinf = [float(this_roi) for this_roi in roi]
        pinfs[ind1] = pinf
    devb20, _, _, pinfb20 = igate(filenames=["ntest.o9717095.59"], option1=4)

    # best fit beta-Poisson
    bp_alpha = 0.18511
    bp_beta = 28.9968
    bp_presp = []
    bp_dev = 0
    for ind in range(len(H0)):
        bp_presp.append(1 - (1 + H0[ind] / bp_beta) ** (-bp_alpha))
        bp_dev += compute_deviance(bp_presp[ind], ind)
    print("beta poisson deviance : ", bp_dev)

    ax.plot(np.log10(H0), np.array(norig) / 20, "ko", label="Data")
    this_label = "RH ($f_{\mathrm{dev}}$= " + f"{dev_rh:.2f})"
    ax.plot(np.log10(H0), pinf_rh, "--", label=this_label, color="grey")
    this_label = "approx. BP ($f_{\mathrm{dev}}$= " + f"{bp_dev:.2f})"
    ax.plot(np.log10(H0), bp_presp, "-", label=this_label, color="grey")
    this_label = "2C, $b_2$=0 ($f_{\mathrm{dev}}$= " + f"{devb20:.2f})"
    ax.plot(np.log10(H0), pinfb20, ":", label=this_label, color="xkcd:red")
    for ind1 in range(len(solinds)):
        this_ind = solinds[ind1]
        this_pinf = pinfs[ind1, :]
        this_dev = df.Fst[this_ind]
        this_label = "2C, $d_1$=0 ($f_{\mathrm{dev}}$= " + f"{this_dev:.2f})"
        ax.plot(np.log10(H0), this_pinf, label=this_label, color=col[ind1])
    ax.legend(loc="lower right")
    ax.set_xlabel("$\log_{10}$(dose)")
    ax.set_ylabel("$P_{\mathrm{response}}$")


def growth_obj(
    df: pd.DataFrame,
    col: List[str],
    ax: mpl.axis,
    solinds: List[int] = [0],
    obj_name: str = "SSE",
):
    """Plot growth data and fits.

    Plots the kinetic/growth data of SA, best fit RH model and a subset of
    the solutions in df whose indices are given by the ``solinds`` argument.

    Parameters
    ----------
    df
        The dataframe to plot the solutions from.
    col
        Colors of the 2C solutions.
    ax
        The axis object to plot the growth objective on.
    solinds
        Indices of the rank 1 solutions to plot.
    obj_name
        The name of the objective function to display in the legend.
    """
    p = get_kinetic_data_params()
    sse_rh = get_rh_fit_data()
    sse_rh = sse_rh[0]

    # RH model
    # Solve integration problem
    solrh = solve_ivp(lambda t, y: rh_growth_model(t, y, p), [0, 6], p["initial_rh"])

    # Make plots
    for ind in range(3):
        if ind == 1:
            ax.plot(
                solrh.t,
                np.log10(solrh.y[ind, :].transpose()),
                color="grey",
                linestyle="--",
                label=f"RH ({obj_name} = {round(sse_rh,2)})",
            )
        else:
            ax.plot(
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
                ax.plot(
                    twoc_t[ind2],
                    twoc_y[ind2],
                    color=col[ind1],
                    label=f"2C, rank {this_sol_ind + 1} ({obj_name} = {round(sse_2c,2)})",
                )
            else:
                ax.plot(twoc_t[ind2], twoc_y[ind2], color=col[ind1])

    # Data

    ax.plot(p["t1"], p["y1"], "ko", label="Data")
    ax.plot(p["t2"], p["y2"], "ko")
    ax.plot(p["t3"], p["y3"], "ko")
    ax.set_ylim([0, 8])
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("SA density (CFU/cm$^2$)")

    handles, labels = ax.get_legend_handles_labels()
    order = [3, 1, 2, 0]
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        frameon=True,
        fontsize=11,
    )


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
    plt.rcParams["legend.frameon"] = True
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
    area = Polygon(vertices, color=cols[2], label="Response")
    ax.add_patch(area)
    plt.xlim([0, x[-1]])
    plt.legend(loc="lower right")
    if not ("xlab" in kwargs.keys() and kwargs["xlab"] is False):
        plt.xlabel(r"$\log_{10}$(dose)")
    if not ("ylab" in kwargs.keys() and kwargs["ylab"] is False):
        plt.ylabel("Probability")


def pareto_plot(col, solinds=[0], ax=None):
    """Pareto front plot.

    Plot all the solutions and the pareto front.

    Parameters
    ----------
    col
        Colors of the 2C solutions.
    solinds
        Indices of the rank 1 solutions to plot.
    ax
        The axis object to plot the growth objective on.
    """

    if ax is None:
        ax = plt.gca()

    fname = "results/all_solutions.csv"
    df_all = pd.read_csv(fname)
    fname = "results/rank_1_solutions.csv"
    df_r1 = pd.read_csv(fname)

    df_nr1 = df_all[df_all.ranks != 1]
    r1_marker = "s"
    ax.plot(df_r1.Fde, df_r1.Fst, r1_marker, color="k", label="Pareto rank 1")
    ax.plot(df_r1.Fde, df_r1.Fst, "-", color="k", label="Pareto front")
    for ind in range(len(solinds)):
        ax.plot(
            df_r1.Fde[solinds[ind]], df_r1.Fst[solinds[ind]], r1_marker, color=col[ind]
        )
    ax.plot(df_nr1.Fde, df_nr1.Fst, ".", color="k", label="Pareto rank >1")
    ax.legend()
    ax.set_xlabel("Growth objective ($f_{\mathrm{SSE}}$)")
    ax.set_ylabel("Dose-response objective ($f_{\mathrm{dev}}$)")


def get_filename(task_no: int = 0) -> Union[str, None]:
    """Get output filename.

    For a given task number, get the file name of where the output is saved.
    Task number = DE solution number + 1.

    Parameters
    ----------
    task_no
        Task number.

    Returns
    -------
    filename
        Output file name.
    """
    filenames = listdir("results/ops/")
    filename = None
    for a_file in filenames:
        if a_file.endswith("." + str(task_no)):
            filename = a_file
            break
    return filename
