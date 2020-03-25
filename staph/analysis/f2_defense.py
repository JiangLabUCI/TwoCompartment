import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Polygon
from typing import List, Union
from os import listdir
from ..utils.data import get_kinetic_data_params, get_singh_data
from ..utils.rh_data import get_rh_fit_data
from ..utils.det_models import rh_growth_model, twocomp_model
from ..analysis.igate_ntest import igate
from ..utils.dev import compute_deviance
import matplotlib as mpl


def f2_defense(display: bool = False):
    """Assemble figure 2.

    Assemble figure by calling the appropriate functions.

    Parameters
    ----------
    display
        If `True`, display the plot.
    """
    # part_cols = ["#4daf4a", "#ff7f00", "#e41a1c"]
    part_cols = ["#70a89f", "#fdb462", "#fb8072"]  # colorbrewer 1
    col_mo = ["#1C9E77", "#ff7f00", "#984ea3"]
    sol_inds = [4]
    annotation_args = {"va": "bottom", "weight": "bold", "fontsize": "12"}
    plt.rcParams["font.size"] = 14
    mpl.rcParams["font.family"] = "arial"
    mpl.rcParams["lines.linewidth"] = 2
    plt.figure(figsize=(9, 8))
    plt.subplot(221)
    growth_obj(col_mo, solinds=sol_inds)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    # plt.text(x1 - 0.15 * (x2 - x1), y2, "B", annotation_args)

    plt.subplot(222)
    dr_obj(col_mo, solinds=sol_inds)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    # plt.text(x1 - 0.15 * (x2 - x1), y2, "C", annotation_args)

    plt.subplot(223)
    growth_data_only()

    plt.tight_layout()
    plt.savefig("results/figs/f2_def.pdf")
    if display:
        plt.show()


def growth_data_only():
    p = get_kinetic_data_params()
    plt.plot(p["t1"], p["y1"], "ko-", label="Data")
    plt.plot(p["t2"], p["y2"], "ko-")
    plt.plot(p["t3"], p["y3"], "ko-")
    plt.ylim([0, 8])
    plt.xticks([0, 2, 4, 6])
    plt.xlabel("Time (days)")
    plt.ylabel("$\log_{10}$ Staph density")


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
        fname = "results/ops/" + get_filename(df["desol_inds"][this_sol_ind] + 1)
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

    plt.plot(np.log10(H0), np.array(norig) / 20, "ko", label="Data")
    this_label = f"RH"
    plt.plot(np.log10(H0), pinf_rh, "--", label=this_label, color="grey")
    this_label = f"BP"
    plt.plot(np.log10(H0), bp_presp, "-", label=this_label, color="grey")
    this_label = f"2C, without co-op"
    plt.plot(np.log10(H0), pinfb20, ":", label=this_label, color="xkcd:red")
    for ind1 in range(len(solinds)):
        this_ind = solinds[ind1]
        this_pinf = pinfs[ind1, :]
        this_dev = df.Fst[this_ind]
        this_label = f"2C, with co-op"
        plt.plot(np.log10(H0), this_pinf, label=this_label, color=col[ind1])
    plt.legend(loc="lower right")
    plt.xlabel("$\log_{10}$(bacterial dose)")
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
    plt.rcParams["legend.frameon"] = True
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
    for ind1 in range(0, 1):
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
    plt.xticks([0, 2, 4, 6])
    plt.xlabel("Time (days)")
    plt.ylabel("$\log_{10}$ staph density")

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 0, 1]
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
