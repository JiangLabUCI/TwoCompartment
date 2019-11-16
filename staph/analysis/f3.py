import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import List
from ..utils.data import get_soap_data
from ..utils.det_models import twocomp_rmf_model
from .f2 import rh_growth_model, twocomp_model, partition_plot
import matplotlib as mpl
from matplotlib.gridspec import GridSpec


def label(xlab: str = "", ylab: str = "", label: str = "", factor: float = 0.15):
    """Label plots.

    Handy function to label plots.

    Parameters
    ----------
    xlab
        X axis label for the plot.
    ylab
        Y axis label for the plot.
    label
        Subfigure label for the plot.
    factor
        Factor by which to left-shift label.
    """
    annotation_args = {
        "va": "bottom",
        "weight": "bold",
        "fontsize": "12",
        "family": "arial",
    }
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    plt.text(x1 - factor * (x2 - x1), y2, label, annotation_args)


def cm2inch(cm):
    """Convert cm to inch.

    Parameters
    ----------
    cm
        Value in centimeters.

    Returns
    -------
    inch
        Value in inches.

    """
    inch = cm / 2.54
    return inch


def f3(display: bool = False):
    """Assemble figure 3.

    Assemble figure 3 using gridspec for improved layout.

    Parameter
    ---------
    display
        If `True`, display the plot.
    """
    fnames = [
        "results/predsr1s24h2523823dl4r1_1000rep.npz",
        "results/predsrmf24h2523823dl4r1_1000rep.npz",
    ]
    col_mo = ["#984ea3", "#ff7f00"]
    cols = ["#4daf4a", "#ff7f00", "#e41a1c"]
    cols = ["#70a89f", "#fdb462", "#fb8072"]  # colorbrewer 1

    mpl.rcParams["font.family"] = "arial"
    lef, rig = 0.08, 0.99
    bot, top = 0.11, 0.95
    hs = 0.3

    fig = plt.figure(1, figsize=(9, 4))
    gs1 = GridSpec(2, 2, top=top, bottom=bot, left=lef, right=rig, hspace=hs)
    ax = fig.add_subplot(gs1[0:2, 0])
    soap_obj(col_mo)
    label(xlab=ax.get_xlabel(), ylab=ax.get_ylabel(), label="A")

    labs = ["B", "C"]
    for ind1, filename in enumerate(fnames):
        with np.load(filename) as data:
            dose = data["doselist"]
            pinf = data["pinf"]
            pcar = data["pcar"]
            ps = data["ps"]
        ax = fig.add_subplot(gs1[ind1, 1])
        if ind1 == 0:
            partition_plot(dose, pinf[0,], pcar[0,], ps[0,], ax, cols=cols, log=True)
            label(ylab=ax.get_ylabel(), label=labs[ind1])
            ax.get_xaxis().set_visible(False)
        elif ind1 == 1:
            partition_plot(dose, pinf[0,], pcar[0,], ps[0,], ax, cols=cols, log=True)
            label(xlab=ax.get_xlabel(), ylab=ax.get_ylabel(), label=labs[ind1])
            ax.legend_.remove()
    plt.savefig("results/figs/f3.pdf")

    fig = plt.figure(2, figsize=(9, 4))
    right_fact = 0.11
    gs2 = GridSpec(
        2, 2, top=top, bottom=bot, left=lef, right=rig, hspace=hs, wspace=0.15
    )
    fnames = [
        "results/pred_1000rep400000nstr1shypF6_multi.npz",
        "results/pred_1000rep400000nstrmfhypF6_multi.npz",
    ]
    labs1 = ["A", "B"]
    labs2 = ["C", "D"]
    for ind1, filename in enumerate(fnames):
        with np.load(filename, allow_pickle=True) as data:
            pres = data["pres"]
            pcar = data["pcar"]
            ps = data["ps"]
            tref = data["tref"]
            popH = data["popH"]
            popI = data["popI"]
            t = data["t"]
            new_ext = data["new_ext"]
            new_exp = data["new_exp"]
            sim_stop_thresh = data["sim_stop_thresh"]
            thresh = data["thresh"]

        # Plot population vs. time
        ax = fig.add_subplot(gs2[0, ind1])
        pop_time(
            t,
            popH,
            popI,
            new_ext,
            new_exp,
            log=True,
            alpha=0.8,
            nplot=2,
            cols=cols,
            plot_thresh=thresh,
            sim_stop_thresh=sim_stop_thresh,
        )
        if ind1 == 0:
            label(xlab="Time (days)", ylab=ax.get_ylabel(), label=labs1[ind1])
        elif ind1 == 1:
            label(xlab="Time (days)", label=labs1[ind1], factor=right_fact)

        # Plot probability vs. time
        ax = fig.add_subplot(gs2[1, ind1])
        plt.rcParams["legend.frameon"] = True
        if ind1 == 0:
            partition_plot(tref, pres, pcar, ps, ax, cols=cols)
            label(xlab="Time (days)", ylab=ax.get_ylabel(), label=labs2[ind1])
        elif ind1 == 1:
            partition_plot(tref, pres, pcar, ps, ax, cols=cols)
            label(xlab=ax.get_xlabel(), label=labs2[ind1], factor=right_fact)
        ax.legend_.remove()

    plt.savefig("results/figs/f4.pdf")
    if display:
        plt.show()


def pop_time(
    t: List[np.ndarray],
    popH: List[np.ndarray],
    popI: List[np.ndarray],
    extinction: List[int],
    explosion: List[int],
    plot_thresh: float,
    sim_stop_thresh: float,
    nplot: int = 1,
    cols: List[str] = ["#4daf4a", "#ff7f00", "#e41a1c"],
    alpha: float = 1.0,
    log: bool = False,
):
    """Plot time course.

    Plot the time course of populations for each outcome type.

    Parameters
    ----------
    t
        List of time points.
    popH
        List of H time courses.
    popI
        List of I time courses.
    extinction
        List of ultimate extinction flags. 1 if pop finally went extinct for
        that repetition, 0 otherwise.
    explosion
        List of ultimate explosion flags. 1 if pop finally exploded for
        that repetition, 0 otherwise.
    plot_thresh
        Threshold value used to compute outcome probabilities.
    sim_stop_thresh
        Threshold value of the simulation at which it is stopped.
    nplot
        Number of time courses of each outcome type to plot.
    cols
        Colors to be used for each outcome type.
    alpha
        Transparency of each time course.
    log
        If `True`, plot the log of the population. If `False`, plot the population.
    """
    nrep = len(t)
    extcount, expcount, carcount = 0, 0, 0
    y_upper = 0
    handles = [0 for ind in range(4)]
    for ind in range(nrep):
        y = popH[ind] + popI[ind]
        y_upper = np.max([np.max(y), y_upper])
        if log and np.min(y) >= 0:
            y = np.log10(y + 1)
        if extinction[ind] and (extcount < nplot):
            extcount += 1
            print(np.min(y))
            handles[0], = plt.step(t[ind], y, color=cols[0], alpha=alpha)
        elif (
            explosion[ind]
            and (expcount < nplot)
            and (np.max(y) > np.log10(plot_thresh))
        ):
            expcount += 1
            print(np.min(y), np.max(y), np.log10(plot_thresh))
            handles[2], = plt.step(t[ind], y, color=cols[2], alpha=alpha, label="y")
        elif carcount < nplot:
            carcount += 1
            print(np.min(y))
            handles[1], = plt.plot(t[ind], y, color=cols[1], alpha=alpha)
    left, right = plt.xlim()
    print(f"Left is : {left}, right is : {right}")
    handles[3], = plt.plot(
        [left, right], np.log10([plot_thresh, plot_thresh]), "k--", alpha=alpha
    )
    legend_flag = 1
    for ind in range(len(handles)):
        if handles[ind] == 0:
            legend_flag = 0
    if legend_flag:
        plt.legend(handles, ["Unaffected", "Carrier", "Response", "Threshold"])
    if log:
        plt.ylim([0, np.log10(sim_stop_thresh)])
        plt.ylabel("$\log_{10}$(Staph.) (CFU)")
    plt.xlim(left, right)


def soap_obj(col: List[str] = ["#4daf4a", "#ff7f00", "#e41a1c"], both: bool = False):
    """Plot soap fit data.

    Plot the fit of RH model, r1 and rmf hypotheses to data on SA kinetics
    after washing with soap.

    Parameters
    ----------
    col
        Colors of the r1 and rmf hypothesis fits.
    both
        Flag on whether to plot immediate inoculation. Inoculation after 24h is
        always plotted.

    Notes
    -----
    Get the relevant data from `get_soap_data`.
    """
    # Get data
    p1 = get_soap_data(1)
    p2 = get_soap_data(2)

    # For 24h dataset
    ptemp = p2
    ptemp["k1"] = ptemp["k1_24h"]
    ptemp["k2"] = ptemp["k2_24h"]
    solrh2 = solve_ivp(
        lambda t, y: rh_growth_model(t, y, ptemp),
        [0, np.max(ptemp["t"])],
        [10 ** ptemp["y0"]],
    )
    ptemp["r1"] = ptemp["r1*"]
    sol2cr2 = solve_ivp(
        lambda t, y: twocomp_model(t, y, ptemp),
        [0, np.max(ptemp["t"])],
        [10 ** ptemp["y0"], 0],
    )
    sol2crmf2 = solve_ivp(
        lambda t, y: twocomp_rmf_model(t, y, ptemp),
        [0, np.max(ptemp["t"])],
        [10 ** ptemp["y0"], 0],
    )

    # Plots
    plt.plot(p2["t"], p2["y"], "ko", label="Data (+24h)")
    if both:
        plt.plot(p1["t"], p1["y"], "kx", label="Data (imm)")
    sse, _ = round(p2["sse_rh"], 3), round(p2["aicc_rh"], 2)
    plt.plot(
        solrh2.t,
        np.log10(solrh2.y.transpose()),
        color="grey",
        linestyle="--",
        label=f"RH (SSE={sse})",
    )
    sse, _ = round(p2["sse_r1"], 3), round(p2["aicc_r1"], 2)
    plt.plot(
        sol2cr2.t,
        np.log10(sol2cr2.y.transpose().sum(axis=1)),
        color=col[0],
        label="$r_1^*$ " + f"(SSE={sse})",
    )
    sse, _ = round(p2["sse_rmf"], 3), round(p2["aicc_rmf"], 2)
    plt.plot(
        sol2crmf2.t,
        np.log10(sol2crmf2.y.transpose().sum(axis=1)),
        color=col[1],
        label="$r_{mf}$" + f" (SSE={sse})",
    )

    if both:
        # Solve integration problems
        # For imm dataset
        ptemp = p1
        ptemp["k1"] = ptemp["k1_imm"]
        ptemp["k2"] = ptemp["k2_imm"]
        solrh1 = solve_ivp(
            lambda t, y: rh_growth_model(t, y, ptemp),
            [0, np.max(ptemp["t"])],
            [10 ** ptemp["y0"]],
        )
        ptemp["r1"] = ptemp["r1*"]
        sol2cr1 = solve_ivp(
            lambda t, y: twocomp_model(t, y, ptemp),
            [0, np.max(ptemp["t"])],
            [10 ** ptemp["y0"], 0],
        )
        sol2crmf1 = solve_ivp(
            lambda t, y: twocomp_rmf_model(t, y, ptemp),
            [0, np.max(ptemp["t"])],
            [10 ** ptemp["y0"], 0],
        )
        plt.plot(solrh1.t, np.log10(solrh1.y.transpose()), color="grey", linestyle="--")
        sse, _ = round(p1["sse_r1"], 2), round(p1["aicc_r1"], 2)
        plt.plot(sol2cr1.t, np.log10(sol2cr1.y.transpose().sum(axis=1)), color=col[0])
        sse, _ = round(p1["sse_rmf"], 2), round(p1["aicc_rmf"], 2)
        plt.plot(
            sol2crmf1.t, np.log10(sol2crmf1.y.transpose().sum(axis=1)), color=col[1]
        )
        plt.ylim([0, 6])

    plt.xlabel("Time (days)")
    plt.ylabel("Staph. density (CFU/cm$^2$)")
    plt.legend()
