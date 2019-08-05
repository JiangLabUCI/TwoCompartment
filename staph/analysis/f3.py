import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import List
from ..utils.data import get_soap_data
from ..utils.det_models import twocomp_rmf_model
from .f2 import rh_growth_model, twocomp_model, partition_plot
import matplotlib as mpl


def label(xlab: str = "", ylab: str = "", label: str = ""):
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
    plt.text(x1 - 0.15 * (x2 - x1), y2, label, annotation_args)


def f3_v2():
    """Assemble figure 3.

    Assemble figure 3 using gridspec for improved layout.
    """
    fnames = [
        "results/predsr1s24h2523823dl5r1_1000rep.npz",
        "results/predsrmf24h2523823dl5r1_1000rep.npz",
    ]
    col_mo = ["#984ea3", "#ff7f00"]
    cols = ["#70a89f", "#fdb462", "#fb8072"]  # colorbrewer 1

    mpl.rcParams["font.family"] = "arial"
    x = np.random.rand(100)

    fig = plt.figure(1, figsize=(9, 8))
    gs = fig.add_gridspec(4, 2, hspace=0.43, bottom=0.05, top=0.95, left=0.08, right=0.98)
    
    ax = fig.add_subplot(gs[0:2, 0])
    soap_obj(col_mo)
    label(xlab=ax.get_xlabel(), ylab=ax.get_ylabel(), label="A")

    labs = ["B", "C"]
    for ind1, filename in enumerate(fnames):
        with np.load(filename) as data:
            dose = data["doselist"]
            pinf = data["pinf"]
            pcar = data["pcar"]
            ps = data["ps"]
        ax = fig.add_subplot(gs[ind1, 1])
        if ind1 == 0:
            partition_plot(dose, pinf[0,], pcar[0,], ps[0,], ax, cols=cols, log=True)
            label(ylab=ax.get_ylabel(), label=labs[ind1])
        elif ind1 == 1:
            partition_plot(dose, pinf[0,], pcar[0,], ps[0,], ax, cols=cols, log=True)
            label(xlab=ax.get_xlabel(), ylab=ax.get_ylabel(), label=labs[ind1])
            ax.legend_.remove()

    fnames = [
        "results/pred_1000rep200000nstr1hypF6_multi.npz",
        "results/pred_1000rep200000nstrmfhypF6_multi.npz",
    ]
    labs1 = ["D", "E"]
    labs2 = ["F", "G"]
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

        # Plot population vs. time
        ax = fig.add_subplot(gs[2+ind1, 0])
        pop_time(t, popH, popI, new_ext, new_exp, log=True, alpha=0.5, nplot=2)
        if ind1 == 0:
            label(ylab=ax.get_ylabel(), label=labs1[ind1])
        elif ind1 == 1:
            label(xlab="Time (days)", ylab="Log10(Pop)", label=labs1[ind1])

        # Plot probability vs. time
        ax = fig.add_subplot(gs[2+ind1, 1])
        if ind1 == 0:
            partition_plot(tref, pres, pcar, ps, ax, cols=cols)
            label(ylab=ax.get_ylabel(), label=labs2[ind1])
        elif ind1 == 1:
            partition_plot(tref, pres, pcar, ps, ax, cols=cols)
            label(xlab="Time (days)", ylab=ax.get_ylabel(), label=labs2[ind1])
        ax.legend_.remove()

    plt.savefig("results/figs/layout.png")
    plt.show()


def f3_24h():
    """Assemble figure 3.

    Assemble figure 3 by calling appropriate functions.
    """
    fnames = [
        "results/predsr1s24h2523823dl5r1_1000rep.npz",
        "results/predsrmf24h2523823dl5r1_1000rep.npz",
    ]
    col_mo = ["#984ea3", "#ff7f00"]
    cols = ["#4daf4a", "#ff7f00", "#e41a1c"]
    cols = ["#70a89f", "#fdb462", "#fb8072"]  # colorbrewer 1

    mpl.rcParams["font.family"] = "arial"
    fig = plt.figure(1, figsize=(9, 8))

    ax = plt.subplot(4, 2, (1, 3))
    soap_obj(col_mo)
    label(xlab=ax.get_xlabel(), ylab=ax.get_ylabel(), label="A")

    labs = ["B", "C"]
    for ind1, filename in enumerate(fnames):
        with np.load(filename) as data:
            dose = data["doselist"]
            pinf = data["pinf"]
            pcar = data["pcar"]
            ps = data["ps"]
            print(pinf.shape)
            # r1 = data["r1"]
            # r2 = data["r2"]
        ax = plt.subplot(4, 2, (ind1 + 1) * 2)
        if ind1 == 0:
            partition_plot(dose, pinf[0,], pcar[0,], ps[0,], ax, cols=cols, log=True)
        elif ind1 == 1:
            partition_plot(dose, pinf[0,], pcar[0,], ps[0,], ax, cols=cols, log=True)
            ax.legend_.remove()
        label(xlab=ax.get_xlabel(), ylab=ax.get_ylabel(), label=labs[ind1])

    fnames = [
        "results/pred_1000rep200000nstr1hypF6_multi.npz",
        "results/pred_1000rep200000nstrmfhypF6_multi.npz",
    ]
    labs1 = ["D", "E"]
    labs2 = ["F", "G"]
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

        # Plot population vs. time
        ax = plt.subplot(4, 2, (ind1 + 2) * 2 + 1)
        pop_time(t, popH, popI, new_ext, new_exp, log=True, alpha=0.5, nplot=2)
        label(xlab="Time (days)", ylab="Log10(Pop)", label=labs1[ind1])

        # Plot probability vs. time
        ax = plt.subplot(4, 2, (ind1 + 3) * 2)
        if ind1 == 0:
            partition_plot(tref, pres, pcar, ps, ax, cols=cols)
        elif ind1 == 1:
            partition_plot(tref, pres, pcar, ps, ax, cols=cols)
        ax.legend_.remove()
        label(xlab="Time (days)", ylab=ax.get_ylabel(), label=labs2[ind1])

    fig.tight_layout()
    plt.savefig("results/figs/f3.png")
    plt.show()


def pop_time(
    t: List[np.ndarray],
    popH: List[np.ndarray],
    popI: List[np.ndarray],
    extinction: List[int],
    explosion: List[int],
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
    for ind in range(nrep):
        y = popH[ind] + popI[ind]
        if log and np.min(y) >= 0:
            y = np.log10(y + 1)
        if extinction[ind] and (extcount < nplot):
            extcount += 1
            print(np.min(y))
            plt.step(t[ind], y, color=cols[0], alpha=alpha)
        elif explosion[ind] and (expcount < nplot):
            expcount += 1
            print(np.min(y))
            plt.step(t[ind], y, color=cols[2], alpha=alpha)
        elif carcount < nplot:
            carcount += 1
            print(np.min(y))
            plt.plot(t[ind], y, color=cols[1], alpha=alpha)
    if log:
        plt.ylim([0, 7])


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
    sse, aicc = round(p2["sse_rh"], 3), round(p2["aicc_rh"], 2)
    plt.plot(
        solrh2.t,
        np.log10(solrh2.y.transpose()),
        color="grey",
        linestyle="--",
        label=f"RH (SSE={sse}, AICc={aicc})",
    )
    sse, aicc = round(p2["sse_r1"], 3), round(p2["aicc_r1"], 2)
    plt.plot(
        sol2cr2.t,
        np.log10(sol2cr2.y.transpose().sum(axis=1)),
        color=col[0],
        label="$r1$ " + f"(SSE={sse}, AICc={aicc})",
    )
    sse, aicc = round(p2["sse_rmf"], 3), round(p2["aicc_rmf"], 2)
    plt.plot(
        sol2crmf2.t,
        np.log10(sol2crmf2.y.transpose().sum(axis=1)),
        color=col[1],
        label="$r_{mf}$" + f"(SSE={sse}, AICc={aicc})",
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
        sse, aicc = round(p1["sse_r1"], 2), round(p1["aicc_r1"], 2)
        plt.plot(sol2cr1.t, np.log10(sol2cr1.y.transpose().sum(axis=1)), color=col[0])
        sse, aicc = round(p1["sse_rmf"], 2), round(p1["aicc_rmf"], 2)
        plt.plot(
            sol2crmf1.t, np.log10(sol2crmf1.y.transpose().sum(axis=1)), color=col[1]
        )
        plt.ylim([0, 6])

    plt.xlabel("Time (days)")
    plt.ylabel("Staph. density (CFU/cm$^2$)")
    plt.legend()
