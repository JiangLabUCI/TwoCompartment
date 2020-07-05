from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as intp
import scipy.io as sio
import seaborn as sns
from matplotlib.gridspec import GridSpec

from ..utils.dev import get_bF_bX, get_consts_bX
from .f2 import growth_obj

RANK_1_COLOR = "#fb8072"


def get_parameters_and_objective_values(
    filename: str = "results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat",
    N: int = 100,
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict, int]:
    """Get all parameters and corresponding objective values.

    Read the solutions present in `filename` and return all parameters and
    corresponding objective values. Also returns the top N.

    Parameters
    ----------
    filename
        The file containing the DEMC solutions.
    N
        Top N solutions by objective value will also be returned.

    Returns
    -------
    log10X_orig
        The parameter sets. Warmup is not excluded.
    Flist
        The objective values. Warmup is not excluded.
    log10X_posterior
        The log10 posterior parameters with warmup excluded.
    log10X_topN
        The top N log10 posterior parameters with warmup excluded.
    modno
        Model number. 3 means Imax was predicted in DEMC.
        6 means r3Imax was predicted in DEMC.

    """
    data = sio.loadmat(filename)
    # 3d array (n_generations * n_chains * n_params)
    log10X_orig = data["solset"][0][0][2]
    Flist = data["solset"][0][0][3]
    n_generations = log10X_orig.shape[0]

    # remove warmup
    warmup_cutoff = int(n_generations / 2)
    log10X_fullposterior = log10X_orig[warmup_cutoff + 1 :, :, :]

    # create log10X_posterior
    r1, r2, r3, Imax, modno = get_r_Imax(log10X_fullposterior, filename)
    log10X_posterior = {
        "r1": np.log10(r1),
        "r2": np.log10(r2),
        "r3": np.log10(r3),
        "Imax": np.log10(Imax),
    }

    # create log10X_topN
    Flist_topN, Xlist_topN = get_bF_bX(desol_ind=np.arange(N))
    r1_topN = np.empty([N])
    r2_topN = np.empty([N])
    r3_topN = np.empty([N])
    Imax_topN = np.empty([N])
    for ind in range(N):
        r1_topN[ind], r2_topN[ind], r3_topN[ind], Imax_topN[ind], _ = get_consts_bX(
            Xlist_topN, ind, filename, verbose=0
        )
    log10X_topN = {
        "r1": np.log10(r1_topN),
        "r2": np.log10(r2_topN),
        "r3": np.log10(r3_topN),
        "Imax": np.log10(Imax_topN),
    }
    return log10X_orig, Flist, log10X_posterior, log10X_topN, Flist_topN, modno


def get_r_Imax(
    log10Xlist: np.ndarray, filename: str, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Pre-process Xlist and return kinetic constants.

    From `log10Xlist`, extract the kinetic constants r1, r2, r3 and Imax.

    Parameters
    ----------
    log10Xlist
        The array of solutions.
    filename
        The name of the file from which the indices were extracted.

    Returns
    -------
    r1
        Rate constant with units (/day).
    r2
        Rate constant with units (/day).
    r3
        Rate constant with units (cm^2/(bacteria * day)).
    Imax
        Carrying capacity with units (bacteria/cm^2).
    modno
        Model number. 3 means Imax was predicted in DEMC.
        6 means r3Imax was predicted in DEMC.

    Notes
    -----
    If the filename has `3mod` in it, Imax was predicted as the 4th element.
    If it has `6mod` in it, r3Imax was predicted as the 4th element and Imax
    has to be computed from its value.
    """
    Xlist = np.power(10, log10Xlist)
    r1 = Xlist[:, :, 0].flatten()
    r2 = Xlist[:, :, 1].flatten()
    r3 = Xlist[:, :, 2].flatten()
    modno = int(filename[filename.find("mod") - 1])
    if modno == 3:
        Imax = Xlist[:, :, 3].flatten()
    elif modno == 6:
        Imax = Xlist[:, :, 3].flatten() / r3
        if verbose:
            print("Mean r3 and Imax are : ", np.mean(r3), np.mean(Imax))
    if verbose:
        print("Mean r3 * Imax is : ", np.mean(r3 * Imax))

    return r1, r2, r3, Imax, modno


def plot_parameter(
    posterior_samples: np.ndarray,
    rank_1_samples: pd.DataFrame,
    topN_samples: np.ndarray,
    ax: mpl.axis,
    label: str,
    rank_1_plottype: str = "interp",
    topN_plottype: str = "rug",
    main_color: str = "#70a89f",
    rank_1_color: str = RANK_1_COLOR,
):
    """
    Plot one of the posterior parameters

    Parameters
    ----------
    posterior_samples
        The posterior samples of the parameter.
    rank_1_samples
        The rank 1 samples of the parameter after multi-objective fitting.
    topN_samples
        The top N samples of the parameter ranked by objective function value.
    ax
        The axis object to plot the parameter on.
    label
        The name of the parameter, to be used as x axis label.
    rank_1_plottype
        How to plot the rank 1 solution. Can be "interp", "base" or "vline".
    topN_plottype
        How to plot the top N solutions. Can be "rug" or "range".
    main_color
        Color of the kernel density estimate of the posterior.
    rank_1_color
        Color of the rank 1 samples of the parameter.
    """
    sns.kdeplot(posterior_samples, color=main_color, cut=0, shade=True)
    if rank_1_plottype == "base":
        ax.plot(rank_1_samples, np.zeros(rank_1_samples.shape), "x", color=rank_1_color)
    elif rank_1_plottype == "interp":
        line = plt.gca().get_lines()[0]
        xd = line.get_xdata()
        yd = line.get_ydata()
        f = intp.interp1d(xd, yd)
        new_y = f(rank_1_samples)
        ax.plot(rank_1_samples, new_y, "x", color=rank_1_color)
    elif rank_1_plottype == "vline":
        ymin, ymax = ax.get_ylim()
        ax.vlines(rank_1_samples, ymin, ymax, color=rank_1_color)

    _, ymax = ax.get_ylim()
    if topN_plottype == "range":
        ax.vlines(
            np.min(topN_samples), 0, ymax, color=rank_1_color, linestyles="dashed"
        )
        ax.vlines(
            np.max(topN_samples), 0, ymax, color=rank_1_color, linestyles="dashed"
        )
    elif topN_plottype == "rug":
        y_zero = np.zeros(topN_samples.shape)
        ax.plot(topN_samples, y_zero, "|", color=rank_1_color, alpha=0.5)

    ax.set_xlabel(label)


def panel_label(label: str, ax: mpl.axis, factor: float = 0.15):
    """Label plots.

    Handy function to label plots.

    Parameters
    ----------
    label
        Subfigure label for the plot.
    ax
        Axis to put the label on.
    factor
        Factor by which to left-shift label.
    """
    annotation_args = {
        "va": "bottom",
        "weight": "bold",
        "fontsize": "14",
        "family": "arial",
    }
    x1, x2 = ax.get_xlim()
    _, y2 = ax.get_ylim()
    ax.text(x1 - factor * (x2 - x1), y2, label, annotation_args)


def plot_parameter_posteriors():
    """Plot the posterior parameters.
    """
    _, _, log10X_posterior, log10X_topN, Flist_topN, _ = (
        get_parameters_and_objective_values()
    )
    rank1sol = pd.read_csv("results/rank_1_solutions.csv")
    print(rank1sol)
    log10X_topN_df = pd.DataFrame(log10X_topN)
    topN_df = np.power(10, log10X_topN_df)
    topN_df["Fde"] = Flist_topN
    topN_df["r3*Imax"] = topN_df["r3"] * topN_df["Imax"]
    print(topN_df.loc[:3, :])

    posterior = [
        log10X_posterior["r1"],
        log10X_posterior["r2"],
        log10X_posterior["r3"],
        log10X_posterior["Imax"],
        np.log10(
            np.power(10, log10X_posterior["r1"]) + np.power(10, log10X_posterior["r2"])
        ),
        log10X_posterior["r3"] + log10X_posterior["Imax"],
    ]
    labels = [
        "$\log_{10}(r_1)$",
        "$\log_{10}(r_2)$",
        "$\log_{10}(r_3)$",
        "$\log_{10}{(I_{\mathrm{max}})}$",
        "$\log_{10}(r_1+r_2)$",
        "$\log_{10}(r_3I_{max})$",
    ]
    rank_1 = [
        np.log10(rank1sol.r1),
        np.log10(rank1sol.r2),
        np.log10(rank1sol["r3"]),
        np.log10(rank1sol["r3*Imax"] / rank1sol["r3"]),
        np.log10(rank1sol["r1"] + rank1sol["r2"]),
        np.log10(rank1sol["r3*Imax"]),
    ]
    top_N = [
        log10X_topN["r1"],
        log10X_topN["r2"],
        log10X_topN["r3"],
        log10X_topN["Imax"],
        np.log10(np.power(10, log10X_topN["r1"]) + np.power(10, log10X_topN["r2"])),
        log10X_topN["r3"] + log10X_topN["Imax"],
    ]

    panel_labels = ["B", "C", "D", "E"]

    lef, rig = 0.10, 0.99
    bot, top = 0.11, 0.95
    hs, ws = 0.65, 0.25
    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(4, 2, top=top, bottom=bot, left=lef, right=rig, hspace=hs, wspace=ws)

    ax = fig.add_subplot(gs[0:2, 0])
    col_mo = ["#1b9e77", "#d95f02"]
    growth_obj(topN_df, col_mo, ax, solinds=[0, 99], obj_name="$f_{\mathrm{SSE}}$")
    panel_label("A", ax)

    for ind in range(2):
        ax = fig.add_subplot(gs[ind, 1])
        plot_parameter(
            posterior[ind],
            rank_1[ind],
            top_N[ind],
            ax=ax,
            label=labels[ind],
            rank_1_plottype=None,
        )
        panel_label(panel_labels[ind], ax)

    for ind in range(2):
        ax = fig.add_subplot(gs[2 + ind, 0])
        plot_parameter(
            posterior[2 + ind],
            rank_1[2 + ind],
            top_N[2 + ind],
            ax=ax,
            label=labels[2 + ind],
            rank_1_plottype=None,
        )
        panel_label(panel_labels[ind + 2], ax)

    ax = fig.add_subplot(gs[2:, 1])
    ax.hexbin(log10X_posterior["r1"], log10X_posterior["r2"], gridsize=15, cmap="Greys")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    panel_label("F", ax)

    plt.savefig("results/figs/f_posterior.pdf")
    # plt.show()
