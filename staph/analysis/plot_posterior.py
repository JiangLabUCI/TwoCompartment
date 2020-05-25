import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate as intp
from typing import Tuple, Dict
import seaborn as sns
import pandas as pd
from ..utils.dev import get_bF_bX, get_consts_bX


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
    return log10X_orig, Flist, log10X_posterior, log10X_topN, modno


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
    label: str,
    rank_1_plottype: str = "interp",
    main_color: str = "#70a89f",
    rank_1_color: str = "#fb8072",
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
    label
        The name of the parameter, to be used as x axis label.
    rank_1_plottype
        How to plot the rank 1 solution. Can be "interp", "base" or "vline".
    main_color
        Color of the kernel density estimate of the posterior.
    rank_1_color
        Color of the rank 1 samples of the parameter.
    """
    sns.kdeplot(posterior_samples, color=main_color, cut=0, shade=True)
    if rank_1_plottype == "base":
        plt.plot(
            rank_1_samples, np.zeros(rank_1_samples.shape), "x", color=rank_1_color
        )
    elif rank_1_plottype == "interp":
        line = plt.gca().get_lines()[0]
        xd = line.get_xdata()
        yd = line.get_ydata()
        f = intp.interp1d(xd, yd)
        new_y = f(rank_1_samples)
        plt.plot(rank_1_samples, new_y, "x", color=rank_1_color)
    elif rank_1_plottype == "vline":
        ymin, ymax = plt.ylim()
        plt.vlines(rank_1_samples, ymin, ymax, color=rank_1_color)
    
    _, ymax = plt.ylim()
    plt.vlines(np.min(topN_samples), 0, ymax, color=rank_1_color, linestyles="dashed")
    plt.vlines(np.max(topN_samples), 0, ymax, color=rank_1_color, linestyles="dashed")

    plt.xlabel(label)


def plot_parameter_posteriors():
    log10Xlist, Flist, log10X_posterior, log10X_topN, modno = (
        get_parameters_and_objective_values()
    )
    df = pd.read_csv("results/rank_1_solutions.csv")
    print(df)

    posterior = [
        log10X_posterior["r1"],
        log10X_posterior["r2"],
        log10X_posterior["r3"],
        log10X_posterior["Imax"],
        np.log10(np.power(10, log10X_posterior["r1"]) + np.power(10, log10X_posterior["r2"])),
        log10X_posterior["r3"] + log10X_posterior["Imax"],
    ]
    labels = [
        "$log_{10}(r_1)$",
        "$log_{10}(r_2)$",
        "$log_{10}(r_3)$",
        "$log_{10}{(I_{max})}$",
        "$log_{10}(r_1+r_2)$",
        "$log_{10}(r_3I_{max})$",
    ]
    rank_1 = [
        np.log10(df.r1),
        np.log10(df.r2),
        np.log10(df["r3"]),
        np.log10(df["r3*Imax"] / df["r3"]),
        np.log10(df["r1"] + df["r2"]),
        np.log10(df["r3*Imax"]),
    ]
    top_N = [
        log10X_topN["r1"],
        log10X_topN["r2"],
        log10X_topN["r3"],
        log10X_topN["Imax"],
        np.log10(np.power(10, log10X_topN["r1"]) + np.power(10, log10X_topN["r2"])),
        log10X_topN["r3"] + log10X_topN["Imax"],
    ]

    # plot parameters
    plt.figure(figsize=(9, 6))
    for ind in range(6):
        plt.subplot(231 + ind)
        plot_parameter(posterior[ind], rank_1[ind], top_N[ind], labels[ind], rank_1_plottype=None)
        print(f"Plotted  {labels[ind]}")

    plt.tight_layout()
    plt.savefig("results/figs/f_posterior.pdf")
    # plt.show()
