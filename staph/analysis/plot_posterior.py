import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from typing import Tuple


def get_parameters_and_objective_values(
    filename: str = "results/6021324_DEMC_40000g_16p6mod1ds0se_staph1o6.mat",
) -> Tuple[np.ndarray, np.ndarray]:
    """Get all parameters and corresponding objective values.

    Read the solutions present in `filename` and return all parameters and
    corresponding objective values.

    Parameters
    ----------
    filename
        The file containing the DEMC solutions.

    Returns
    -------
    log10Xlist
        The parameter sets. Warmup is not excluded.
    Flist
        The objective values. Warmup is not excluded.
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

    """
    data = sio.loadmat(filename)
    log10Xlist = data["solset"][0][0][
        2
    ]  # 3d array (n_generations * n_chains * n_params)
    Flist = data["solset"][0][0][3]
    n_generations = log10Xlist.shape[0]
    warmup_cutoff = int(n_generations / 2)
    log10X_posterior = log10Xlist[warmup_cutoff + 1 :, :, :]
    r1, r2, r3, Imax, modno = get_r_Imax(log10X_posterior, filename)
    return log10Xlist, Flist, r1, r2, r3, Imax, modno


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


def plot_parameter(parameter, label, col="#70a89f"):
    plt.hist(parameter, color=col)
    plt.xlabel(label)
    print(label)


def plot_parameter_posteriors():
    log10Xlist, Flist, r1, r2, r3, Imax, modno = get_parameters_and_objective_values()

    param1 = np.log10(r1)
    param2 = np.log10(r2)
    param3 = np.log10(r3)
    param4 = np.log10(Imax)
    param5 = np.log10(r1 + r2)
    param6 = np.log10(r3 * Imax)
    labels = [
        "none",
        "$log_{10}(r_1)$",
        "$log_{10}(r_2)$",
        "$log_{10}(r_3)$",
        "$log_{10}{(I_{max})}$",
        "$log_{10}(r_1+r_2)$",
    ]
    print(log10Xlist.shape, Flist.shape)

    # plot parameters

    # plt.plot(Flist)
    # plt.xlabel("Generations")
    # plt.ylabel("Objective function value")

    plt.figure(figsize=(9, 6))
    plt.subplot(231)
    plot_parameter(param1, labels[1])
    plt.subplot(232)
    plot_parameter(param2, labels[2])
    plt.subplot(233)
    plot_parameter(param3, labels[3])
    plt.subplot(234)
    plot_parameter(param4, labels[4])
    plt.subplot(235)
    plot_parameter(param5, labels[5])
    plt.subplot(236)
    plot_parameter(param6, "$log_{10}(r_3I_{max})$")
    # plt.hist2d(param4, param5, cmap=plt.get_cmap("Greys"), bins=20)
    # plt.xlabel("")

    plt.tight_layout()
    plt.savefig("results/figs/f_posterior.pdf")
    # plt.show()
