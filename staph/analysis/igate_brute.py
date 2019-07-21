import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
from ..utils.dev import transform_x


def igate(filename: str):
    """Investigate output of compute_devs_brute.

    Produce plots and post-process the results from brute-force fitting 
    the 2C model to the Singh data.

    Parameters
    ----------
    filename
        List of file names created by the output of optimization.
    """
    with np.load(filename, allow_pickle=True) as data:
        desol_ind = data["desol_ind"]
        bXlist = data["bXlist"]
        bFlist = data["bFlist"]
        all_devs = data["all_devs"]
        tot_devs = data["tot_devs"]
        all_statuses = data["all_statuses"]
        b2listu = data["b2listu"]
        d1listu = data["d1listu"]
        nb2 = data["nb2"]
        nd1 = data["nd1"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    best_dev = np.min(tot_devs)
    print(tot_devs, best_dev)
    for ind1 in range(nb2):
        for ind2 in range(nd1):
            linear_ind = ind1 * nb2 + ind2
            b2, d1 = transform_x([b2listu[ind1], d1listu[ind2]])
            ax.scatter(b2, d1, tot_devs[linear_ind], color="darkblue")
            if best_dev == tot_devs[linear_ind]:
                ax.scatter(b2, d1, tot_devs[linear_ind], color="darkred")
                title = f"b2 = {b2:.2e}, d1 = {d1:.2f}"
                plt.title(title)
    plt.xlabel("b2")
    plt.ylabel("d1")
    plt.zlabel("deviance")
    plt.show()
