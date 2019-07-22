import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
from ..utils.dev import transform_x
from collections import Counter


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

    xx = np.zeros([len(b2listu), len(d1listu)])
    yy = np.zeros([len(b2listu), len(d1listu)])
    zz = np.zeros([len(b2listu), len(d1listu)])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    best_dev = np.min(tot_devs)
    print(f"Best parameters (DEMC) = {bXlist}")
    print(f"Best objective (DEMC) = {bFlist}")
    print(f"Best deviance = {best_dev}")
    for ind1 in range(nb2):
        for ind2 in range(nd1):
            linear_ind = ind1 * nb2 + ind2
            b2, d1 = transform_x([b2listu[ind1], d1listu[ind2]])
            xx[ind1, ind2] = b2
            yy[ind1, ind2] = d1
            zz[ind1, ind2] = tot_devs[linear_ind]
            ax.scatter(b2, d1, tot_devs[linear_ind], color="darkblue")
            if best_dev == tot_devs[linear_ind]:
                ax.scatter(b2, d1, tot_devs[linear_ind], color="darkred")
                title = f"b2 = {b2:.2e}, d1 = {d1:.2f}"
                print("Paramters : " + title)
                for ind3 in range(len(all_statuses[linear_ind])):
                    status = all_statuses[linear_ind][ind3]
                    p_inf = np.mean((status == 3) + (status == 4) + (status == 5))
                    print(
                        f"Deviance = {all_devs[linear_ind][ind3]:.2f} , pinf = {p_inf:.3f}, {Counter(all_statuses[linear_ind][ind3])}"
                    )
                plt.title(title)
    plt.xlabel("b2")
    plt.ylabel("d1")
    ax.plot_wireframe(xx, yy, zz)
    plt.show()
