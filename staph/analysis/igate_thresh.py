import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..utils.dev import transform_x


def igate(filename: str, option1: int = 1):
    """Investigate output of `thresh_brute_min`.

    Produce plots and post-process the results from brute-force fitting
    the 2C model to the Singh data.

    Parameters
    ----------
    filenames
        List of file names created by the output of optimization.
    option1
        1 plots landscape.
    """
    with np.load(filename, allow_pickle=True) as data:
        b2listu = data["b2listu"]
        d1listu = data["d1listu"]
        all_devs = data["all_devs"]
        nb2 = data["nb2"]
        nd1 = data["nd1"]
        t_type = data["t_type"]
        optim_objs = data["optim_objs"]
        optim_thresh = data["optim_thresh"]
        max_loads = data["max_loads"]
        sim_stop_thresh = data["sim_stop_thresh"]
        print("ttype is : ", t_type)
        if t_type is None:
            print("ttype is none")
        elif not t_type:
            print("ttype is numpy none", type(t_type))
            t_type = None
        else:
            print("t_type is not none")

    xx = np.zeros([len(b2listu), len(d1listu)])
    yy = np.zeros([len(b2listu), len(d1listu)])
    zz = np.zeros([len(b2listu), len(d1listu)])
    # best_dev = np.min(all_devs)
    tot_devs = np.zeros(len(all_devs))
    x = np.zeros(len(all_devs))
    y = np.zeros(len(all_devs))
    z = np.zeros(len(all_devs))
    for ind in range(len(all_devs)):
        tot_devs[ind] = np.sum(all_devs[ind])
    for ind1 in range(nb2):
        for ind2 in range(nd1):
            linear_ind = ind1 * nd1 + ind2
            b2, d1 = transform_x([b2listu[ind1], d1listu[ind2]], t_type=t_type)
            xx[ind1, ind2] = b2
            yy[ind1, ind2] = d1
            zz[ind1, ind2] = np.sum(all_devs[linear_ind, :])
            x[linear_ind] = b2
            y[linear_ind] = d1
            z[linear_ind] = np.sum(all_devs[linear_ind, :])
    b_index = np.argmin(z)
    b_b2 = x[b_index]
    b_d1 = y[b_index]
    b_dev = z[b_index]
    title_str = f"Bdev = {b_dev:.3f} @ b2 = {b_b2:.3f}, d1 = {b_d1:.3f}"
    print(title_str)
    b_index_linear = np.argmin(optim_objs)
    print(f"Best threshold = {optim_thresh[b_index_linear]:.8e}")
    print(f"Max pop for that threshold = {max_loads[b_index_linear]:.8e}")
    print(f"Simulation stop threshold = {sim_stop_thresh:.8e}")

    if option1 == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        inds = tot_devs < 25
        ax.scatter(x[inds], y[inds], z[inds], color="darkblue")
        plt.xlabel("b2")
        plt.ylabel("d1")
        ax.set_zlabel("deviance")
        ax.plot_wireframe(xx, yy, zz)
        plt.title(title_str)
    plt.show()
