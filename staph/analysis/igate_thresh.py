import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from ..utils.dev import transform_x
from ..utils.pareto_rank import get_pareto_ranks


def igate(filenames: str, option1: int = 1):
    """Investigate output of `thresh_brute_min`.

    Produce plots and post-process the results from brute-force fitting
    the 2C model to the Singh data.

    Parameters
    ----------
    filenames
        List of file names created by the output of optimization.
    option1
        One of 1, 2 or 3.
    
    Notes
    -----
    If option1 = 1, solution landscape is plotted.
    If option1 = 2, create `rank_1_solutions.npz` and `all_solutions.csv`.
    If option1 = 3, plot the final loads for each dose with threshold.
    """
    min_devs = []
    d1s = np.empty([0, 1])
    b2s = np.empty([0, 1])
    desolinds = []
    min_devs = np.empty([0, 1])
    bXs = np.empty([0, 4])
    bFs = np.empty([0, 1])
    threshs = np.empty([0, 1])

    for filename in filenames:
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
            desol_ind = data["desol_ind"]

            # Assign dataframe variables
            desolinds.append(desol_ind)
            ndesol = len(desol_ind)
            bXs = np.vstack([bXs, data["bXlist"]])
            temp = data["bFlist"]
            temp.shape = (ndesol, 1)
            bFs = np.vstack([bFs, temp])
            if t_type is None:
                pass
            elif not t_type:
                t_type = None
            else:
                print("t_type is not none")

            xx = np.zeros([len(b2listu), len(d1listu)])
            yy = np.zeros([len(b2listu), len(d1listu)])
            zz = np.zeros([len(b2listu), len(d1listu)])
            tot_devs = np.zeros(len(all_devs))
            x = np.zeros(len(all_devs))
            y = np.zeros(len(all_devs))
            z = np.zeros(len(all_devs))
            for ind1 in range(len(all_devs)):
                tot_devs[ind1] = np.sum(all_devs[ind1])
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
            b_index_linear = np.argmin(optim_objs)
            b_thresh = optim_thresh[b_index_linear]

            # Assign dataframe variables
            b2s = np.vstack([b2s, b_b2])
            d1s = np.vstack([d1s, b_d1])
            min_devs = np.vstack([min_devs, b_dev])
            threshs = np.vstack([threshs, b_thresh])
            if option1 == 3:
                with np.load(filename, allow_pickle=True) as data:
                    final_loads = data["final_loads"]
                try:
                    assert final_loads.shape != ()
                except AssertionError:
                    print("Final loads was not saved, returning!")
                    return
                this_loads = np.log10(final_loads[b_index, :, :].transpose() + 1)
                x = np.ones(this_loads.shape[0])
                for ind1 in range(6):
                    g = this_loads[:, ind1] < np.log10(b_thresh)
                    r = this_loads[:, ind1] >= np.log10(b_thresh)
                    plt.plot(
                        x[r] * ind1 + np.random.random(x[r].shape) * 0.5,
                        this_loads[r, ind1],
                        "r.",
                    )
                    plt.plot(
                        x[g] * ind1 + np.random.random(x[g].shape) * 0.5,
                        this_loads[g, ind1],
                        "g.",
                    )
                plt.plot([0, 6], np.log10([b_thresh, b_thresh]))
                plt.xlabel("Dose number")
                plt.ylabel("$\log_{10}$(final load)")

    df = np.hstack([bXs, bFs, min_devs, d1s, b2s, threshs])
    colnames = ["r1", "r2", "r3", "r3*Imax", "Fde", "Fst", "d1", "b2", "thresh"]
    df = pd.DataFrame(df, columns=colnames)
    print(df)

    if option1 == 1:
        title_str = f"Bdev = {b_dev:.3f} @ b2 = {b_b2:.3f}, d1 = {b_d1:.3f}"
        print(title_str)
        print(f"Best threshold = {b_thresh:.8e}")
        print(f"Max pop for that threshold = {max_loads[b_index_linear]:.8e}")
        print(f"Simulation stop threshold = {sim_stop_thresh:.8e}")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        inds = tot_devs < 25
        ax.scatter(x[inds], y[inds], z[inds], color="darkblue")
        plt.xlabel("b2")
        plt.ylabel("d1")
        ax.set_zlabel("deviance")
        ax.plot_wireframe(xx, yy, zz)
        plt.title(title_str)
    elif option1 == 2:
        # Save rank 1 solutions in a numpy file
        Fvals = np.vstack([df.Fde, df.Fst]).transpose()
        df["ranks"] = get_pareto_ranks(Fvals)
        output_filename = "results/all_solutions.csv"
        df.to_csv(output_filename)
        df = df[df.ranks == 1]
        df["desol_inds"] = list(df.axes[0])
        print("Rank 1 dataframe is : ")
        print(df)
        output_filename = "results/rank_1_solutions.csv"
        df.to_csv(output_filename)
    plt.show()
