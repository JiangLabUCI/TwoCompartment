import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
from ..utils.pareto_rank import get_pareto_ranks
from ..utils.data import get_singh_data
from ..utils.rh_data import get_rh_fit_data


matplotlib.rcParams.update({"font.size": 15})


def igate(
    filenames: List[str] = "6021324_1000rep0se70ite_Powell_0to9b2d1_1o1_cpu.npz",
    choice: int = 1,
    problem_type: int = 1,
):
    """Investigate output of compute_devs_min.

    Produce plots and post-process the results from fitting the 2C model to
    the Singh data.

    Parameters
    ----------
    filenames
        List of file names created by the output of optimization.
    choice
        Integer representing the plot/post-processing step (see Notes).
    problem_type
        If 1, optimize for b2 and d1. If 2, optimize only for d1 with b2 = 0.

    Notes
    -----

    choice = 1
    - Make the pareto plot with all solutions.
    - Solutions of a given rank are joined by a line.
    - Alternating ranks are colored alternately.

    choice = 2
    - Plot all solution paramter values.
    - Highlight better performers in red.
    - Also plot select combinations of objectives and paramters.
    
    choice = 3
    - Compute pareto ranks of all solutions.
    - Save all solutions with rank in a file.
    - Save rank 1 solutions in a file.
    """
    # h0, norig, ntot, tiny, A, H0 = get_singh_data()
    rh_best_sse, rh_best_dev, _, _ = get_rh_fit_data()

    # Preprocess data
    min_devs = []
    d1s = np.empty([0, 1])
    b2s = np.empty([0, 1])
    de_sol_inds = []
    min_devs = np.empty([0, 1])
    bXs = np.empty([0, 4])
    bFs = np.empty([0, 1])
    for fname in filenames:
        with np.load(fname, allow_pickle=True) as data:
            desol_ind = data["desol_ind"]
            ndesol = len(desol_ind)
            bXs = np.vstack([bXs, data["bXlist"]])
            temp = data["bFlist"]
            temp.shape = (ndesol, 1)
            bFs = np.vstack([bFs, temp])
            optim_objs = data["optim_objs"]
            # Extflag = data["ExtFlag"]
            # niter = data["niter"]
            # method = data["method"]
            de_sol_inds.append(desol_ind)
            print("DE solution indices", desol_ind)
            for ind in range(ndesol):
                min_devs = np.vstack([min_devs, optim_objs[ind].fun])

                if problem_type == 1:
                    b2s = np.vstack([b2s, optim_objs[ind].x[0]])
                    d1s = np.vstack([d1s, optim_objs[ind].x[1]])
                    Fst_cutoff = 25
                elif problem_type == 2:
                    d1s = np.vstack([d1s, optim_objs[ind].x])
                    Fst_cutoff = 18000

    if problem_type == 1:
        z = np.hstack([bXs, bFs, min_devs, d1s, b2s])
        colnames = ["r1", "r2", "r3", "r3*Imax", "Fde", "Fst", "d1", "b2"]
    elif problem_type == 2:
        z = np.hstack([bXs, bFs, min_devs, d1s])
        colnames = ["r1", "r2", "r3", "r3*Imax", "Fde", "Fst", "d1"]
    df = pd.DataFrame(z, columns=colnames)
    print("Dropping following row(s) due to negative values : ")
    print(df[df.d1 < 0])
    df = df[df.d1 > 0]
    df = df[df.Fst < Fst_cutoff]
    if choice == 1:
        d1b2_vals = np.vstack([df.Fde, df.Fst]).transpose()
        df["ranks"] = get_pareto_ranks(d1b2_vals)
        ms = 10
        c1 = [237 / 255, 125 / 255, 49 / 255]  # Orange
        c2 = [27 / 255, 158 / 255, 119 / 255, 0.7]  # green

        fig = plt.figure(figsize=[6, 5])
        ax = fig.add_subplot(111)
        ctr = 0
        for ind in range(1, int(max(df.ranks))):
            df2 = df[df.ranks == ind].sort_values("Fde")
            xvals, yvals = list(df2.Fde), list(df2.Fst)
            if not (xvals == []):
                if np.mod(ctr, 2) == 0:
                    plt.plot(xvals, yvals, color="xkcd:salmon")
                else:
                    plt.plot(xvals, yvals, color="xkcd:bright blue")
                ctr += 1
        h1, = plt.plot(
            bFs,
            min_devs,
            ".",
            label="2C suboptimal fits",
            markersize=ms,
            color=c1,
            markeredgecolor=[0, 0, 0],
        )
        h2, = plt.plot(
            [rh_best_sse, rh_best_sse], [0, 100], "--k", label="Rose & Haas best fit"
        )
        h3, = plt.plot(
            bFs[min_devs < rh_best_dev],
            min_devs[min_devs < rh_best_dev],
            ".",
            label="2C pareto-optimal fits",
            markersize=ms,
            color=c2,
            markeredgecolor=[0, 0, 0],
        )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.plot([min(bFs) - 0.1, 1.05], [rh_best_dev, rh_best_dev], "--k")
        plt.xlabel("Growth objective function")
        plt.ylabel("Dose-response objective function")
        plt.xlim([min(bFs), 0.8])
        plt.ylim([min(min_devs) - 0.5, Fst_cutoff])
        plt.legend(handles=[h1, h2, h3], loc="upper right")

        plt.savefig("results/imgs/pareto_plot.png")
    elif choice == 2:
        df["Imax"] = df["r3*Imax"] / df["r3"]
        colnames.append("Imax")
        df2 = np.log10(df)
        df2 = (df2 - df2.min()) / (df2.max() - df2.min())
        try:
            plt.plot(
                np.transpose(df2[df.Fst < rh_best_dev].values), "xkcd:red", alpha=0.7
            )
        except:
            print("Possibly no solution has better deviance than RH")
        plt.plot(np.transpose(df2[df.Fst > rh_best_dev].values), "xkcd:grey", alpha=0.7)
        plt.xticks(range(len(df2.columns)), labels=colnames)
        plt.figure(tight_layout=True)
        plt.subplot(231)
        plt.plot(df.d1, df.Fst, "ro")
        plt.ylabel("Fst")
        plt.subplot(232)
        plt.plot(df.b2, df.Fst, "ro")
        plt.subplot(233)
        plt.plot(df.Fde, df.Fst, "ro")
        plt.xlabel("Fde")
        plt.subplot(234)
        plt.plot(df.d1, df.Fde, "ro")
        plt.xlabel("d1")
        plt.ylabel("Fde")
        plt.subplot(235)
        plt.plot(df.b2, df.Fde, "ro")
        plt.xlabel("b2")
        plt.subplot(236)
        plt.plot(df.Fst)
        plt.ylabel("Fst")
    elif choice == 3:
        # Save rank 1 solutions in a numpy file
        Fvals = np.vstack([df.Fde, df.Fst]).transpose()
        df["ranks"] = get_pareto_ranks(Fvals)
        output_filename = "results/all_solutions.npz"
        with open(output_filename, "wb") as f:
            np.savez(f, df=df)
        df = df[df.ranks == 1]
        df["desol_inds"] = list(df.axes[0])
        print("Rank 1 dataframe is : ")
        print(df)
        output_filename = "results/rank_1_solutions.csv"
        df.to_csv(output_filename)
    plt.show()
