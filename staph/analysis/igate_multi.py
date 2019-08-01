import numpy as np
import matplotlib.pyplot as plt
from .f2 import partition_plot


def igate(fname="results/pred_10rep200nstr1hypF_multi.npz", option1=2):
    part_cols = ["#4daf4a", "#ff7f00", "#e41a1c"]
    with np.load(fname, allow_pickle=True) as data:
        pres = data["pres"]
        ps = data["ps"]
        pcar = data["pcar"]
        nstep = data["nstep"]
        nrep = data["nrep"]
        tref = data["tref"]
        explosion = data["explosion"]
        extinction = data["extinction"]
        pop_flag = data["pop_flag"]
        status = data["status"]
    print(f"p_response = {np.mean(explosion)}")
    print(f"p_car = {1 - np.mean(explosion + extinction)}")
    print(f"p_unaffected = {np.mean(extinction)}")
    if option1 == 1:
        if pop_flag:
            with np.load(fname, allow_pickle=True) as data:
                popH = data["popH"]
                popI = data["popI"]
                t = data["t"]
            for ind in range(len(popH)):
                y = popH[ind] + popI[ind]
                if (np.max(y) > 1e8) or (np.min(y) < 0):
                    print(ind, status[ind], np.max(y), np.min(y))
                    continue
                if extinction[ind]:
                    plt.plot(t[ind], y, color="green", alpha=0.3)
                elif explosion[ind]:
                    plt.plot(t[ind], y, color="darkred", alpha=0.3)
                else:
                    plt.plot(t[ind], y, color="darkblue", alpha=0.3)
            plt.show()
    elif option1 == 2:
        # ax = plt.subplots()
        ax = plt.subplot()
        partition_plot(x=tref, pinf=pres, pcar=pcar, ps=ps, ax=ax, xlab="Time (days)")
        plt.xlabel("Time (days)")
    plt.show()
