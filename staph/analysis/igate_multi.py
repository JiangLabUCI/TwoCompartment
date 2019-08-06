import numpy as np
import matplotlib.pyplot as plt
from .f2 import partition_plot
from .f3 import pop_time


def igate(fname="results/pred_10rep200nstr1hypF_multi.npz", option1=2):
    with np.load(fname, allow_pickle=True) as data:
        pres = data["pres"]
        ps = data["ps"]
        pcar = data["pcar"]
        tref = data["tref"]
        explosion = data["explosion"]
        extinction = data["extinction"]
        pop_flag = data["pop_flag"]
        status = data["status"]
        imax = data["imax"]
    print(f"p_response = {np.mean(explosion)}")
    print(f"p_car = {1 - np.mean(explosion + extinction)}")
    print(f"p_unaffected = {np.mean(extinction)}")
    print(f"imax = {imax}")
    if option1 == 1:
        with np.load(fname, allow_pickle=True) as data:
            popH = data["popH"]
            popI = data["popI"]
            t = data["t"]
            new_exp = data["new_exp"]
            new_ext = data["new_ext"]
        for ind in range(len(t)):
            y = popH[ind] + popI[ind]
            if (np.max(y) > 1e8) or (np.min(y) < 0):
                print(
                    f"Index = {ind}, max = {np.max(y)}, min = {np.min(y)}, max > imax = {np.max(y)>imax}"
                )
                if pop_flag:
                    print(f"Status : {status[ind]}")
                continue
        pop_time(
            t=t,
            popH=popH,
            popI=popI,
            extinction=new_ext,
            explosion=new_exp,
            log=True,
            imax=imax,
        )
        plt.show()
    elif option1 == 2:
        # ax = plt.subplots()
        ax = plt.subplot()
        partition_plot(x=tref, pinf=pres, pcar=pcar, ps=ps, ax=ax, xlab="Time (days)")
        plt.xlabel("Time (days)")
    plt.show()
