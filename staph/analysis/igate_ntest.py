import numpy as np
import matplotlib.pyplot as plt
from typing import List
from ..utils.data import get_singh_data
from ..utils.rh_data import get_rh_fit_data


def igate(filenames=List[str], option1: int = 1):
    """Investigate output files.

    Parameters
    ----------
    filenames
        List of file names containing the outputs of interest.
    option1
        The kind of visualization. Either of 1 or 2.

    Notes
    -----
    option1
    1 : print all best deviances and rate constants.
    2 : Plot deviance, rate constants vs iterations. Also plot each vs. the
    other.
    3 : Plot best fit dose-response.
    4 : Return best fit's deviance, b2, d1 and pinf.
    """
    for ind1, filename in enumerate(filenames):
        with open("results/ops/" + filename) as f:
            d = f.read()
        d = d.split("\n")
        b2 = []
        d1 = []
        dev = []
        for ind1, line in enumerate(d):
            if line.startswith("Rates are :"):
                if line.endswith("]"):
                    this_line = line.split()
                else:
                    this_line = (d[ind1] + d[ind1 + 1]).replace("\n", "").split()
                b2.append(float(this_line[6]))
                d1.append(float(this_line[7]))
            if line.startswith("Which gives"):
                this_line = line.split()
                dev.append(float(this_line[6]))
            if line.startswith("Initial_guess is :"):
                this_line = (
                    line.replace(",", "").replace("(", "").replace(")", "").split()
                )
                init_guess = float(this_line[3]), float(this_line[4])
        if option1 == 1:
            best_ind = np.argmin(dev)
            print(f"{dev[best_ind]:.2f}, {b2[best_ind]:.2f}, {d1[best_ind]:.2f}")
        elif option1 == 2:
            plt.figure()
            plt.subplot(231)
            plt.title(
                f"Best dev = {min(dev):.2f} (ite = {np.argmin(dev)} of {len(dev)})"
            )
            plt.plot(dev)
            plt.plot(dev, ".")
            plt.subplot(232)
            # plt.plot(b2)
            plt.plot(np.log10(b2), ".")
            plt.subplot(233)
            # plt.plot(d1)
            plt.plot(np.log10(d1), ".")
            plt.subplot(234)
            plt.plot(b2, dev, "r.")
            plt.subplot(235)
            plt.plot(d1, dev, "r.")
            plt.subplot(236)
            plt.plot(b2, d1, "r.")
        elif option1 == 3:
            sdata = get_singh_data()
            qstr = f"Which gives best dev of : {np.min(dev):.4f}"
            for ind1, line in enumerate(d):
                if line.startswith("Best F values :"):
                    Fde = line
                if line.startswith(qstr):
                    roi = d[ind1 + 2]
                    break
            roi = roi.replace("[", "").replace("]", "").split()
            roi = roi[3:]
            pinf = [float(this_roi) for this_roi in roi]
            Fde = Fde[:-1].replace("[", "").split()
            Fde = float(Fde[4])
            rh = get_rh_fit_data()
            lab_data = "Singh data"
            lab_2c = f"2C (dev = {np.min(dev):.2f})"
            lab_rh = f"RH (dev = {rh[1]:.2f})"
            title = f"SSE={Fde:.2f} (RH SSE={rh[0]:.2f})"
            plt.figure()
            plt.plot(
                np.log10(sdata[0]), np.array(sdata[1]) / sdata[2], "ko", label=lab_data
            )
            plt.plot(np.log10(sdata[0]), 1 - np.exp(-rh[2] / rh[3]), "rx", label=lab_rh)
            plt.plot(np.log10(sdata[0]), pinf, "gs", label=lab_2c)
            plt.title(title)
            plt.xlabel("log10(dose)")
            plt.ylabel("p(respons)")
            plt.legend()
        elif option1 == 4:
            min_ind = np.argmin(dev)
            qstr = f"Which gives best dev of : {np.min(dev):.4f}"
            for ind1, line in enumerate(d):
                if line.startswith("Best F values :"):
                    Fde = line
                if line.startswith(qstr):
                    roi = d[ind1 + 2]
                    break
            roi = roi.replace("[", "").replace("]", "").split()
            roi = roi[3:]
            pinf = [float(this_roi) for this_roi in roi]
            return dev[min_ind], b2[min_ind], d1[min_ind], pinf
    plt.show()
