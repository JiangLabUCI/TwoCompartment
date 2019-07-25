import numpy as np
import matplotlib.pyplot as plt
from typing import List


def igate(filenames=List[str]):
    """Investigate output files from Powell.
    """
    for ind1, filename in enumerate(filenames):
        with open(filename) as f:
            d = f.read()
        d = d.split("\n")
        b2 = []
        d1 = []
        dev = []
        for line in d:
            if line.startswith("Rates are :"):
                this_line = line.split()
                b2.append(float(this_line[6]))
                d1.append(float(this_line[7]))
            if line.startswith("Objective is :"):
                this_line = line.split()
                dev.append(float(this_line[3]))
            if line.startswith("Initial_guess is :"):
                this_line = (
                    line.replace(",", "").replace("(", "").replace(")", "").split()
                )
                init_guess = float(this_line[3]), float(this_line[4])
        plt.subplots()
        plt.subplot(231)
        plt.title(f"Best dev = {min(dev):.2f}")
        plt.plot(dev)
        plt.plot(dev, ".")
        plt.subplot(232)
        # plt.plot(b2)
        plt.plot(np.log10(b2), ".")
        plt.subplot(233)
        # plt.plot(d1)
        plt.plot(np.log10(d1), ".")
        plt.subplot(234)
        # plt.axvline(x=init_guess[0], color="k")
        plt.plot(b2, dev, "r.")
        plt.subplot(235)
        plt.axvline(x=init_guess[1], color="k")
        plt.plot(d1, dev, "r.")
        plt.subplot(236)
        plt.plot(b2, d1, "r.")
        plt.plot(init_guess[0], init_guess[1], "ko")

    plt.show()
