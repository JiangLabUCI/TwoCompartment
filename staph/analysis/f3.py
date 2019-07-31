import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import List
from ..utils.data import get_soap_data
from ..utils.det_models import twocomp_rmf_model
from .f2 import rh_growth_model, twocomp_model, partition_plot


def f3_24h():
    """Assemble figure 3.

    Assemble figure 3 by calling appropriate functions.
    """
    fnames = [
        "results/predsr1s24h2523823dl5r1_1000rep.npz",
        "results/predsrmf24h2523823dl5r1_1000rep.npz",
    ]
    col_mo = ["#984ea3", "#ff7f00"]
    annotation_args = {"va": "bottom", "weight": "bold", "fontsize": "12"}
    plt.subplots(nrows=2, ncols=2, sharex="all", figsize=(9, 8))
    cols = ["#4daf4a", "#ff7f00", "#e41a1c"]
    plt.subplot(221)
    soap_obj(col_mo)
    x1, x2 = plt.xlim()
    _, y2 = plt.ylim()
    plt.text(x1 - 0.15 * (x2 - x1), y2, "A", annotation_args)

    labs = ["B", "C"]
    for ind1, filename in enumerate(fnames):
        with np.load(filename) as data:
            dose = data["doselist"]
            pinf = data["pinf"]
            pcar = data["pcar"]
            ps = data["ps"]
            print(pinf.shape)
            # r1 = data["r1"]
            # r2 = data["r2"]
        ax = plt.subplot(2, 2, ind1 + 2)
        if ind1 == 0:
            partition_plot(dose, pinf[0,], pcar[0,], ps[0,], ax, cols=cols)
        elif ind1 == 1:
            partition_plot(dose, pinf[0,], pcar[0,], ps[0,], ax, cols=cols)
        x1, x2 = plt.xlim()
        _, y2 = plt.ylim()
        plt.text(x1 - 0.15 * (x2 - x1), y2, labs[ind1], annotation_args)
    plt.show()


def soap_obj(col: List[str] = ["#4daf4a", "#ff7f00", "#e41a1c"], both: bool = False):
    """Plot soap fit data.

    Plot the fit of RH model, r1 and rmf hypotheses to data on SA kinetics 
    after washing with soap. 

    Parameters
    ----------
    col
        Colors of the r1 and rmf hypothesis fits.
    both
        Flag on whether to plot immediate inoculation. Inoculation after 24h is
        always plotted.
    
    Notes
    -----
    Get the relevant data from `get_soap_data`.
    """
    # Get data
    p1 = get_soap_data(1)
    p2 = get_soap_data(2)

    # For 24h dataset
    ptemp = p2
    ptemp["k1"] = ptemp["k1_24h"]
    ptemp["k2"] = ptemp["k2_24h"]
    solrh2 = solve_ivp(
        lambda t, y: rh_growth_model(t, y, ptemp),
        [0, np.max(ptemp["t"])],
        [10 ** ptemp["y0"]],
    )
    ptemp["r1"] = ptemp["r1*"]
    sol2cr2 = solve_ivp(
        lambda t, y: twocomp_model(t, y, ptemp),
        [0, np.max(ptemp["t"])],
        [10 ** ptemp["y0"], 0],
    )
    sol2crmf2 = solve_ivp(
        lambda t, y: twocomp_rmf_model(t, y, ptemp),
        [0, np.max(ptemp["t"])],
        [10 ** ptemp["y0"], 0],
    )

    # Plots
    plt.plot(p2["t"], p2["y"], "ko", label="Data (+24h)")
    if both:
        plt.plot(p1["t"], p1["y"], "kx", label="Data (imm)")
    sse, aicc = round(p2["sse_rh"], 3), round(p2["aicc_rh"], 2)
    plt.plot(
        solrh2.t,
        np.log10(solrh2.y.transpose()),
        color="grey",
        linestyle="--",
        label=f"RH (SSE={sse}, AICc={aicc})",
    )
    sse, aicc = round(p2["sse_r1"], 3), round(p2["aicc_r1"], 2)
    plt.plot(
        sol2cr2.t,
        np.log10(sol2cr2.y.transpose().sum(axis=1)),
        color=col[0],
        label="$r1$ " + f"(SSE={sse}, AICc={aicc})",
    )
    sse, aicc = round(p2["sse_rmf"], 3), round(p2["aicc_rmf"], 2)
    plt.plot(
        sol2crmf2.t,
        np.log10(sol2crmf2.y.transpose().sum(axis=1)),
        color=col[1],
        label="$r_{mf}$" + f"(SSE={sse}, AICc={aicc})",
    )

    if both:
        # Solve integration problems
        # For imm dataset
        ptemp = p1
        ptemp["k1"] = ptemp["k1_imm"]
        ptemp["k2"] = ptemp["k2_imm"]
        solrh1 = solve_ivp(
            lambda t, y: rh_growth_model(t, y, ptemp),
            [0, np.max(ptemp["t"])],
            [10 ** ptemp["y0"]],
        )
        ptemp["r1"] = ptemp["r1*"]
        sol2cr1 = solve_ivp(
            lambda t, y: twocomp_model(t, y, ptemp),
            [0, np.max(ptemp["t"])],
            [10 ** ptemp["y0"], 0],
        )
        sol2crmf1 = solve_ivp(
            lambda t, y: twocomp_rmf_model(t, y, ptemp),
            [0, np.max(ptemp["t"])],
            [10 ** ptemp["y0"], 0],
        )
        plt.plot(solrh1.t, np.log10(solrh1.y.transpose()), color="grey", linestyle="--")
        sse, aicc = round(p1["sse_r1"], 2), round(p1["aicc_r1"], 2)
        plt.plot(sol2cr1.t, np.log10(sol2cr1.y.transpose().sum(axis=1)), color=col[0])
        sse, aicc = round(p1["sse_rmf"], 2), round(p1["aicc_rmf"], 2)
        plt.plot(
            sol2crmf1.t, np.log10(sol2crmf1.y.transpose().sum(axis=1)), color=col[1]
        )
        plt.ylim([0, 6])

    plt.xlabel("Time (days)")
    plt.ylabel("Staph. density (CFU/cm$^2$)")
    plt.legend()
