import numpy as np
import matplotlib.pyplot as plt
from ..utils.data import get_kinetic_data_params
from ..utils.rh_data import get_rh_fit_data
from scipy.integrate import solve_ivp
from typing import List, Dict

col_mo = ["#984ea3", "#ff7f00"]
annotation_args = {"va": "bottom", "weight": "bold", "fontsize": "12"}


def growth_obj(col: List[str], solinds: List[int] = [0]):
    """Plot growth data and fits.

    Plots the kinetic/growth data of SA, best fit RH model and two rank 1
    solutions of the 2C model.

    Parameters
    ----------
    col
        Colors of the 2C solutions.
    solinds
        Indices of the rank 1 solutions to plot.
    """
    fname = "results//rank_1_solutions.npz"
    with np.load(fname) as data:
        df = data["df"]

    p = get_kinetic_data_params()
    sse_rh = get_rh_fit_data()
    sse_rh = sse_rh[0]

    # RH model
    # Solve integration problem
    solrh = solve_ivp(lambda t, y: rh_growth_model(t, y, p), [0, 6], p["initial_rh"])

    # Make plots
    for ind in range(3):
        if ind == 1:
            plt.plot(
                solrh.t,
                np.log10(solrh.y[ind, :].transpose()),
                color="grey",
                linestyle="--",
                label=f"RH (SSE = {round(sse_rh,2)})",
            )
        else:
            plt.plot(
                solrh.t,
                np.log10(solrh.y[ind, :].transpose()),
                color="grey",
                linestyle="--",
            )

    # 2C model
    # For each solution index
    for ind1 in range(len(solinds)):
        this_sol_ind = solinds[ind1]
        p["r1"] = df[this_sol_ind, 0]
        p["r2"] = df[this_sol_ind, 1]
        p["r3"] = df[this_sol_ind, 2]
        p["r3Imax"] = df[this_sol_ind, 3]
        rmse_2c = df[this_sol_ind, 4]

        # Solve integration problems
        twoc_t = []
        twoc_y = []
        for ind2 in range(3):
            sol2c = solve_ivp(
                lambda t, y: twocomp_model(t, y, p),
                [0, 6],
                [p["initial_rh"][ind2], 0],
                dense_output=True,
            )
            twoc_t.append(sol2c.t)
            twoc_y.append(np.log10(sum(sol2c.y, 1)))

            if ind2 == 1:
                plt.plot(
                    twoc_t[ind2],
                    twoc_y[ind2],
                    color=col[ind1],
                    label=f"2C (SSE = {round(rmse_2c,2)})",
                )
            else:
                plt.plot(twoc_t[ind2], twoc_y[ind2], color=col[ind1])

    # Data

    plt.plot(p["t1"], p["y1"], "ko", label="Data")
    plt.plot(p["t2"], p["y2"], "ko")
    plt.plot(p["t3"], p["y3"], "ko")
    plt.ylim([0, 8])
    plt.xlabel("Time (days)")
    plt.ylabel("Staph. density (CFU/cm$^2$)")

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 0, 1, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.show()


def rh_growth_model(t: float, y: float, p: Dict) -> float:
    """Rose and Haas model.

    Returns the derivative of SA density at a given time `t`.

    Parameters
    ----------
    t
        Time at which derivative is needed.
    y
        SA density (CFU/cm^2).
    p
        Dictionary containing the parameters k1, k2, k3 and Nmax.
    
    Returns
    -------
    dy
        Derivative of SA density.

    Notes
    -----
    The model for SA density is given in [1]_ and is

    .. math:: \frac{dy}{dt} = -k_1 e^{-k_2 t} y + k_3 y (N_{max} - y)

    References
    ----------
    .. [1] Rose, J. B., & Haas, C. N. (1999). A risk assessment framework for 
    the evaluation of skin infections and the potential impact of 
    antibacterial soap washing. American Journal of Infection Control, 27(6),
     S26-33. Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/10586143
    """
    dy = -p["k1"] * np.exp(-p["k2"] * t) * y + p["k3"] * y * (p["Nmax"] - y)
    return dy


def twocomp_model(t: float, y: List[float], p: Dict) -> float:
    """2C model.

    Returns the derivative of SA density at a given time `t`.

    Parameters
    ----------
    t
        Time at which derivative is needed.
    y
        SA density (CFU/cm^2).
    p
        Dictionary containing the parameters r1, r2, r3 and r3Imax.
    
    Returns
    -------
    dy
        Derivative of SA density.

    Notes
    -----
    The model for SA density is given by

    .. math:: \frac{dy_0}{dt} = -r_1 y_0 - r_2 y_0
    .. math:: \frac{dy_1}{dt} = r_2 y_0 + r_3I_{max} y_1 - r_3 y_1^2
    """
    dhdt = -p["r1"] * y[0] - p["r2"] * y[0]
    didt = p["r2"] * y[0] + p["r3Imax"] * y[1] - p["r3"] * y[1] * y[1]
    return [dhdt, didt]

