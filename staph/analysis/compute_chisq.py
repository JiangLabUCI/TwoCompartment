import numpy as np
from scipy.stats import chi2
import pandas as pd
from typing import List
from tabulate import tabulate
from ..utils.rh_data import get_rh_fit_data


def get_gof_values(test_stat: float, k: int, m: int, sig: float = 0.05) -> List[float]:
    """Get goodness of fit values.

    Return the test statisic, degrees of freedeom, chisquared critical value
    and the p-value.

    Parameters
    ----------
    test_stat
        Test statisic of the model.
    k
        Number of data points.
    m
        Number of parameters.
    sig
        Significance, defaults to 0.05.

    Returns
    -------
    test_stat
        Test statisic of the model.
    dof
        Degrees of freedom.
    crit
        Critical chisquared value at 0.95 significance.
    p
        p-value of the fit.
    """
    dof = k - m
    sig = 0.05
    crit = chi2.ppf(1 - sig, k - m)
    p = 1 - chi2.cdf(test_stat, k - m)
    conc = "Fail to reject" if p > sig else "Reject"
    if k - m - 1 == 0:
        AIC = 1e2
    else:
        AIC = test_stat + 2 * m * (k) / (k - m - 1)
    BIC = test_stat + m * np.log(k)
    return [
        round(test_stat, 2),
        dof * 1.0,
        round(crit, 2),
        round(p, 2),
        conc,
        round(AIC, 2),
        round(BIC, 2),
    ]


def compute_chisq(dev_2c: List[float] = [11.67]):
    """Print goodness of fit.

    Parameters
    ----------
    dev_2c
        Deviances of the 2c model.
    """
    _, dev_rh, _, _ = get_rh_fit_data()

    k = 6
    m_rh = 5  # k1, k2, k3, Nmax, k
    m_rh_dronly = 1  # k

    m_2c = 6  # r1, r2, r3, Imax, b1, d2
    m_2c_dronly = 2  # b1, d2

    df = pd.DataFrame(
        columns=[
            "Name",
            "Deviance",
            "Dof",
            "Crit value",
            "p value",
            "Conclusion",
            "AIC",
            "BIC",
        ]
    )
    df.loc[1] = ["RH (total)"] + get_gof_values(dev_rh, k, m_rh)
    df.loc[2] = ["2C (total)"] + get_gof_values(dev_2c[0], k, m_2c)
    df.loc[3] = ["RH (dose response only)"] + get_gof_values(dev_rh, k, m_rh_dronly)
    for ind in range(len(dev_2c)):
        df.loc[4 + ind] = ["2C (dose response only)"] + get_gof_values(
            dev_2c[ind], k, m_2c_dronly
        )
    n = len(df)
    df.loc[n] = ["Het host (dose response only)"] + get_gof_values(1.02, k, 3)
    df.loc[n + 1] = ["Bet Pos (dose response only)"] + get_gof_values(5.49, k, 2)

    print(df)
    print(
        tabulate(
            df,
            tablefmt="latex_booktabs",
            # floatfmt=(".2f", ".2f", ".2e", ".2e", ".2f", ".2e", ".2f", ".2f"),
            showindex=False,
        )
    )