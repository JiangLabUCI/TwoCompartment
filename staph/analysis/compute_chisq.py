from scipy.stats import chi2
import pandas as pd
from typing import List
from ..utils.rh_data import get_rh_fit_data


def get_gof_values(dev: float, k: int, m: int, sig: float = 0.05) -> List[float]:
    """Get goodness of fit values.

    Return the deviance, degrees of freedeom, chisquared critical value
    and the p-value.

    Parameters
    ----------
    dev
        Deviance of the model.
    k
        Number of data points.
    m
        Number of parameters.
    sig
        Significance, defaults to 0.05.

    Returns
    -------
    dev
        Deviance of the model.
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
    p = 1 - chi2.cdf(dev, k - m)
    conc = "Accept" if p > sig else "Reject"
    return [dev, dof * 1.0, crit, p, conc]


def compute_chisq(dev_2c: float = 11.67):
    """Print goodness of fit.

    Parameters
    ----------
    dev_2c
        Deviance of the 2c model.
    """
    _, dev_rh, _, _ = get_rh_fit_data()

    k = 6
    m_rh = 5  # k1, k2, k3, Nmax, k
    m_rh_dronly = 1  # k

    m_2c = 6  # r1, r2, r3, Imax, b1, d2
    m_2c_dronly = 2  # b1, d2

    df = pd.DataFrame(
        columns=["Name", "Deviance", "Dof", "Crit value", "p value", "Conclusion"]
    )
    df.loc[1] = ["RH (total)"] + get_gof_values(dev_rh, k, m_rh)
    df.loc[2] = ["RH (dose response only)"] + get_gof_values(dev_rh, k, m_rh_dronly)
    df.loc[3] = ["2C (total)"] + get_gof_values(dev_2c, k, m_2c)
    df.loc[4] = ["2C (dose response only)"] + get_gof_values(dev_2c, k, m_2c_dronly)

    print(df)
