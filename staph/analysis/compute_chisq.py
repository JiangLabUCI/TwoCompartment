import numpy as np
from scipy.stats import chi2
import pandas as pd
from typing import List
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
    conc = "Accept" if p > sig else "Reject"
    return [test_stat, dof * 1.0, crit, p, conc]


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
        columns=["Name", "Deviance", "Dof", "Crit value", "p value", "Conclusion"]
    )
    df.loc[1] = ["RH (total)"] + get_gof_values(dev_rh, k, m_rh)
    df.loc[2] = ["2C (total)"] + get_gof_values(dev_2c[0], k, m_2c)
    df.loc[3] = ["RH (dose response only)"] + get_gof_values(dev_rh, k, m_rh_dronly)
    for ind in range(len(dev_2c)):
        df.loc[4 + ind] = ["2C (dose response only)"] + get_gof_values(
            dev_2c[ind], k, m_2c_dronly
        )

    print(df)


def decision_kimura_chisq(p, sig=0.05):
    """Return decision.

    Return the hypothesis decision based on the p value and significance level.

    Parameters
    ----------
    p
        P value of the decision boundary.
    sig
        Significance level, defaults to 0.05.

    """
    decision = "Unable to reject simpler model" if p > sig else "Reject simpler model"
    return decision


def sse_chisq(
    sse_2par: float = 0.003, sse_1par: float = 0.925, sd_estimate: float = 1.0
):
    """Compute chi-squared statistic.

    Use the estimator given in Kimura 1990 to compute chi squared statistic.

    Parameters
    ----------
    sse_2par
        Sum of squared error for 2 parameter model.
    sse_1par
        Sum of squared error for 1 parameter model.
    sd_estimate
        Estimate of the standard deviation.

    Notes
    -----
    This is the estimator when all the sigma^2 are known. Given by equation (10) in [1]_.

    References
    ----------
    .. [1] Kimura, D. K. (1990). Testing Nonlinear Regression Parameters under
    Heteroscedastic, Normally Distributed Errors. Biometrics, 46(3), 697. 
    https://doi.org/10.2307/2532089
    """

    assert sse_1par > sse_2par

    p = 2  # Number of parameters
    I = 4  # Number of populations (datapoints)
    dof = p * (I - 1)  # degrees of freedom

    print(f"Using an estimated sd = {sd_estimate}")

    chisq_v = (sse_1par - sse_2par) / sd_estimate ** 2
    P = 1 - chi2.cdf(chisq_v, dof)
    sig = 0.05
    conc = decision_kimura_chisq(p=P, sig=sig)
    print(conc + f" : P = {P:.5e}")
    return P, conc
