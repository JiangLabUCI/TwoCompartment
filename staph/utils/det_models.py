import numpy as np
from typing import Dict, List


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


def twocomp_rmf_model(t: float, y: List[float], p: Dict) -> float:
    """2C model with rmf.

    Returns the derivative of SA density at a given time `t`.

    Parameters
    ----------
    t
        Time at which derivative is needed.
    y
        SA density (CFU/cm^2).
    p
        Dictionary containing the parameters r1, r2, r3, r3Imax and rmf.
    
    Returns
    -------
    dy
        Derivative of SA density.

    Notes
    -----
    The model for SA density is given by

    .. math:: \frac{dy_0}{dt} = -r_1 y_0 - r_2 y_0
    .. math:: \frac{dy_1}{dt} = r_2 y_0 + r_3I_{max} y_1 - r_3 y_1^2 - rmf y_1
    """
    dhdt = -p["r1"] * y[0] - p["r2"] * y[0] - p["rmf"] * y[0]
    didt = p["r2"] * y[0] + p["r3Imax"] * y[1] - p["r3"] * y[1] * y[1] - p["rmf"] * y[1]
    return [dhdt, didt]
