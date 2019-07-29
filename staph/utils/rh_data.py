import numpy as np
from .dev import compute_deviance
from typing import Tuple


def get_rh_fit_data() -> Tuple[float, float, np.array, float]:
    """Get Rose and Haas best fit sse and deviance.

    Returns
    -------
    x
        Tuple with 4 floats representing best fit sse, best fit dev, 
        integrated dose and best fit k presented in Rose and Haas.

    References
    ----------
    .. [1] Rose, J. B., & Haas, C. N. (1999). A risk assessment framework for 
    the evaluation of skin infections and the potential impact of 
    antibacterial soap washing. American Journal of Infection Control, 27(6), 
    S26-33. Retrieved from http://www.ncbi.nlm.nih.gov/pubmed/10586143
    """
    rh_best_sse = 0.73
    integrated_dose = np.array(
        [2.4280e06, 6.2665e06, 1.2732e07, 2.4983e07, 3.3440e07, 3.9136e07]
    )  # days . #/cm^2
    k = 1.31e7  # days . #/cm^2
    rh_best_dev = 0
    for ind in range(len(integrated_dose)):
        response = 1 - np.exp(-integrated_dose[ind] / k)
        rh_best_dev += compute_deviance(response, ind)
    x = rh_best_sse, rh_best_dev, integrated_dose, k
    return x
