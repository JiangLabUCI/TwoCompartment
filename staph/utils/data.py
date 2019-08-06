import numpy as np
import pandas as pd
from typing import Tuple, List, Any, Dict
from scipy.stats import truncnorm
from numba import njit


@njit(cache=False)
def get_singh_data() -> Tuple[List[int], List[int], int, float, float, Any]:
    """Return Singh 1971 data.

    Return h0, norig, ntot, tiny, A and H0.

    Returns
    -------
    h0 : array_like
        The inoculum density (CFU/cm^2).
    norig : array_like
        The number of people showing positive response.
    ntot : float
        The total number of people receiving each inoculum. This is the same
        for each dose and only a float is returned.
    tiny : float
        A small number.
    A : float
        Area over which the inoculum is spread (cm^2).
    H0 : array_like (int)
        The inoculum load (CFU).
    
    Notes
    -----
    H0 = h0 * A
    """
    h0 = [40, 220, 2000, 105_000, 1_600_000, 10_000_000]  # CFU/cm^2
    norig = [4, 8, 13, 14, 19, 20]
    ntot = 20
    tiny = 1e-12
    A = 3.0  # cm^2
    H0 = np.array(h0, dtype=np.int32) * np.int32(A)  # CFU

    return h0, norig, ntot, tiny, A, H0


def get_kinetic_data_params() -> Dict:
    """Get kinetic data and params.

    Returns
    -------
    p
        Dictionary with kinetic data and parameters.
    """
    # From Singh 1971
    data = np.genfromtxt("staph/data/singh1971_fig1.csv", delimiter=",")
    data = np.round(data, 2)
    p = {}
    p["t1"] = [0, 1, 2, 3, 4, 6]  # days
    p["y01"] = np.log10(4e2)
    p["y1"] = np.array([p["y01"], *data[1:6, 1]])  # log bacteria/cm^2
    p["t2"] = [0, 1, 2, 3, 4]  # days
    p["y02"] = np.log10(1.5e4)
    p["y2"] = [p["y02"], *data[7:11, 1]]  # log bacteria/cm^2
    p["t3"] = [0, 1, 2, 3, 4, 6]  # days
    p["y03"] = np.log10(2e7)
    p["y3"] = [p["y03"], *data[12:17, 1]]  # log bacteria/cm^2

    # From Rose and Haas 1999
    p["k1"] = 1.8175e18
    p["k2"] = 3.8103e17
    p["k3"] = 3.2136e-7
    p["Nmax"] = 8_930_893
    p["initial_rh"] = 10 ** np.array([p["y01"], p["y02"], p["y03"]])
    p["initial_2c"] = 10 ** np.array([[p["y01"], 0], [p["y02"], 0], [p["y03"], 0]])

    return p


def get_b1d2(
    b2: float, d1: float, r3: float, r3Imax: float, A: float = None
) -> Tuple[float, float]:
    """Find b1 and d2 given other paramters.

    Return b1 and d2 stochastic parameters given b1, d2, r3 and r3Imax.

    Parameters
    ----------
    b2 : float
        Second order birth rate of twocomp model (stochastic,
        units = 1 / (bacteria * day)).
    d1 : float
        First order death rate of twocomp model. (stochastic,
        units = 1 / day)
    r3 : float
        Second order logistic rate of twocomp model. (deterministic,
        units = cm^2 / (bacteria * day))
    r3Imax : float
        First order logistic rate of twocomp model. (deterministic,
        units = 1 / day)
    A : float
        If `None`, use from `get_singh_data`. Else use the value provided.

    Returns
    -------
    b1 : float
        First order birth rate of twocomp model (stochastic,
        units = 1 / day)
    d2 : float
        Second order death rate of twocomp model (stochastic,
        units = 1 / (bacteria * day)).

    Notes
    -----
    b1 - d1 = r3 * Imax     =>   b1 = d1 + r3 * Imax
    b2 - d2 = 2 * (-r3)/A   =>   d2 = b2 + 2*r3/A
    b1 has same units as d1 (1 / day)
    d2 has same units as b2 (1 / (bacteria * day))
    """

    if A is None:
        _, _, _, _, A, _ = get_singh_data()
    b1 = d1 + r3Imax
    d2 = b2 + 2 * r3 / A
    return b1, d2


def calc_for_map(arguments, func):
    """
    Wrapper function to be used with multiprocessing.map to evaluate
    the objective.
    """
    return func(*arguments)


def get_k_ec(strain: int = 1) -> Tuple[float, float, float, float]:
    """Return antibiotic constants.

    Antibiotic constants for MRSA under Ciprofloxacin influence from [1]_.

    Parameters
    ----------
    strain
        Integer (1 or 2) referring to the strain of MRSA (8043 or 8282).

    References
    ----------
    .. [1] Campion, J. J., McNamara, P. J., & Evans, M. E. (2005). 
    Pharmacodynamic Modeling of Ciprofloxacin Resistance in Staphylococcus 
    aureus. Antimicrobial Agents and Chemotherapy, 49(1), 209–219. 
    https://doi.org/10.1128/AAC.49.1.209-219.2005
    """
    assert strain in [1, 2]
    if strain == 1:
        # MRSA 8043
        ks = 1.56  # hr^-1
        EC50s = 0.21  # \mu g/mL
        kr = 1.17  # hr^-1
        EC50r = 5.19  # \mu g/mL
    elif strain == 2:
        # MRSA 8282
        ks = 1.39  # hr^-1
        EC50s = 0.18  # \mu g/mL
        kr = 1.10  # hr^-1
        EC50r = 2.72  # \mu g/mL
    return (ks, EC50s, kr, EC50r)


def get_fcs_fcr(C: float, strain: int = 1) -> Tuple[float, float]:
    """Return effect of antibiotic.

    For a given concentration of antibiotic C and strain of MRSA, return the
    killing rate of the antibiotic on susceptible and resistan subpopulations.

    Parameters
    ----------
    C
        Concentration of antibiotic (\\mu g/mL).
    strain
        Integer (1 or 2) referring to the strain of MRSA (8043 or 8282).
    """
    ks, EC50s, kr, EC50r = get_k_ec(strain)
    # Effect of antibiotic
    fcs = ks * C / (EC50s + C)  # On susceptible population
    fcr = kr * C / (EC50r + C)  # On resistant population
    return fcs, fcr


def get_oie_data(n: int = 10) -> np.ndarray:
    """Get sample of MRSA load in bed-sheets.

    Return MRSA load (CFU) in 100cm^2 of bed-sheets.

    Parameters
    ----------
    n
        The number of samples.

    Returns
    -------
    x
        The MRSA load samples.

    Notes
    -----
    From [1]_, Table 1.
    
    References
    ----------
    .. [1] Oie, S., Suenaga, S., Sawa, A., & Kamiya, A. (2007). Association
    between isolation sites of methicillin-resistant Staphylococcus aureus 
    (MRSA) in patients with MRSA-positive body sites and MRSA contamination in
    their surrounding environmental surfaces. Japanese Journal of Infectious 
    Diseases, 60(6), 367–369.

    """
    mu = 380.2  # CFU
    sigma = 2198  # CFU
    np.random.seed(0)
    x = np.random.normal(mu, sigma, 10000)  # CFU
    x = x[x > 0]
    x = np.int32(x[:n])
    return x


def get_occurence_dist(n: int = 100) -> np.ndarray:
    """Return MRSA densities/cm^2.

    Use parameters defined in [1]_ to sample MRSA density values.

    Parameters
    ----------
    n
        The number of random samples.
    
    Returns
    -------
    rv
        The sampled random values. (CFU/cm^2)

    References
    ----------
    .. [1] Kurashige, E. J. O., Oie, S., & Furukawa, H. (2016). 
    Contamination of environmental surfaces by methicillin-resistant 
    Staphylococcus aureus (MRSA) in rooms of inpatients with MRSA-positive 
    body sites. Brazilian Journal of Microbiology, 47(3), 703–705. 
    https://doi.org/10.1016/j.bjm.2016.04.002
    """
    mean = 159.5
    std = 396.4
    maxval = 1620
    minval = 0
    area = 100  # cm^2
    # The clip values are defined over the range of the standard normal.
    # Hence compute them according to
    a, b = (minval - mean) / std, (maxval - mean) / std
    rv = truncnorm.rvs(a=a, b=b, loc=mean, scale=std, size=n) / area
    return rv


def get_bedrail_data(
    n: int = 10, tmax: float = 6.0, sex: str = "F", seed: int = 0
) -> Tuple[List[float], List[int], float]:
    """Return MRSA exposure time points and loads.

    Use data from several publications to construct samples of MRSA exposure 
    time points and loads.

    Parameters
    ----------
    n
        The number of samples to return.
    tmax
        The maximum time to simulate until. (in days)
    sex
        The sex of the person being simulated for. (Hand size depends on sex)
    seed
        The seed of the numpy random number generator.

    Returns
    -------
    times 
        A tuple of `n` MRSA exposure time point sequences. Each sequence is 
        a `np.ndarray` of time points from 0 to tmax at which exposure occurs.
    loads
        A tuple of `n` MRSA exposure load sequences. Each sequence is 
        a `np.ndarray` of loads corresponding to the sequence of exposure time
        points in `times` parameter.
    A
        The area of the hand, to be used to calculate the rates.

    Notes
    -----
    1. Contact frequency [1]_
    Table II: -> Patients touch bedside rails 111 times
    Methods: -> Fully occupied 6 bed cubicle
             -> 66 observation hours
    111 contacts / (6 persons * 66 hours) or
    111 / (6 * 66) contact/(person hour)

    2. MRSA load from [2]_
    Table 1: Use `get_occurence_dist` to sample the densities.

    3. Hand size from [3]_
    Men: 146.5 cm^2
    Women: 132.42 cm^2

    4. Transfer efficiency from [4]_
    Abstract: Findings
    Transfer from rail to fingertip ranged from 22% to 38%.

    References
    ----------
    .. [1] Cheng, V. C. C., Chau, P. H., Lee, W. M., Ho, S. K. Y., 
    Lee, D. W. Y., So, S. Y. C., … Yuen, K. Y. (2015). Hand-touch 
    contact assessment of high-touch and mutual-touch surfaces among 
    healthcare workers, patients, and visitors. Journal of Hospital 
    Infection, 90(3), 220–225. https://doi.org/10.1016/j.jhin.2014.12.024 
    .. [2] Kurashige, E. J. O., Oie, S., & Furukawa, H. (2016). 
    Contamination of environmental surfaces by methicillin-resistant 
    Staphylococcus aureus (MRSA) in rooms of inpatients with MRSA-positive 
    body sites. Brazilian Journal of Microbiology, 47(3), 703–705. 
    https://doi.org/10.1016/j.bjm.2016.04.002
    .. [3] Agarwal, P., & Sahu, S. (2010). Determination of hand and palm 
    area as a ratio of body surface area in Indian population. Indian 
    Journal of Plastic Surgery : Official Publication of the Association 
    of Plastic Surgeons of India, 43(1), 49–53. 
    https://doi.org/10.4103/0970-0358.63962
    .. [4] Ali, S., Moore, G., & Wilson, A. P. R. (2012). Effect of surface 
    coating and finish upon the cleanability of bed rails and the spread of 
    Staphylococcus aureus. Journal of Hospital Infection, 80(3), 192–198. 
    https://doi.org/10.1016/j.jhin.2011.12.005
    """

    np.random.seed(seed)
    # contact/(person hour) * hour/day
    contact_frequency = 111 / (6 * 66) * 24  # contacts/(person day)
    if sex == "F":
        hand_size = 132.42  # cm^2
    else:
        hand_size = 146.50  # cm^2
    min_efficiency = 0.22
    max_efficiency = 0.38

    times = []
    loads = []
    for _ in range(n):
        t = np.random.exponential(
            scale=1 / contact_frequency,
            size=np.int32(np.round(tmax * contact_frequency * 2)),
        )
        t2 = np.cumsum(t)  # days
        t = t[t2 < tmax]
        n = len(t)
        this_mrsa_density = get_occurence_dist(n)  # CFU / cm^2
        this_transfer_eff = np.random.uniform(
            low=min_efficiency, high=max_efficiency, size=n
        )
        # (CFU / cm^2) * (cm^2) * (efficiency)
        load = np.int32(np.round(this_mrsa_density * hand_size * this_transfer_eff))
        times.append(t)
        loads.append(load)
    return times, loads, hand_size


def get_soap_data(dsno: int = 1, parno: int = 5) -> Dict:
    """Data from Rose 1999.

    Get the original data, RH fitted constants and 2C fitted constants.

    Parameters
    ----------
    dsno
        Dataset number. 1 (control soap, 24h before inoculation), 
        2 (control soap, immediate inoculation)
    parno
        Index of rank 1 solutions in [0,5]. 
    
    Returns
    p
        Dictionary of relevant parameters and data.
    
    """
    # Helper constants
    permin_to_perday = 60 * 24  # (min/hr) * (hr/day) = min/day
    min_to_day = 1 / 60 * 1 / 24  # (hr/min) * (day/hr) = day/min

    p = {}

    # Original data
    # Times at which data was collected
    p["tmins"] = np.array([0, 30, 120, 300])  # minutes
    p["t"] = p["tmins"] * min_to_day  # min * day/min -> day

    # Experimentally observed data
    data = np.genfromtxt("results/rose1999_fig5.csv", delimiter=",")
    data = np.round(data, 2)
    if dsno == 1:
        p["y0"] = np.log10(data[0, 1])  # Initial load
        p["y"] = np.log10(data[:4, 1])
    elif dsno == 2:
        p["y0"] = np.log10(data[4, 1])  # Initial load
        p["y"] = np.log10(data[4:, 1])

    # RH fitted constants
    p["k1_imm"] = 0.105 * permin_to_perday  # /min * min/day = /day
    p["k2_imm"] = 0.0383 * permin_to_perday  # /min * min/day = /day
    p["k1_24h"] = 0.167 * permin_to_perday  # /min * min/day = /day
    p["k2_24h"] = 0.0586 * permin_to_perday  # /min * min/day = /day
    p["k3"] = 3.2136e-7  # cm^2/(# * day)
    p["Nmax"] = 8_930_893  # #/cm^2

    data2 = pd.read_csv("results/pred_consts.csv")
    p["sse_rh"] = float(data2[(data2.Dataset == dsno) & (data2.Hyp == -1)].SSE)
    p["aicc_rh"] = np.inf

    # 2C fitted constants
    data2 = data2[(data2.Dataset == dsno) & (data2.Parameterset == parno + 1)]
    p["r1*"] = float(data2[data2.Hyp == 2].Parameter)
    p["sse_r1"] = float(data2[data2.Hyp == 2].SSE)
    p["aicc_r1"] = float(data2[data2.Hyp == 2].AICc)
    p["rmf"] = float(data2[data2.Hyp == 6].Parameter)
    p["sse_rmf"] = float(data2[data2.Hyp == 6].SSE)
    p["aicc_rmf"] = float(data2[data2.Hyp == 6].AICc)

    data3 = pd.read_csv("results/rank_1_solutions.csv")
    p["r2"] = float(data3.iloc[parno].r2)
    p["r3"] = float(data3.iloc[parno].r3)
    p["r3Imax"] = float(data3.iloc[parno]["r3*Imax"])

    return p
