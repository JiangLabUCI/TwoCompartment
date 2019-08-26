import numpy as np
from numba import njit
from typing import Tuple
from .roulette import roulette


@njit(cache=False)
def get_propensity(rates: np.ndarray, curH: int, curI: int) -> np.ndarray:
    """
    Returns the propensity for the two-comp model.

    Parameters
    ----------
    rates
        The stochastic rate parameters in the following order:
        r1, r2, b1, b2, d1, d2, rmf.
    curH
        The number of bacteria in the pre-infection compartment.
    curI
        The number of bacteria in the pre-infection compartment.

    Returns
    -------
    prop
        The propensity vector for the two compartment model.

    Notes
    -----
    The elementary reactions of the model are :

    0 : H -r1-> \\phi
    1 : H -r2-> I
    2 : I -b1-> I + I
    3 : I + I -b2-> I + I + I
    4 : I -d1-> \\phi
    5 : I + I -d2-> I
    6 : H -rmf-> \\phi
    7 : I -rmf-> \\phi

    All rate constants are stochastic rate constants ($c_{\\mu}$ in Gillespie
    1976) unless specified otherwise. Units of $b_2$ and $d_2$ are
    1 / (bacteria * day))
    """

    r1, r2 = rates[0], rates[1]
    b1, b2 = rates[2], rates[3]
    d1, d2 = rates[4], rates[5]
    rmf = rates[6]
    prop = np.zeros(8, dtype=np.float64)
    prop[0] = r1 * curH
    prop[1] = r2 * curH
    prop[2] = b1 * curI
    prop[3] = b2 * curI * (curI - 1) / 2
    prop[4] = d1 * curI
    prop[5] = d2 * curI * (curI - 1) / 2
    prop[6] = rmf * curH
    prop[7] = rmf * curI
    return prop


@njit(cache=False)
def tau_twocomp_carrier_rmf(
    init_load: np.ndarray,
    rates: np.ndarray,
    imax: float,
    nstep: int,
    seed: int,
    t_max: float = 6.0,
    store_flag: bool = True,
) -> Tuple[int, float, np.ndarray, np.ndarray, int]:
    """Simulates the two-comp model with rmf.

    Parameters
    ----------
    init_load
        The initial number of bacteria in the pre-infection compartment or
        both compartments.
    rates
        The stochastic rate parameters in the following order:
        r1, r2, b1, b2, d1, d2, rmf.
    imax
        The maximum population size (units of CFU).
    nstep
        The maximum number of steps to execute the simulation for.
    seed
        The seed of the numpy random generator.
    t_max
        The stopping time of the simulation.
    store_flag
        The states are saved if this flag is True. Defaults to True.

    Returns
    -------
    extflag : int
        Set to 1 if the bacteria go extinct.
    t : float
        The final time of the simulation.
    pop_array : (4, iter,) array_like
        The population states over the course of the simulation.
    t_array : (iter,) array_like
        The sampling times over the course of the simulation.
    status : int
        The status of the simulation end.
        0 : Status not set.
        1 : Succesful completion, terminated when all species went extinct.
        2 : Succesful completion, terminated when `max_t` crossed.
        3 : Succesful completion, terminated when I(t) > imax.
        4 : Succesful completion, curI overflow.
        5 : Succesful completion, curH overflow.
        -1 : Somewhat succesful completion, terminated when `max_iter` reached.
        -2 : Unsuccessful completion, curI overflow or curH + curI overflow.

    Notes
    -----
    The elementary reactions of the model are in get_propensity function.

    See Also
    --------
    get_propensity : Calcultes the propensity vector for this model.
    """
    # Set the seed
    np.random.seed(seed)

    extflag = 0
    status = 0

    if init_load.shape[0] == 1:
        curH, curI = init_load[0], 0
    else:
        curH, curI = init_load[0], init_load[1]

    # Propensity array and total
    prop_array = get_propensity(rates, curH, curI)
    prop_total = np.sum(prop_array)
    nprop = 8
    prop_crit = np.zeros(nprop, dtype=np.float64)

    # Tau leaping parameters
    nc = 10
    ind = 0
    tiny = 1e-12
    high = 1e12
    eps = 0.03
    u, t = 0.0, 0.0
    bflag = 0
    pop_array = np.zeros((2, nstep + 1), dtype=np.int64)
    pop_array[0, 0], pop_array[1, 0] = curH, curI
    t_array = np.zeros(nstep + 1)

    # Model parameters
    J = np.ones(nprop, dtype=np.int32)
    K = np.zeros(nprop, dtype=np.int32)
    g0, g1 = 1.0, 0.0
    maxval = np.array([0, 0], dtype=np.float64)
    mu = np.array([0, 0], dtype=np.float64)
    sigsq = np.array([0, 0], dtype=np.float64)
    vH = np.array([-1, -1, 0, 0, 0, 0, -1, 0], dtype=np.int32)
    vI = np.array([0, 1, 1, 1, -1, -1, 0, -1], dtype=np.int32)

    while ind < nstep:
        prop_array = get_propensity(rates, curH, curI)
        prop_total = np.sum(prop_array)
        if prop_total == 0:
            prop_total = tiny

        # 1. Number of times each rection can fire for reactions with vij<0
        # A reaction is critical if Lj < nc and aj > 0. aj > 0 merely means
        # that it is not extinct.
        # Skip the calculation of Lj and directly evaluate if reactions
        # are critical. If Rj is critical, `Jj = 0`.

        # `prop0 > 0` if `0 < curH`. `L0` is `curH`
        # if `L0 < nc` , J0 = 0, else 1
        J[0] = 0 if ((0 < curH) and (curH < nc)) else 1  # = J1 = J6
        J[1] = J[0]
        J[6] = J[0]
        # J2, J3 = 1 always, never critical (birth reactions)
        # `prop4 > 0` if `0 < curI`. `L4` is `curI`
        # if `L4 < nc`, J4 = 0, else 1
        J[4] = 0 if ((0 < curI) and (curI < nc)) else 1  # = J7
        J[7] = J[4]
        # `prop5 > 0` if `1 < curI`. `L5 = curI - 1`.
        # if `L5 < nc`, J5 = 0, else 1
        J[5] = 0 if ((1 < curI) and (curI - 1 < nc)) else 1

        # 2. Generate candidate taup
        # Jncr or set of non critical reactions will never be empty because
        # reactions 2 and 3 are never critical.
        # 0 : curH, 1 : curI. Each one is a reactant species.
        mu[0] = np.abs(
            -1.0 * prop_array[0] * J[0]
            + -1.0 * prop_array[1] * J[1]
            - 1.0 * prop_array[6] * J[6]
        )
        mu[1] = np.abs(
            1.0 * prop_array[2] * J[2]
            + 1.0 * prop_array[3] * J[3]
            - 1.0 * prop_array[4] * J[4]
            - 1.0 * prop_array[5] * J[5]
            - 1.0 * prop_array[7] * J[7]
        )
        sigsq[0] = (
            1.0 * prop_array[0] * J[0]
            + 1.0 * prop_array[1] * J[1]
            + 1.0 * prop_array[6] * J[6]
        )
        sigsq[1] = (
            1.0 * prop_array[2] * J[2]
            + 1.0 * prop_array[3] * J[3]
            + 1.0 * prop_array[4] * J[4]
            + 1.0 * prop_array[5] * J[5]
            + 1.0 * prop_array[7] * J[7]
        )

        # If mu0 becomes zero, must avoid inf while calculating maxvals
        if mu[0] == 0:
            mu[0] = tiny
            sigsq[0] = tiny
        if mu[1] == 0:
            mu[1] = tiny
            sigsq[1] = tiny

        # Avoid infinite g2 by checking if g2 is 1. Same for g3.
        g0 = 1.0
        g1 = 2.0 if (curI == 1) else 2 + 1 / (curI - 1)
        taup = high

        maxval[0] = max(eps * curH / g0, 1.0)
        maxval[1] = max(eps * curI / g1, 1.0)

        # Equation 33 in Cao et al. 2006
        taup = min(
            maxval[0] / abs(mu[0]),
            maxval[0] * maxval[0] / sigsq[0],
            maxval[1] / abs(mu[1]),
            maxval[1] * maxval[1] / sigsq[1],
        )

        # 3. Do SSA if taup < 10/a0.
        if taup < 10 / prop_total:
            ind2 = 0
            while (ind < nstep) and (ind2 < 100):
                u = np.random.rand()
                t += -np.log(u) / prop_total
                reaction_index = roulette(prop_array=prop_array)
                curH += vH[reaction_index]
                curI += vI[reaction_index]

                # Update propensities
                prop_array = get_propensity(rates, curH, curI)
                prop_total = np.sum(prop_array)
                if prop_total == 0:
                    prop_total = tiny

                # Update loop indices
                ind2 = ind2 + 1
                ind = ind + 1
                if store_flag:
                    pop_array[0, ind], pop_array[1, ind] = curH, curI
                    t_array[ind] = t

                if curH < 0:
                    if prop_array[reaction_index + 1] < 0:
                        status = 5
                        extflag = 0
                        bflag = 1
                        break
                    else:
                        print("I don't think this should happen (curH < 0)")

                if curH + curI == 0:
                    status = 1
                    extflag = 1
                    bflag = 1
                    break
                if t > t_max:
                    status = 2
                    bflag = 1
                    break

            if bflag == 1:
                break
            continue

        # 4. Compute sum of rates of critical reactions
        for ind3 in range(nprop):
            prop_crit[ind3] = prop_array[ind3] * (1 - J[ind3])
        prop_crit_total = np.sum(prop_crit)
        u = np.random.rand()
        if prop_crit_total < tiny:
            prop_crit_total = tiny
        taupp = -np.log(u) / prop_crit_total

        # 5. Leap depending on taup and taupp
        # We want K to be reset to all 0s. This is so that by default,
        # a reaction will not fire. This avoids a third condition to set
        # K to 0 for critical reactions in the taupp section below.
        K[K < np.inf] = 0.0
        if taup < taupp:
            # Set tau to taup
            tau = taup
            # Compute number of times each rxn is fired
            for ind3 in range(nprop):
                # To avoid unnecessary random number generation, don't
                # generated Poisson numbers for critical reactions (which
                # have J = 0).
                if J[ind3] == 1:
                    K[ind3] = np.random.poisson(prop_array[ind3] * tau)
        else:
            tau = taupp
            # Fire all non critical reactions. Fire one critical reaction
            # once based on propensity.
            # First compute jc or crit_index_to_fire
            crit_index_to_fire = roulette(prop_array=prop_crit)
            for ind3 in range(nprop):
                if J[ind3]:
                    K[ind3] = np.random.poisson(prop_array[ind3] * tau)
                elif ind3 == crit_index_to_fire:
                    K[ind3] = 1

        # Update states based on K
        for ind2 in range(nprop):
            curH += vH[ind2] * K[ind2]
            curI += vI[ind2] * K[ind2]
        # curI += np.dot(v[1, :], K)
        t += tau
        # Increase loop index
        ind = ind + 1
        # Store the current state of the simulation
        if store_flag:
            pop_array[0, ind], pop_array[1, ind] = curH, curI
            t_array[ind] = t

        if curI < 0:
            if K[3] < 0 and rates[3] > 0:
                status = 4
            else:
                print("I don't think this should happen (curI < 0)")
                status = -2
            extflag = 0
            break

        if curH + curI < 0:
            print("I dont think should happen (curH + curI < 0)")
            status = -2
            extflag = 1
            break

        # Regular extinction criterion
        if curH + curI == 0:
            status = 1
            extflag = 1
            break

        if t > t_max:
            status = 2
            break

        # Stop if initial density is reached
        if (curI) > imax:
            status = 3
            extflag = 0
            break

    if ind >= nstep - 1:
        status = -1

    return (extflag, t, pop_array[:, : ind + 1], t_array[: ind + 1], status)
