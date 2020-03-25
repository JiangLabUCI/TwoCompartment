import numpy as np
import matplotlib.pyplot as plt
from ..staph.utils.predict import predict_fit


def test_predict():
    pop_array, t_array = predict_fit(
        n_cores=8,
        nrep=1,
        nstep=200000,
        rank_1_sol_inds=[0],
        seed=0,
        doselist=[1e7],
        hyp="rmf",
        inoc_time="24h",
        pop_array_flag=True,
    )
    # pop_array2, t_array2 = predict_fit(
    #     filename="results/rank_1_solutions.csv",
    #     n_cores=8,
    #     nrep=10,
    #     nstep=5000,
    #     rank_1_sol_inds=[0, 5],
    #     seed=0,
    #     doselist=[1000],
    #     hyp="base",
    #     inoc_time="base",
    #     pop_array_flag=True,
    # )
    for ind in range(len(t_array)):
        plt.plot(t_array[ind], np.log10(pop_array[ind][0, :]), "r")
        plt.plot(t_array[ind], np.log10(pop_array[ind][1, :]), "b")
        # plt.plot(t_array2[ind], pop_array2[ind][0, :], color="darkred")
        # plt.plot(t_array2[ind], pop_array2[ind][1, :], color="darkblue")

    # plt.show()
