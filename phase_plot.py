import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

df = pd.read_csv("results/rank_1_solutions.csv")
print(df)
index = 4
r1 = df.r1[index]
r2 = df.r2[index]
r3 = df.r3[index]
r3Imax = df["r3*Imax"][index]


def f(Y, t):
    # y1, y2 = Y
    # return [y2, -np.sin(y1)]
    H, I = Y
    # if H < 0:
    #     H = 0
    hprime = -r1 * H - r2 * H
    iprime = r2 * H + r3Imax * I - r3 * I * I
    return [hprime, iprime]


# logflag = True
logflag = False
nquivs = 15

if logflag:
    y1 = 10 ** np.linspace(0, 7, 20)
    y2 = 10 ** np.linspace(0, 7, 20)
else:
    y1 = np.linspace(0.0, 1e7, nquivs)
    y2 = np.linspace(0.0, 1e7, nquivs)

Y1, Y2 = np.meshgrid(y1, y2)
t = 0
u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
NI, NJ = Y1.shape

for i in range(NI):
    for j in range(NJ):
        if logflag:
            x = Y1[i, j]
            y = Y2[i, j]
        else:
            x = Y1[i, j]
            y = Y2[i, j]
        yprime = f([x, y], t)
        u[i, j] = yprime[0]
        v[i, j] = yprime[1]


if logflag:
    Q = plt.quiver(
        np.log10(Y1),
        np.log10(Y2),
        np.log10(u + np.min(u) + 1),
        np.log10(v + np.min(v) + 1),
        color="r",
    )
else:
    Q = plt.quiver(Y1, Y2, u, v, color="k")

plt.plot(0, 0, "ro", label="Fixed point")
plt.plot(0, r3Imax / r3, "ro")

# y10s = [0.5e7, 1e7, 1e7, 1e7]
# y20s = [0, 0, 0.2e7, 1e7]

y10s = [1e7]
y20s = [0.2e7]

for ind in range(len(y10s)):
    # y10 = ind * (10 ** 6)
    # print(y10)
    tspan = np.linspace(0, 6, 20)
    y0 = [y10s[ind], y20s[ind]]
    ys = odeint(f, y0, tspan)
    if logflag:
        plt.plot(np.log10(ys[:, 0]), np.log10(ys[:, 1]), "b-")  # path
        plt.plot(
            np.log10([ys[0, 0]]), np.log10([ys[0, 1]]), "o", label="Start"
        )  # start
        plt.plot(np.log10([ys[-1, 0]]), np.log10([ys[-1, 1]]), ">", label="End")  # end
    else:
        plt.plot(ys[:, 0], ys[:, 1], "b-")  # path
        plt.plot([ys[0, 0]], [ys[0, 1]], "bo", label="Start")  # start
        plt.plot([ys[-1, 0]], [ys[-1, 1]], "bx", label="End")  # end
    # print(ys)


plt.xlabel("$H$")
plt.ylabel("$I$")
plt.legend()
# plt.xlim([-2, 8])
# plt.ylim([-4, 4])
plt.show()

