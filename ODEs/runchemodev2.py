from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from oderk4 import oderk4


def chemode(t, u, pars):
    dudt = np.zeros(3)
    dudt[0] = pars[0] * (u[0] - pars[1] * u[0] ** 2 + u[1] - u[0] * u[1])
    dudt[1] = (-u[1] - u[0] * u[1] + u[2]) / pars[0]
    dudt[2] = pars[2] * (u[0] - u[2])

    return dudt


def chemjac(t, u):
    dfdu = np.zeros((3, 3))
    dfdu[0, :] = [-0.04, 1e4 * u[2], 1e4 * u[1]]
    dfdu[1, :] = [0.04, -1e4 * u[2] - 6e7 * u[1], -1e4 * u[1]]
    dfdu[2, :] = [0, 6e7 * u[1], 0]

    return dfdu


u0 = [30, 1, 30]
pars = (
    120,
    1.5e-4,
    0.1,
)

abs_tol = 1e-12
rel_tol = 1e-5

start_45 = time()
sol_45 = solve_ivp(
    chemode, [0, 2], u0, method="RK45", args=[pars], atol=abs_tol, rtol=rel_tol
)
solve_time_45 = time() - start_45

start_stiff = time()
sol_stiff = solve_ivp(
    chemode, [0, 2], u0, method="Radau", args=[pars], atol=abs_tol, rtol=rel_tol
)
solve_time_stiff = time() - start_stiff

start_rk4 = time()
t_rk4, sol_rk4 = oderk4(lambda t, u: chemode(t, u, pars), [0, 2], u0, 65)
solve_time_rk4 = time() - start_rk4

print(
    "CPU time {:>10} {:>10} {:>10}\n {:>18.4f}  {:>9.4f}  {:>9.4f} ".format(
        "Radau", "RK45", "oderk4", solve_time_stiff, solve_time_45, solve_time_rk4
    )
)
fig, ax = plt.subplots(3)
for i in range(3):
    ax[i].plot(sol_stiff.t, sol_stiff.y[i, :], "b-")
    ax[i].plot(sol_45.t, sol_45.y[i, :], "r--")
    ax[i].plot(t_rk4, sol_rk4[i, :], "k:")
    ax[i].set_ylabel("Component {}".format(i + 1))
ax[0].legend(["RK45", "Stiff", "RK4"])
plt.tight_layout()
plt.show()
