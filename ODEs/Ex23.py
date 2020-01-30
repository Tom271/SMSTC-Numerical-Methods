import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt


def chemode(t, y):
    f = np.zeros(3)
    f[0] = -0.04 + y[0] + 1e4 * y[1] * y[2]
    f[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2
    f[2] = 3e7 * y[1] ** 2
    return f


def chemjac(t, y):
    dfdy = np.zeros((3, 3))
    dfdy[0, :] = [-0.04, 1e4 * y[2], 1e4 * y[1]]
    dfdy[1, :] = [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]]
    dfdy[2, :] = [0, 6e7 * y[1], 0]
    return dfdy


y0 = [1, 0, 0]
t0 = 0
r = ode(chemode, chemjac).set_integrator(
    "vode", with_jacobian=True, rtol=1e-6, atol=1e-10
)
r.set_initial_value(y0, t0)
T_end = 5
dt = 0.01
t, sol = [], []

while r.successful() and r.t < T_end:
    t.append(r.t)
    sol.append(r.y)
    r.integrate(r.t + dt)

chem1, chem2, chem3 = np.array(sol).T
# plt.plot(t, chem1, label="Chemical 1")
plt.plot(t, chem2, label="Chemical 2")
# plt.plot(t, chem3, label="Chemical 3")

plt.legend()
plt.show()
