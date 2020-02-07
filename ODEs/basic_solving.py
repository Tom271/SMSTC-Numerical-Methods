"""
This file shows how to use scipy to solve ODEs, rather than writing your own solver
"""
import numpy as np
from scipy.integrate import ode, odeint
import matplotlib.pyplot as plt

y0, t0 = 0.2, 0


def logistic(y, t):
    return y * (1 - y)


t = np.arange(0, 5 + 0.001, 0.001)
sol = odeint(logistic, y0, t, rtol=1e-4, atol=1e-4)
print(sol[t == 1])
plt.plot(t, sol, label="odeint")


def logistic(t, y):
    return y * (1 - y)


r = ode(logistic).set_integrator("vode",rtol=1e-6,atol=1e-6)
r.set_initial_value(y0, t0)
T_end = 5
dt = 0.01
t, sol = [], []

while r.successful() and r.t < T_end:
    t.append(r.t)
    sol.append(r.y)
    r.integrate(r.t + dt)


print(sol[t == 1])
plt.plot(t, sol, label="ode")
plt.legend()
plt.show()
