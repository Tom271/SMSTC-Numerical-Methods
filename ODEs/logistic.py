import numpy as np
import matplotlib.pyplot as plt
from oderk4 import oderk4


def logistic_ode(t, y):
    f = y * (1 - y)
    return f


def logistic_sol(t):
    return 1 / (1 + np.exp(-t))


t, y = oderk4(logistic_ode, [0, 10], [0.5], 1000)
plt.plot(t, y, label="odeRK4 Sol")
plt.plot(t, logistic_sol(t), 'r--', label="True Sol")
plt.legend()
plt.show()


err_fig, err_ax = plt.subplots()
err_ax.plot(t[1:], np.abs(y[1:, 0] - logistic_sol(t)[1:]), "r--", label="RK4 Error")
err_ax.legend()
plt.show()
