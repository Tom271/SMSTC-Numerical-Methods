import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
T = 1
N = 500
dt = T / N

dW = np.sqrt(dt) * np.random.normal(size=N)
W = np.cumsum(dW)

plt.plot(np.arange(0, T + dt, dt), np.concatenate([[0], W]), "r-")
plt.xlabel("t")
plt.ylabel("W(t)")
plt.show()
