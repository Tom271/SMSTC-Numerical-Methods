import numpy as np
import matplotlib.pyplot as plt

T = 1
N = 500
dt = T / N
dW = np.zeros(N)
W = np.zeros(N)

dW[0] = np.sqrt(dt) * np.random.normal()
W[0] = dW[0]
np.random.seed(seed=100)
for j in range(1, N):
    dW[j] = np.sqrt(dt) * np.random.normal()
    W[j] = W[j - 1] + dW[j]

plt.plot(np.arange(0.0, T + dt, dt), np.concatenate([[0], W]), "r--")
plt.xlabel("t")
plt.ylabel("W(t)")

plt.show()
