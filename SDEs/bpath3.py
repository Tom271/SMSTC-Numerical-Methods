import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
T = 1
N = 500
dt = T / N
t = np.arange(dt, 1 + dt, dt)

M = 1000
dW = np.sqrt(dt) * np.random.normal(size=(M, N))
W = np.cumsum(dW, axis=0)

U = np.exp(np.tile(t, (M, 1)) + 0.5 * W)
Umean = np.mean(U, axis=0)
plt.plot(t, Umean, "b-", label="Mean of 1000 paths")
for _ in range(4):
    plt.plot(t, U[_, :], "r--", label="5 individual paths")
plt.xlabel("t")
plt.ylabel("U(t)")
plt.show()

averr = np.linalg.norm((Umean - np.exp(9 * t / 8)), ord=np.inf)
print("Sample Error is {}".format(averr))
