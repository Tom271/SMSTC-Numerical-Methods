import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
lamda = 2
mu = 1
Xzero = 1
T = 1
N = 2 ** 8
dt = T / N
dW = np.sqrt(dt) * np.random.normal(size=N)
W = np.cumsum(dW)

Xtrue = Xzero * np.exp((lamda - 0.5 * mu ** 2) * (np.arange(dt, T + dt, dt)) + mu * W)

plt.plot(np.arange(dt, T + dt, dt), Xtrue, "m-")

R = 4
Dt = R * dt
L = N // R
Xem = np.zeros(L)
Xtemp = Xzero
for j in range(L):
    Winc = np.sum(dW[R * (j - 1) + 1 : R * j])
    Xtemp += Dt * lamda * Xtemp + mu * Xtemp * Winc
    Xem[j] = Xtemp


plt.plot(np.arange(Dt, T + Dt, Dt), Xem, "r--*")
plt.xlabel("t")
plt.ylabel("X")
plt.show()
emerr = np.abs(Xem[-1] - Xtrue[-1])

print(emerr)
