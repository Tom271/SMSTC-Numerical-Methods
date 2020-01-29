import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
alpha = 2
beta = 1
T = 1
N = 200
dt = T / N
Xzero = 1
Xzero2 = np.sqrt(Xzero)

Dt = dt
Xem1 = np.zeros(N)
Xem2 = np.zeros(N)
Xtemp1 = Xzero
Xtemp2 = Xzero2

for j in range(N):
    Winc = np.sqrt(dt) * np.random.normal()
    f1 = alpha - Xtemp1
    g1 = beta * np.sqrt(abs(Xtemp1))

    Xtemp1 += Dt * f1 + Winc * g1
    Xem1[j] = Xtemp1

    f2 = (4 * alpha - beta ** 2) / (8 * Xtemp2) - Xtemp2 / 2
    g2 = beta / 2
    Xtemp2 += Dt * f2 + Winc * g2
    Xem2[j] = Xtemp2

plt.plot(np.arange(0, T, Dt), np.sqrt(Xem1), "b-", label="Direct Solution")
plt.plot(np.arange(0, T, Dt), Xem2, "ro", label="Solution via Chain Rule")
plt.xlabel("t")
plt.ylabel("V(X)")
plt.show()

Xdiff = np.linalg.norm(np.sqrt(Xem1) - Xem2, np.inf)
print("Difference is {}".format(Xdiff))
