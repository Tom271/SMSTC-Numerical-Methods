import numpy as np


def oderk4(f, trange, y0, n):
    """ODERK4   Classical RK4 for y'=f(t,y)  where y,f are vectors

           call as
           [t,w] = oderk4(f,trange,y0,n)
           f        is the name of the function M-file for the RHS
           trange    are initial and final times (other times not used)
           y0       is the initial condition y(t0)=y0
           n        is the number of equal sized steps

           t        output time vector of length n+1
           w        output solution.  w[s,:] approx y[t[s]]
    """

    t0 = trange[0]  # initial time
    t1 = trange[-1]  # final time

    w = np.zeros((n + 1, len(y0)))  # make space
    t = np.zeros(n + 1)

    h = (t1 - t0) / n  # compute step size h

    t[0] = t0
    w[0, :] = y0  # initial condition

    for s in range(n):  # loop from t0 to t1
        k1 = h * f(t[s], w[s, :]).T
        k2 = h * f(t[s] + h / 2, w[s, :] + k1 / 2).T
        k3 = h * f(t[s] + h / 2, w[s, :] + k2 / 2).T
        k4 = h * f(t[s] + h, w[s, :] + k3).T
        w[s + 1, :] = w[s, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6  # take one step
        t[s + 1] = s * h + t0

    return t, w.T
