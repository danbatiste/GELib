from fourier import convolve, fourier, inverse_fourier
import numpy as np
from cmath import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    a, b, = -10, 10
    ts = np.linspace(a, b, 1000)
    t2s = np.linspace(-1, 1, 1000)
    f1 = np.frompyfunc(lambda t: 1 if (-1 < t < 1) else 0, 1, 1)
    f2 = np.frompyfunc(lambda t: (1 - (t+1)/2) if (-1 < t < 1) else 0, 1, 1)
    f1 = f1(ts)
    f2 = f2(t2s)
    f3 = np.array([convolve(f1, f2, a, b, x) for x in ts])
    """
    a, b, = -10, 10
    ts = np.linspace(a, b, 2000)
    f1 = np.frompyfunc(lambda t: 1 if (-2 < t < -1.98) or (5.00 < t < 5.02) else 0, 1, 1)
    f2 = np.frompyfunc(lambda t: (1 - (t+1)/2) if (-1 < t < 1) else 0, 1, 1)
    f3 = np.array([convolve(f1, f2, a, b, x) for x in ts])

    
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(ts, f1(ts))
    ax[1].plot(ts, f2(ts))
    ax[2].plot(ts, f3)
    plt.show()