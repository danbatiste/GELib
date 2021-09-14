from cmath import *
import numpy as np
import math

def integrate(f, a, b, steps=1000):
    delta = (b - a)/steps

    total = 0
    for i in range(steps):
        x0 = a + i*delta
        total += f(x0)*delta
    return total


#def fourier(f, n, a, b, steps=100):
    """Gets the constant cn of f for the given n"""
    delta = 1/steps
    integral = 0
    f_ = lambda t: f(b*(t) + a*(1-t)) # Linear interpolation from a to b
    for i in range(steps):
        t = 0 + delta*i
        integral += delta*f_(t)*e**(-n*2*pi*complex(0,1)*t)
    return integral


#def fourier(points, n):
    """Gets the constant cn of f for the given n"""
    steps = len(points)
    ts = np.linspace(0, 1, steps, dtype=np.float64)
    delta = ts[1]
    c_n = 0
    for i, t in enumerate(ts):
        if i == 0: delta = t
        else: delta = t - ts[i-1]
        c_n += delta*points[i]*e**(-n*2*pi*complex(0,1)*t)
    return c_n


def fourier(points, n):
    """Gets the constant cn of f for the given n"""
    steps = len(points)
    delta = 1/steps
    c_n = 0
    for i in range(steps):
        t = 0 + i*delta
        c_n += delta*points[i]*e**(-n*2*pi*complex(0,1)*t)
    return c_n


def inverse_fourier(constants, n0, steps=100):
    """Takes the fourier constants and makes some data with them"""
    ts = np.linspace(0, 1, steps+1, dtype=np.float64)
    f = []#np.zeros(steps, dtype=np.complex128)
    ns = np.array(range(n0, n0+len(constants)))
    for i, t in enumerate(ts, start=0):
        # Not sure why I need to divide by (len(ns)/steps) but data is bad otherwise
        f_t = constants*(e**(-ns*2*pi*complex(0, 1)*t))/(len(ns)/steps)
        f.append(sum(f_t))

    return np.array(f)


#def set_convolve_(sig1, sig2):
    steps = len(sig1)
    dt = 1/steps
    sig3 = np.zeros((steps,))
    for x in range(steps):
        total = 0
        for t in range(0, steps):
            total += sig1[t]*sig2[x-t]*dt
        sig3[x] = total
    return sig3


#def set_convolve(sig1, sig2, a, b, x):
    steps = max(len(sig1), len(sig2))
    kernel_len = min(len(sig1), len(sig2))
    ts = np.linspace(a, b, steps)
    index_x = len(np.array([t for t in ts if not t <= x])) #getindex(ts, x)
    dt = 1/steps
    total = 0
    for t in range(0, kernel_len):
        if index_x - t < 0: continue
        total += sig1[t]*sig2[index_x-t]*dt
    return total


def convolve_broken(sig1, sig2, a, b, x, steps=1000):
    ts = np.linspace(a, b, steps)
    total = 0
    dt = 1/steps
    for t in ts:
        total += sig1(t)*sig2(x - t)*dt
    return total

def convolve(sig1, sig2, a, b, x, steps=1000):
    ts = np.linspace(a, b, steps)
    dt = (b-a)/steps
    total = 0
    for t in ts:
        total += sig1(t)*sig2(x - t)*dt
    return total