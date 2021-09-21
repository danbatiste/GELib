from cmath import *
import numpy as np
import cupy as cp
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


def discrete_fourier_3d(points, n, m):
    """Gets the constant cn,m of f for the given n"""
    steps = len(points)
    delta = 1/steps
    c_nm = 0
    for i in range(steps):
        t = 0 + i*delta
        c_nm += delta*points[i]*e**(-n*2*pi*complex(0,1)*t)
    return c_nm

def fourier(points, n):
    """Gets the constant cn of f for the given n"""
    steps = len(points)
    delta = 1/steps
    c_n = 0
    points = cp.array(points, dtype=cp.complex64)
    i = cp.array(range(steps))
    t = i*delta
    c_n = delta*points* e**(-n*2*pi*complex(0,1)*t)
    c_n = cp.sum(c_n)
    return cp.asnumpy(c_n)

def dft_2d(image, k, l): # Red channel
    M, N = image.shape
    IMAGE_kl = 0
    image = cp.array(image)
    ns = cp.arange(0, N)
    ms = cp.arange(0, M)
    ns, ms = cp.meshgrid(ns, ms)
    xs = image*cp.exp(-complex(0, 1)*2*pi*(k*ms/M + l*ns/N))
    #fsum = np.frompyfunc(lambda n, m: image[n][m]*np.exp(const*(k*m/M + l*n/N)), 2, 1)
    #fsum(ns, ms)

    IMAGE_kl = cp.sum(cp.sum(xs))/(M*N)
    #for m in range(M):
    #    for n in range(N):
    #        x = const*(k*m/M + l*n/N)
    #        IMAGE_kl += image[m, n]*exp(x)
    return IMAGE_kl

def invdft_2d(constants, m, n, M, N):
    K, L = constants.shape
    
    ks = np.arange(0, K)
    ls = np.arange(0, L)
    ks, ls = np.meshgrid(ks, ls)
    const = complex(0, 1)*2*pi
    xs = const*(ks*m/M + ls*n/N)
    IMAGE_kl = np.sum(xs)
    #for m in range(M):
    #    for n in range(N):
    #        x = const*(k*m/M + l*n/N)
    #        IMAGE_kl += image[m, n]*exp(x)
    return IMAGE_kl


if __name__ == "__main__":
    image = np.ones((1000, 1000, 3))[:,:,0]
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])/9
    K, L = image.shape
    IMAGE_DFT = np.fft.fft2(image)
    #cp.array([[dft_2d(image, k, l)
    #                for k in range(K)]
    #                for l in range(L)])
    KERNEL_DFT = np.fft.fft2(kernel)
    #cp.array([[dft_2d(kernel, k, l)
    #                for k in range(K)]
    #                for l in range(L)])
    RESULT_DFT = np.multiply(IMAGE_DFT,KERNEL_DFT)
    print(RESULT_DFT)

    RESULT = invdft_2d(RESULT_DFT)
    print(RESULT)