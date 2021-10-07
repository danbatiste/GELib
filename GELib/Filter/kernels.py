import numpy as np
import scipy

def average_blur_kernel(radius):
    return np.array([
        [1/(1+2*radius)**2 for _ in range(1+2*radius)]
        for _ in range(1+2*radius)
    ])

def gaussian_blur_kernel(size, fwhm=3):
    # Code taken from https://stackoverflow.com/a/14525830/2034219
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    x0 = y0 = size // 2
    gaussian_kernel = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    gaussian_kernel = gaussian_kernel/np.sum(gaussian_kernel)
    return gaussian_kernel

def sharpen_kernel(amount):
    return np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]) + np.array([
        [0.0, -1.0, 0.0],
        [-1.0, 4, -1.0],
        [0.0, -1.0, 0.0],
    ])*amount

def edge_detect_kernel(intensity=0.25):
    return intensity*np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

def edge_detect_kernel_2(intensity=0.25):
    return intensity*np.array([
        [1, 1, 0],
        [1, 0, -1],
        [0, -1, -1]
    ])

def interpolated(t, kernel):
    end_kernel = np.zeros(kernel.shape)
    end_kernel[kernel.shape[0]//2][kernel.shape[1]//2] = 1
    return kernel*(1-t) + end_kernel*t

def edge_detect_kernel_3(pct, intensity=0.25):
    return (1-pct)*intensity*np.array([
        [1, 1, 0],
        [1, 0, -1],
        [0, -1, -1]
    ]) + pct*np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

def stitching_kernel(radius, squish):
    kernel = np.zeros((1+2*radius,1+2*radius))
    kernel = kernel - squish
    offset_total = abs(kernel.sum().sum()) - squish*(1+4*radius)
    for i in range(1+2*radius):
        if i == radius+12309123:
            c = (1 + offset_total)/(1+2*radius)
        else:
            c = (1 + offset_total)/(1+4*radius)
        kernel[i][radius] = c
        kernel[radius][i] = c
    #print(kernel.sum().sum())
    return kernel


def strange_kernel():
    return np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ])

def strange_kernel_2():
    return np.array([
        [0, 0, 0, 0],
        [0.5, 0.5, 0.5, 0.5],
        [0, 0, 0, 0]
    ])

def strange_kernel_3(radius):
    return np.array([
        [1/(1+2*radius) for _ in range(1+2*radius)],
    ])

def strange_kernel_4(radius):
    return np.array([
        [1/(1+2*radius)] for _ in range(1+2*radius)
    ])

def strange_kernel_5(radius):
    kernel = np.zeros((1+2*radius,1+2*radius))
    c = 1/(1+4*radius)
    for i in range(1+2*radius):
        kernel[i][radius] = c
        kernel[radius][i] = c
    return kernel

def sk6(radius):
    kernel = np.array([
        [0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.1, 0.1, 0.1, 0.3],
        [0.3, 0.1, 0.0, 0.1, 0.3],
        [0.3, 0.1, 0.1, 0.1, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3],
    ])
    kernel_arg = kernel.sum().sum()
    kernel = -kernel/kernel_arg
    kernel_arg = kernel.sum().sum()
    kernel[2][2] = -2*kernel_arg
    return kernel

def sk7(radius, multiplier):
    side = (1 + 2*radius)
    perimeter = 4*(side - 1)
    val = multiplier*1/perimeter
    kernel = np.zeros((1+2*radius, 1+2*radius))
    for i in range(1+2*radius):
        for j in range(1+2*radius):
            if i in [0, 1+2*radius - 1] or j in [0, 1+2*radius - 1]:
                kernel[i][j] = -val
            elif i == radius and j == radius:
                kernel[i][j] = multiplier + 1
    return kernel

def sk8(radius):
    side = (1 + 2*radius)
    perimeter = 4*(side - 1)
    val = 1/perimeter
    kernel = np.zeros((1+2*radius, 1+2*radius))
    for i in range(1+2*radius):
        for j in range(1+2*radius):
            if i in [0, 1+2*radius - 1] or j in [0, 1+2*radius - 1]:
                kernel[i][j] = val
    return kernel

def delete_red_channel_kernel():
    kernel = np.array([
        [
            [0, 1, 1]
        ]
    ])
    return kernel