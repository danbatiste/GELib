import numpy as np

def average_blur_kernel(radius):
    return np.array([
        [1/(1+2*radius)**2 for _ in range(1+2*radius)]
        for _ in range(1+2*radius)
    ])

def sharpen_kernel():
    return np.array([
        [-0.5, -1.0, -0.5],
        [-1.0, 7.0, -1.0],
        [-0.5, -1.0, -0.5],
    ])

def edge_detect_kernel(intensity=0.25):
    return intensity*np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
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
    print(kernel.sum().sum())
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