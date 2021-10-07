import numpy as np

def img_integral(img, slice=0):
    img_integral_x = np.sum(np.sum(img, axis=2), axis=0)
    img_integral_y = np.sum(np.sum(img, axis=2), axis=1)
    return [img_integral_x, img_integral_y][slice]

