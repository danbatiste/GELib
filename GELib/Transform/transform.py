import numpy as np
import scipy.ndimage as ndimage
import cv2





def time_transform_func(t):
    def f(coords):
        x, y, channel = coords
        return (100, 100, channel)
    return f

def transform_func(coords):
    y, x, channel = coords
    center = (200, 200)
    x0, y0 = center
    center_dist = np.sqrt((y - y0)**2 + (x - x0)**2)
    relative_coords = (y - y0, x - x0)
    circle_radius = (relative_coords[0]**2 + relative_coords[1]**2)/(200*200)
    if circle_radius == 0: circle_radius = 0.01
    intensity = 2
    delta = circle_radius*intensity
    return (y*delta + y0, x*delta + x0, channel)

def mercator(coords):
    y, x, channel = coords
    center = (200, 200)
    y0, x0 = (center[0] - y, center[1] - x)
    R = 200
    longitude = x / R
    latitude = 2 * np.arctan(np.exp(y/R)) - np.pi/2

    # Get the x, y, z of the sphere in 3D space
    S = 120
    sx = S * np.cos(latitude) * np.cos(longitude)
    sy = S * np.cos(latitude) * np.sin(longitude)
    sz = S * np.sin(latitude)

    return (sx, sy, channel)

def zmap(coords):
    y, x, channel = coords
    center = (200, 200)
    y0, x0 = (center[0] - y, center[1] - x)
    z0 = (y0**2 + x0**2)/(200*200)

    scale = np.array([[1, 0], [0, 0], [0, 1]])
    offset = np.array([0, 0])
    point = np.array([x0, y0, z0])

    output = np.matmul(scale, point)# + offset
    x1, y1 = output.reshape((2,))

    return (y1, x1, channel)



def main():
    img_path = "GELib/Transform/test_image.jpg"
    mode="constant"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (400, 400))
    #cv2.imshow("Window", img)
    #cv2.waitKey(0)

    img = ndimage.geometric_transform(img, zmap, mode=mode)

    cv2.imshow("Window", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()