from fourier import fourier, inverse_fourier
from cmath import *
import numpy as np

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    try: frames = os.mkdir("frames")
    except: pass

    x = np.linspace(1, log(10000, 10).real)
    y = np.array((10**x)//1, dtype=int)
    output_is = y
    frames = len(output_is)
    fps = 20
    print(f"CREATING {frames} FRAMES")
    for framenumber, i in enumerate(output_is):
        # Settings
        nfactor = i
        n0, nf = np.array([0, 1])*nfactor
        a, b = -2*pi, 2*pi
        steps = 1000
        f = lambda x: cos(x).real

        # Formatting of data
        f = np.frompyfunc(f, 1, 1)
        complex_ = np.frompyfunc(complex, 2, 1)
        x = np.linspace(a, b, steps)
        y = f(x)
        points = complex_(x, y)

        # Driver code
        fourier_constants = np.array([fourier(points, n) for n in range(n0, nf)])
        original_function = inverse_fourier(fourier_constants, n0, steps=steps)
       
        # Graph code
        x2 = original_function.real #np.linspace(a, b, len(original_function))# #np.linspace(a, b, len(fourier_constants))#real(fourier_constants)
        y2 = original_function.imag
        plt.scatter(x2, y2, c='b', s=0.4)#, imag(sine_function))
        plt.scatter(x, y, c="r", s=0.2)
        plt.xlim([-10, 10])
        plt.ylim([-2, 2])
        plt.title(f"constants = {str(i).zfill(len(str(1000)))}")
        filename = "frame{}.png".format(str(framenumber).zfill(len(str(frames))))
        plt.savefig(os.path.join("frames", filename))
        plt.clf()
        print("OUTPUT: ", filename)
    command = f"ffmpeg -framerate {fps} -start_number 0000 -i frames/frame%0{len(str(frames))}d.png output.webm"
    os.system(command)