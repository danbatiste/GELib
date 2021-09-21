import numpy as np
from PIL import Image
from copy import copy
import os
import time
try:
    from . import kernels
    from . import filter_math
except:
    import kernels
    import filter_math

def custom_kernel(kernel):
    def filter_func(image):
        return filter_math.convolve(image, kernel, verbose=False)
    return filter_func

def average_blur(image):
    kernel = kernels.average_blur_kernel(2)
    return filter_math.convolve(image, kernel, verbose=False)

def stitch_20_1(image):
    kernel = kernels.stitching_kernel(20, 0.01)
    return filter_math.convolve(image, kernel, verbose=False)

def edge_detection(image):
    kernel = kernels.edge_detect_kernel(intensity=0.4)
    return filter_math.convolve(image, kernel)

def sharpen(image):
    kernel = kernels.sharpen_kernel(1)
    return filter_math.convolve(image, kernel)

if __name__ == "__main__":
    start = time.time()
    kernel = kernels.edge_detect_kernel_2(intensity=-1)
    image = np.asarray(Image.open("hdtree.jpg"))
    image = filter_math.convolve(image, kernel, verbose=True)
    name = str(time.time())[:10]
    Image.fromarray(image).save(f"results/{name}.jpg")
    print(f"TOTAL MINUTES: {(time.time() - start)/60}")
    Image.fromarray(image).show()

    """
    frames = 120
    fps = 60
    try: os.mkdir("animations/TREE_EDGE_DETECTION")
    except: pass
    ts = np.linspace(0,1,frames)
    
    for frame, t in enumerate(ts):
        kernel = kernels.stitching_kernel(int(20*t), 1*t)
        out_image = filter_math.convolve(image, kernel)
        Image.fromarray(out_image).save("animations/TREE_EDGE_DETECTION/frame" + str(frame).zfill(3) + ".jpg")
        print(f"Finished frame {frame}")

    command = f"ffmpeg -framerate {fps} -start_number 0000 -i frame%03d.jpg TREE_EDGE_DETECTION.mp4"
    os.system(command)
    #"""