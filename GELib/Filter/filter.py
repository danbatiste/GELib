import numpy as np
from PIL import Image
from copy import copy
import time
import kernels

def get_neighbor_range(x, y, height, width, radius):
    neighbors = np.array([
                [x+a, y+b] if (0 <= x+a < width-1) and (0 <= y+b < height-1) else [-1, -1]
                for a in range(-radius, radius+1)
                for b in range(-radius, radius+1)
            ])
    xmin = x - radius if x - radius >= 0 else 0
    xmax = x + radius if x + radius < width - 1 else width
    ymin = y - radius if y - radius >= 0 else 0
    ymax = y + radius if y + radius < height - 1 else height
    neighbor_range = [[xmin, ymin], [xmax, ymax]]
    return neighbor_range


def convolve(image, kernel, verbose=False):
    height, width = len(image), len(image[0])
    kernel_width, kernel_height = len(kernel), len(kernel[0])
    kernel_result_pixel = [kernel_width//2, kernel_height//2]

    new_image = copy(image)
    last_percent = 0
    increment = 0.05
    if verbose: print("Kernel sum: ", kernel.sum().sum())
    for j in range(height):
        # Progress bar type thing
        cur_percent = increment*(((100/increment)*j/height)//1)
        if cur_percent > last_percent and verbose:
            print(f"{cur_percent:.2f}%")
            last_percent = cur_percent
        
        for i in range(width):
            result_pixel = np.array([0, 0, 0], dtype=float)
            for y, kernel_row in enumerate(kernel):
                for x, weight in enumerate(kernel_row):
                    offset_x = x - kernel_result_pixel[0]
                    offset_y = y - kernel_result_pixel[1]
                    
                    image_coord_x = offset_x + i
                    image_coord_y = offset_y + j

                    if not ((0 <= image_coord_x < width - 1) and (0 <= image_coord_y < height - 1)):
                        continue
                    
                    gross_pixel = weight*image[image_coord_y][image_coord_x]
                    result_pixel += gross_pixel
            
            result_pixel = np.array([value if value >= 0 else 0 for value in result_pixel], dtype=float)
            result_pixel = np.array([value if value <= 255 else 255 for value in result_pixel], dtype=float)
            new_image[j][i] = result_pixel
    return new_image


if __name__ == "__main__":
    start = time.time()
    kernel = kernels.stitching_kernel(10, 0.01)
    print(kernel)
    image = np.asarray(Image.open("Filters/earth.png", mode="RGB"))
    image2 = convolve(image, kernel, verbose=True)
    print("Shape:", image2.shape)
    name = str(time.time())[:10]
    Image.fromarray(image2).save(f"Filters/results/{name}.jpg")
    print(f"TOTAL MINUTES: {(time.time() - start)/60}")
    Image.fromarray(image2).show()

    """
    frames = 120
    fps = 60
    try: os.mkdir("animations/TREE_EDGE_DETECTION")
    except: pass
    ts = np.linspace(0,1,frames)
    
    for frame, t in enumerate(ts):
        kernel = edge_detect_kernel(intensity=0.3 + 3.0*t)
        out_image = convolve(image, kernel)
        Image.fromarray(out_image).save("animations/TREE_EDGE_DETECTION/frame" + str(frame).zfill(3) + ".jpg")
        print(f"Finished frame {frame}")

    command = f"ffmpeg -framerate {fps} -start_number 0000 -i frame%03d.jpg TREE_EDGE_DETECTION.mp4"
    os.system(command)
    """