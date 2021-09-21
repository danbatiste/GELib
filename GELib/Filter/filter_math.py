import numpy as np
from copy import copy
import scipy.signal as scp


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


#def slow_convolve(image, kernel, verbose=False):
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
            result_pixel = np.array(0*copy(new_image[0][0]), dtype=float)
            for y, kernel_row in enumerate(kernel):
                for x, weight in enumerate(kernel_row):
                    offset_x = x - kernel_result_pixel[0]
                    offset_y = y - kernel_result_pixel[1]
                    
                    image_coord_x = offset_x + i
                    image_coord_y = offset_y + j

                    if not ((0 <= image_coord_x < width - 1) and (0 <= image_coord_y < height - 1)):
                        continue
                    
                    gross_pixel = weight*image[image_coord_y][image_coord_x]
                    #print(gross_pixel)
                    #print(result_pixel)
                    result_pixel += gross_pixel
            
            result_pixel = np.array([value if value >= 0 else 0 for value in result_pixel], dtype=float)
            result_pixel = np.array([value if value <= 255 else 255 for value in result_pixel], dtype=float)
            #print("NEW IMAGE:", new_image[j][i])
            #print("NEW IMAGE:", new_image)
            #print("RESULT PIXEL:", result_pixel)
            #if new_image[j][i] == 0:
            #    R, G, B = result_pixel
            #    result_pixel = int(B) << 16 | int(G) << 8 | int(R)
            new_image[j][i] = result_pixel
    return new_image

def convolve(image, kernel, verbose=False):
    # Convolve image with kernel
    image = image.astype(np.float64) # Converting dtype before convolving fixed bugs
    x, y, channels = image.shape
    for i in range(channels):
        image[:,:,i] = np.convolve(image[:,:,i].reshape((x*y,)), kernel.reshape((kernel.shape[0]*kernel.shape[1],)), mode="same").reshape((x,y))
    
    # Filter out values only in 0 <= x <= 255
    image[image<=0] = 0
    image[image>=254] = 254
    image = image.astype(np.uint8)
    return image