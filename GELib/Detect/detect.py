import numpy as np
from PIL import Image
from copy import copy
import time
import kernels


def convolve(image, kernel, verbose=False):
    height, width = len(image), len(image[0])
    kernel_width, kernel_height = len(kernel), len(kernel[0])
    kernel_result_pixel = [kernel_width//2, kernel_height//2]

    output_array = np.zeros(np.array(image).shape)
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
            

            # Code for greater than 0 and less than 255
            # result_pixel = np.array([value if value >= 0 else 0 for value in result_pixel], dtype=float)
            # result_pixel = np.array([value if value <= 255 else 255 for value in result_pixel], dtype=float)
            
            # Code for enforcing int
            result_pixel = np.array([int(value) for value in result_pixel], dtype=np.int16)
            output_array[j][i] = result_pixel
    return output_array


def main():
    kernel = kernels.always_pass()
    image = np.asarray(Image.open("GELib/Detect/plusses.tiff"))
    result = convolve(image, kernel)
    print(np.sum(result, axis=2))


if __name__ == "__main__":
    main()