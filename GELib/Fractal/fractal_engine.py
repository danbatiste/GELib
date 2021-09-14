from PIL import Image
import cProfile, io
from cmath import *
import numpy as np
import cupy as cp
import pstats
import math
import time
import cv2
import os
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


from GELib.GPU.gpu_functions import gpu_free_memory
from GELib.Fractal.fractal_color_functions import *
from GELib.Fractal.fractal_functions import *


def create_jpg_pixels(corners, resolution,
                      fractal_function, color_function, iterations, t,
                      gpu_render, fractal_function_input_type, fractal_function_output_type):
    """Lower left corner to upper right corner"""
    x0, y0 = corners[0]
    x1, y1 = corners[1]
    
    if gpu_render:
        # Doing the actual math on the GPU
        # N = 1000 # Hardcoded for now
        # M = resolution//1000 #(resolution must be a multiple of 500)
        # the thread count should be resolution**2
        GRID_SIZE = (resolution**2//500,)
        BLOCK_SIZE = (500,)
        x = cp.linspace(x0, x1, resolution, dtype=fractal_function_input_type)
        y = 1j*cp.linspace(y0, y1, resolution, dtype=fractal_function_input_type)
        x, y = cp.meshgrid(x, y)
        xyplane = x + y
        
        # Initialize output matrix, then calculate fractal values
        output_fractal_values = cp.zeros(resolution**2, dtype=fractal_function_output_type).reshape(resolution, resolution)   # Can this just be one object, the xyplane? Or do I have to use a separate array for output...
        fractal_function(GRID_SIZE, BLOCK_SIZE, (xyplane, output_fractal_values, t, iterations, resolution**2)) # The kernel

        # Initialize pixel matrix, then calculate pixel values
        output_rgbx_ints = cp.zeros(resolution**2, dtype=cp.int32).reshape(resolution, resolution)
        color_function(GRID_SIZE, BLOCK_SIZE, (output_fractal_values, output_rgbx_ints, t, resolution**2))
        
        # Debug
        #print(np.max(output_fractal_values))
        #print(output_fractal_values)
        
        # Freeing memory
        x = None
        y = None
        xyplane = None
        output_fractal_values = None
        
        #Finally returning the output values
        return output_rgbx_ints
    else:
        #y, x = np.mgrid[y0:y1:resolution*1j,x0:x1:resolution*1j]
        x, y = np.mgrid[x0:x1:resolution*1j,y0:y1:resolution*1j]
        # Get the values from the fractal function
        return color_function(fractal_function(x, y, iterations, t), t)


class Frame():
    def __init__(self, corners, image_resolution, width, height,
                 fractal_function, color_function, iterations, t,
                 gpu_render=True, fractal_function_input_type=cp.complex64,
                 fractal_function_output_type=cp.int32):
        # Args
        self.fractal_function = fractal_function
        self.image_resolution = image_resolution
        self.corners = corners
        self.iterations = iterations
        self.color_function = color_function
        self.height = height
        self.width = width
        self.t = t
        # KWArgs
        self.gpu_render = gpu_render
        self.fractal_function_input_type = fractal_function_input_type
        self.fractal_function_output_type = fractal_function_output_type
          
    def show_image(self, mode="RGBX"):
        rgbx_pixels = create_jpg_pixels(self.corners, self.image_resolution, self.fractal_function,self.color_function, self.iterations, self.t)
        ImgRGBX = Image.fromarray(rgbx_pixels, mode=mode)
        ImgRGBX.show()
    
    def render_image(self, filename, mode="RGBX"):
        rgbx_pixels = create_jpg_pixels(self.corners, self.image_resolution, 
                                        self.fractal_function, self.color_function, self.iterations, self.t,
                                        self.gpu_render, self.fractal_function_input_type, self.fractal_function_output_type)
        #print(pixel32[:5])
        if self.gpu_render:
            rgbx_pixels = cp.asnumpy(rgbx_pixels)
        ImgRGBX = Image.fromarray(rgbx_pixels, mode=mode)
        rgbx_pixels = None # Freeing memory
        ImgRGBX = ImgRGBX.resize((self.width, self.height), resample=Image.ANTIALIAS)
        ImgRGBX.save(filename)
        #imageio.imwrite(filename, pixel32, format="jpg")
        
    def render_image_nozoom(self, filename, mode="RGBX"):
        # Supposedly more efficient, but numbers dont show it
        rgbx_pixels = create_jpg_pixels(self.corners, self.image_resolution, self.fractal_function, self.color_function, self.iterations, self.t)
        ImgRGBX = Image.fromarray(rgbx_pixels, mode=mode)
        ImgRGBX = ImgRGBX.resize((self.width, self.height), resample=Image.ANTIALIAS)
        ImgRGBX.save(filename)

    def return_image(self, mode="RGBX"):
        rgbx_pixels = create_jpg_pixels(self.corners, self.image_resolution, 
                                        self.fractal_function, self.color_function, self.iterations, self.t,
                                        self.gpu_render, self.fractal_function_input_type, self.fractal_function_output_type)
        rgbx_pixels = cp.asnumpy(rgbx_pixels)
        ImgRGBX = Image.fromarray(rgbx_pixels, mode=mode)
        ImgRGBX = ImgRGBX.resize((self.width, self.height), resample=Image.ANTIALIAS)
        return ImgRGBX
    

    
class Animation():
    def __init__(self, frames, video_extension="mp4", image_extension="jpg", fps=60, file_path=None):
        self.frames = frames
        self.video_extension = video_extension.replace(".", "")
        self.image_extension = image_extension.replace(".", "")
        self.fps = fps
        self.file_path = file_path
        self.file_name = file_path.split(os.sep)[-1]
    
    def animate(self):
        # Start the timer
        start_time = time.time()

        total_frames = len(self.frames)
        
        # Create folder for the animation
        animation_name = "ANIM" + str(int(time.time()))[-10:]
        frames_folder = os.path.join("animations", animation_name)
        try:
            os.mkdir("animations")
        except IOError:
            pass
        try:
            os.mkdir(frames_folder)
            print("Frame folder created.")
        except IOError:
            pass
        
        # Put fract file into folder
        if self.file_path != None:
            with open(self.file_path, 'r') as ifile:
                file_contents = ifile.read()
            with open(os.path.join("animations", animation_name, self.file_name), 'w') as ofile:
                ofile.write(file_contents)
        
        # Loop through all frames and render
        for i in range(total_frames):
            if i <= 10 or i%10 == 0:
                print(f"Rendering frame {i}")
            if i%100 == 0:
                gpu_free_memory()
            frame = self.frames[i]
            filename = "frame" + str(i).zfill(len(str(total_frames))) + f".{self.image_extension}"
            path = os.path.join(frames_folder, filename)
            frame.render_image(path)
        print(f"All frames rendered in {round(time.time() - start_time, 2)}s.")
        gpu_free_memory()
        # Convert to .{extension} (default is mp4) with ffmpeg
        zeroes = len(str(total_frames))
        command = f"ffmpeg -framerate {self.fps} -start_number 0000 -i frame%0{zeroes}d.{self.image_extension} {animation_name}.{self.video_extension}"
        home_dir = os.getcwd()
        os.chdir(frames_folder)
        print(os.getcwd())
        print(command)
        os.system(command)
        os.chdir(home_dir)
        #min-command = ffmpeg -i .\ANIM881930.mp4 -c:v libx264 -crf 30 2ANIM881930.mp4
        print(f"ffmpeg'd animation to a .{self.video_extension}")
        print(f"Animation finished in {round(time.time() - start_time, 2)}s.")

    def stream(self):
        total_frames = len(self.frames)
        for i in range(total_frames):
            if i%100 == 0:
                gpu_free_memory()
            frame = self.frames[i]
            img = np.asanyarray(frame.return_image())
            cv2.imshow("img", img)

            # cv2 press esc key to stop
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
