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

from ..GPU import gpu_free_memory
from .fractal_color_functions import *
from .fractal_functions import *


def create_jpg_pixels(corners, detail_resolution,
                      fractal_function, color_function, iterations, t,
                      gpu_render, fractal_function_input_type, fractal_function_output_type):
    """Lower left corner to upper right corner"""
    x0, y0 = corners[0]
    x1, y1 = corners[1]
    
    detail_width, detail_height = detail_resolution

    if gpu_render:
        # Doing the actual math on the GPU
        # N = 1000 # Hardcoded for now
        # M = resolution//1000 #(resolution must be a multiple of 500)
        # the total thread count should be resolution**2 (GRID_SIZE*BLOCK_SIZE)
        THREADS_PER_BLOCK = 700
        GRID_SIZE = ((detail_width*detail_height)//THREADS_PER_BLOCK,)
        BLOCK_SIZE = (THREADS_PER_BLOCK,)
        x = np.linspace(x0, x1, detail_width, dtype=fractal_function_input_type)
        y = 1j*np.linspace(y0, y1, detail_height, dtype=fractal_function_input_type)
        x, y = cp.meshgrid(x, y)
        xyplane = x + y
        
        # Initialize output matrix, then calculate fractal values
        output_fractal_values = cp.zeros((detail_width*detail_height,), dtype=fractal_function_output_type).reshape(detail_height, detail_width)   # Can this just be one object, the xyplane? Or do I have to use a separate array for output...
        fractal_function(GRID_SIZE, BLOCK_SIZE, (xyplane, output_fractal_values, t, iterations, GRID_SIZE[0]*BLOCK_SIZE[0]))#detail_width*detail_height)) # The kernel

        # Initialize pixel matrix, then calculate pixel values
        output_rgbx_ints = cp.zeros((detail_width*detail_height,), dtype=cp.int32).reshape(detail_height, detail_width)
        color_function(GRID_SIZE, BLOCK_SIZE, (output_fractal_values, output_rgbx_ints, t,  GRID_SIZE[0]*BLOCK_SIZE[0]))#detail_width*detail_height))
        
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
        x, y = np.mgrid[x0:x1:detail_width*1j,y0:y1:detail_height*1j]
        # Get the values from the fractal function
        return color_function(fractal_function(x, y, iterations, t), t)


class Frame():
    def __init__(self, corners, detail_resolution, image_width, image_height,
                 fractal_function, color_function, iterations, t,
                 filters_to_apply=[], # kwarg so as to not break older .fract files
                 gpu_render=True, fractal_function_input_type=cp.complex64,
                 fractal_function_output_type=cp.int32):
        # Args
        self.fractal_function = fractal_function
        self.detail_resolution = detail_resolution
        self.corners = corners
        self.iterations = iterations
        self.color_function = color_function
        self.image_height = image_height
        self.image_width = image_width
        self.t = t

        # KWArgs
        self.filters_to_apply = filters_to_apply
        self.gpu_render = gpu_render
        self.fractal_function_input_type = fractal_function_input_type
        self.fractal_function_output_type = fractal_function_output_type

    def show_image(self, mode="RGBX"):
        print("Rendering...")
        img = self.return_image(mode=mode)
        img.show() # Maybe replace with cv2.imshow()?
        
    def render_image(self, filename, mode="RGBX"):
        img = self.return_image(mode=mode)
        img.save(filename)

    def return_image(self, mode="RGBX"):
        # Get pixels
        rgbx_pixels = create_jpg_pixels(self.corners, self.detail_resolution, 
                                        self.fractal_function, self.color_function, self.iterations, self.t,
                                        self.gpu_render, self.fractal_function_input_type, self.fractal_function_output_type)
        if self.gpu_render:
            rgbx_pixels = cp.asnumpy(rgbx_pixels)
        ImgRGBX = np.asarray(Image.fromarray(rgbx_pixels, mode=mode).convert("RGB"))
        rgbx_pixels = None # Free memory
        
        # Apply filters (CPU)
        for image_filter in self.filters_to_apply:
            ImgRGBX = image_filter(ImgRGBX)
        
        # Denoising
        #img = cv2.fastNlMeansDenoisingColored(ImgRGBx, None, 20, 3, 7, 11)
        #img = Image.fromarray(img, mode="RGB")

        # Anti aliasing
        img = Image.fromarray(ImgRGBX, mode="RGB")
        img = img.resize((self.image_width, self.image_height), resample=Image.LANCZOS)
        return img
    

    
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
        render_start_time = time.time()
        for i in range(total_frames):
            # ETA display & occasional memory clearing
            if i <= 10 or i%10 == 0:
                if i == 0:
                    est_remaining_time = "---:--:--"
                else:
                    total_seconds = time.time() - render_start_time
                    seconds_per_frame = total_seconds/i
                    remaining_frames = total_frames - i
                    est_remaining_time_seconds = str(int((seconds_per_frame*remaining_frames) % 60)).zfill(2)
                    est_remaining_time_mins = str(int(((seconds_per_frame*remaining_frames))%3600//60)).zfill(2)
                    est_remaining_time_hrs = str(int(seconds_per_frame*remaining_frames//3600)).zfill(3)
                    est_remaining_time = f"{est_remaining_time_hrs}:{est_remaining_time_mins}:{est_remaining_time_seconds}"
                print(f"Rendering frame {i}".ljust(20), f"ETA: {est_remaining_time}".ljust(20))
            if i%100 == 0:
                gpu_free_memory()
            # Render the frame
            frame = self.frames[i]
            filename = "frame" + str(i).zfill(len(str(total_frames))) + f".{self.image_extension}"
            path = os.path.join(frames_folder, filename)
            frame.render_image(path)

        # Finishing up
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
