from math import ceil
from cmath import *
import numpy as np
import cupy as cp
import pstats
import math
import time
import os
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

from GELib.GPU.gpu_functions import gpu_free_memory
from GELib.Filter import filters, kernels # Required
from GELib.Fractal.fractal_engine import *



fract_file_name = "bug_in_code.fract"
file_path = os.path.join("fract_files", fract_file_name)
# Runs the code in the fract file
# Not safe for web application
with open(file_path, 'rb') as fract_file:
    exec(fract_file.read())

## Frames and fps will be initialized after the fract file is run
testanim = Animation(frames, fps=fps, file_path=file_path)
testanim.stream()#.animate()#.stream() #There is also .animate
_="""
s = io.StringIO()
pr = cProfile.Profile()
pr.enable()
#create_jpg_pixels(corners, image_resolution, fractal_function, color_function, iterations, t, gpu_render=gpu_render)
frame = frames[1500]
frame.show_image()
#frame.render_image(f"img{round(time.time())}.jpg")
gpu_free_memory()
pr.disable
ps = pstats.Stats(pr, stream=s)
stats = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
stats.print_stats(0.01)
#"""

_="""
s = io.StringIO()
pr = cProfile.Profile()
pr.enable()
testanim.stream()
pr.disable
ps = pstats.Stats(pr, stream=s)
stats = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
stats.print_stats(0.06)
#"""