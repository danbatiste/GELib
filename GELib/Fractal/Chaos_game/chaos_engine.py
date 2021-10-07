import numpy as np
import random as rand
from copy import copy

import matplotlib.pyplot as plt


class CFractal():
    def __init__(self, start_vertices, transforms,
                 window=np.array([-1, 1], [-1, 1]), detail_resolution=np.array([1000,1000])):
        self.start_vertices = start_vertices
        self.vertices = copy(start_vertices)
        self.transforms = transforms
        self.window = window
        self.width, self.height = detail_resolution

        # Statistics
        self.points_rendered = 0

    def render(self):
        for transform_func, args, kwargs in self.transforms:
            vertex = transform_func(self.vertices, self.window, *args, **kwargs)
        self.vertices.append(vertex)

        # Statistics
        self.points_rendered += 1


    def show(self):
        vertices = np.array(self.vertices)
        xs, ys = vertices[:,:,:0], vertices[:,:,:1]

        plt.plot(xs, ys)
        plt.show()
