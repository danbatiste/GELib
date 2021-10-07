from GELib.Fractal.Chaos_game.chaos_engine import *
from GELib.Fractal.Chaos_game.transforms import *






start_vertices = [
    np.array([-1, -1]),
    np.array([1, -1]),
    np.array([0, 1])
] # Triangle

transforms = [
    (avg, [])
]
cfractal = CFractal(start_vertices, )