import numpy as np

def always_pass():
    a = 0
    b = 1
    return np.array([
        [a, b, a],
        [b, b, b],
        [a, b, a]
    ])/(255*3)