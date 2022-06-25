import random as rand
import numpy as np

def avg(vertices, window):
    while 1:
        vertex = rand.choice(vertices)
        rx, ry = rand.random(), rand.random()
        x, y = window[0][0]*(1-rx) + window[1][0]*(rx), window[0][1]*(1-ry) + window[1][1]*(ry), 
        point = np.array([x, y])
        if vertex != point:
            break
    return (vertex + point)/2 # Moves from the vertex to halfway to the point
