from copy import copy, deepcopy
def shift(object, dx, dy, inplace=False):
    if not inplace:
        object = copy(object)
    object.x += dx
    object.y += dy
    return object