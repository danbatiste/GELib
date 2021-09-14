import numpy as np


class Canvas():
    def __init__(self, width, height):
        self.height = height
        self.width = width

    def render(self, group):
        canvas = -np.ones((self.width, self.height, 3))
        for object in sorted(group.objects, key=lambda obj: obj.z):
            object_image = object.draw()
            obj_width, obj_height, _ = object_image.shape
            obj_center = [obj_width//2, obj_height//2]
            for i, row in enumerate(object_image):
                offset_x = object.x - obj_center[0]
                offset_y = object.y - obj_center[1]
                image_coord_x = offset_x + i
                if not (0 <= image_coord_x < self.width - 1):
                    continue
                for j, pixel_color in enumerate(row):
                    if -1 in pixel_color:
                        continue
                    image_coord_y = offset_y + j
                    if not (0 <= image_coord_y < self.height - 1):
                        continue
                    canvas[image_coord_y][image_coord_x] = pixel_color
        return canvas


class Group():
    def __init__(self, objects):
        self.objects = objects

    def get_corners(self):
        return np.array([
            [0, 0],
            [400, 400]
        ]) # Hardcoded but later find these w math by going over all objects

    def draw(self):
        corners = self.get_corners()
        self.width = corners[1][0] - corners[0][0]
        self.height = corners[1][1] - corners[0][1]
        canvas = -np.ones((self.width, self.height, 3)) # Why does this run?
        for object in sorted(self.objects, key=lambda obj: obj.z):
            object_image = object.draw()
            obj_width, obj_height, _ = object_image.shape
            obj_center = [obj_width//2, obj_height//2]
            for i, row in enumerate(object_image):
                offset_x = object.x - obj_center[0]
                offset_y = object.y - obj_center[1]
                image_coord_x = offset_x + i
                if not (0 <= image_coord_x < self.width - 1):
                    continue
                for j, pixel_color in enumerate(row):
                    image_coord_y = offset_y + j
                    if not (0 <= image_coord_y < self.height - 1):
                        continue
                    canvas[image_coord_y][image_coord_x] = pixel_color
        return canvas

    def bake(self, x, y, z):
        image = self.render()
        return Baked_Object(image, x, y, z)


class Object():
    def __init__(self, x, y, z, color):
        self.x = x
        self.y = y
        self.z = z
        self.color = np.array(color[::-1])/255 #BGR to RGB to float

    def bake(self):
        image = self.draw()
        return Baked_Object(image, self.x, self.y, self.z)





class Circle(Object):
    def __init__(self, x, y, z, color, radius):
        super().__init__(x, y, z, color)
        self.radius = radius

    def draw(self):
        obj_image = -np.ones((self.radius*2, self.radius*2, 3))
        for x in range(-self.radius, self.radius):
            for y in range(-self.radius, self.radius):
                if (x**2 + y**2)**(1/2) <= self.radius:
                    j, i = y + self.radius, x + self.radius
                    obj_image[j][i] = self.color

        return obj_image

  
class Square(Object):
    def __init__(self, x, y, z, color, size):
        super().__init__(x, y, z, color)
        self.size = size

    def draw(self):
        return np.array([
            [self.color for _ in range(self.size)] for _ in range(self.size)
        ])


class Baked_Object(Object):
    """An image. A pixel matrix."""
    def __init__(self, image, x, y, z):
        super().__init__(x, y, z, [-1, -1, -1])
        self.image = image

    def draw(self):
        return np.array(self.image)






# TODO
# DONE Baked object (turns a group into an image)
# Bezier curve object
# Line/rect object
# Arc object
# Object that is a collection of objects (like a folder, or like a group in C4D)
# Then transformations can be applied to that folder