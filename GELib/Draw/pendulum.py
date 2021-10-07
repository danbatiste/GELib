from objects import *
import cv2
from PIL import Image
import numpy as np


class Pendulum():
    def __init__(self, base_x, base_y, th0, l):
        self.g = 9.8
        self.th0 = th0
        self.th = th0
        self.l = l
        self.v = 0
        self.a = 0
        self.base_x = base_x
        self.base_y = base_y

    def update(self):
        self.a = -self.g*np.sin(self.th)/self.l
        self.v = self.v + self.a #np.sqrt(self.l/(2*self.g))/np.sqrt(np.cos(self.th) - np.cos(self.th0))
        self.th = self.th + self.v
        self.x = self.base_x + self.l*np.cos(np.pi/2 + self.th)
        self.y = self.base_y + self.l*np.sin(np.pi/2 + self.th)




def main():
    canvas = Canvas(400, 400)
    t = 0
    pendulum = Pendulum(200, 200, np.pi/3, 50)

    while 1:
        t += 1
        pendulum.update()
        objects = [
            Circle(pendulum.x, pendulum.y, -1, [255, 255, 0], 4),
        ]
        objects = objects
        pendulum_group = Group(objects)
        img = canvas.render(pendulum_group)
        img = cv2.line(img, (200, 200), (int(pendulum.x), int(pendulum.y)), [255, 255, 255])

        cv2.imshow("img", img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


if __name__ == "__main__":
    main()