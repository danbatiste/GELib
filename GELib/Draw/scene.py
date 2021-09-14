import cv2
from PIL import Image
import numpy as np
from objects import *
import time

def main():
    t = 0

    canvas = Canvas(400, 400)
    while 1:
        t += 1
        objects = [
            Square(300, 100, 1, [0, 0, 0], 20),
            Circle(200, 200, -1, [255, 255, 0], 200),
            Square(0, 0, 10, [0, 0, 0], 400),
            Square(200, 200, -11, [255, 255, 255], 400),
        ]
        objects = objects

        pacman = Group(objects)
        pacman_image = Group([Baked_Object(pacman.draw(), 300, 300, 1)])
        img = canvas.render(pacman_image)

        cv2.imshow("img", img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


if __name__ == "__main__":
    main()