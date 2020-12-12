import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random as random
import cv2
import os

# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename),0)
#         if img is not None:
#             images.append(img)
#     return images
# def increase_brightness(img, value=30):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)

#     lim = 255 - value
#     v[v > lim] = 255
#     v[v <= lim] += value

#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     return img
img_rgb = cv.imread('fianl.jpeg')
# scale_percent = 90 # percent of original size
# width = int(img_rgb.shape[1] * scale_percent / 100)
# height = int(img_rgb.shape[0] * scale_percent / 100)
# dim = (width, height)
#resize image
#img_rgb = cv.resize(img_rgb, dim, interpolation = cv.INTER_AREA)
#img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('mobi.jpeg',0)
t1 = cv.imread('bottle.jpeg',0)
t2 = cv.imread('marker.jpeg',0)
t3 = cv.imread('nbook.jpeg',0)
t4 = cv.imread("book.jpeg", 0)
l = [template, t1, t2, t3, t4]
#l = load_images_from_folder('Obj/book')


for i in l:
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    rgb = (r, g, b)
    #print(i.shape[::-1])
    w, h = i.shape[::-1]
    res = cv.matchTemplate(img_gray,i,cv.TM_CCOEFF_NORMED)

    threshold = 0.8
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), rgb, 2)

cv.imwrite('res.png',img_rgb)
