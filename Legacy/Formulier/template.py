import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import helpers.dirdict as dirdict


def templer(form):
    print(form)
    img = cv.imread(form, 0)
    if img is None:
        print('img empty')
    else:
        cut(img)


def cut(img):
    img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
    blur = cv.GaussianBlur(img, (5, 5), 2)
    ret, thresh = cv.threshold(blur, 245, 255, 0, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(thresh, mask, (50, 50), 0)
    cv.imshow(' ', thresh)
    cv.waitKey(0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    dest = 'output/'
    usedkeys = []
    harvest = []
    rect = thresh.copy()
    thresh = cv.bitwise_not(thresh)
    for cnt in contours:
        M = cv.moments(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        if w > h:
            b = w
        else:
            b = h
        if b > 64:
            crop = thresh[y + 1:y + b - 2, x + 1:x + int((b * 0.9)) - 2]
            hel = 50
            cv.rectangle(rect, (x, y), (x + b - 6, y + b - 6), (hel, 0, hel), 10)
            resize = cv.resize(crop, (64, 64))
            harvest.append(resize)
            r = str(np.random.random_integers(100000, 999999))
            while r in usedkeys:
                r = str(np.random.random_integers(100000, 999999))
            usedkeys.append(r)
            fold = GiveFolder(x, y)
            cv.imwrite(dest + fold + r + '.png', resize)
    print(len(harvest))


def GiveFolder(x, y):
    xc = 1
    if x < 420:
        xc = 0
    elif x > 840:
        xc = 2
    rows = dirdict.DirDict[xc]
    for key in rows:
        if abs(key - y) < 15:
            return dirdict.DirDict[xc][key] + '/'
    return 'unknown/'


for name in os.listdir('forms'):
    templer('forms\\' + name)

# templer('forms/afbeelding (2).tif')
