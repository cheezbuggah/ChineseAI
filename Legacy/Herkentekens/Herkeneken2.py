import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('voorbeeldplaatjes/papier.jpg',0)
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# plot all the images and their histograms
images = [th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
contours,_= cv.findContours(th3,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
print(len(contours))
harvest = []
for cnt in contours:
    M = cv.moments(cnt)
    x,y,w,h = cv.boundingRect(cnt)
    if w>h:
        b = w
    else:
        b = h
    if b>64:
        crop = th3[y:y+b,x:x+b]
        hel = 200
        # cv.rectangle(img,(x,y),(x+b,y+b),(hel,hel,hel),10)
        resize = cv.resize(crop,(64,64))
        harvest.append(resize)

print(len(harvest))
for i in range(int(len(harvest)/6)):
    for j in range(6):
        plt.subplot(2, 3, j + 1)
        plt.imshow(harvest[j+i*6])
    plt.show()

for i in range(3):
    plt.subplot(111),plt.imshow(images[0],'gray')
plt.show()