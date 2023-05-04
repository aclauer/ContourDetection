import numpy as np
import cv2 as cv

im = cv.imread('0002.png')

imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

print(np.shape(contours))
print(contours)

cv.drawContours(im, contours, -1, (0,255,0), 3)

cv.namedWindow("Image Display")
cv.imshow("Image Window", im)
cv.waitKey(0)