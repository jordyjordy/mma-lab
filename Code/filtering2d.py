import cv2 # this imports the OpenCV functions
import sys
import numpy as np

im = cv2.imread('../Images/bookshelf.jpg')
cv2.imshow('non dilated', im)
cv2.waitKey()
hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
kernel = np.ones((5,5))
im_dil = cv2.dilate(im,kernel,5)
im = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imshow('dilated', im_dil)
cv2.waitKey()