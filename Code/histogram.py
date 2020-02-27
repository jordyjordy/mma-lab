import cv2 # this imports the OpenCV functions
import sys
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('../Images/theDress.jpeg')

def my_colorHist(im,num_bins):
    hist= np.zeros((3,num_bins))

    # hist1 = cv2.calcHist([im],[0],None,[num_bins],[0,256]).reshape(256,)
    # hist2 = cv2.calcHist([im],[1],None,[num_bins],[0,256]).reshape(256,)
    # hist3 = cv2.calcHist([im],[2],None,[num_bins],[0,256]).reshape(256,)

    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            hist[0, im[y, x, 0]*num_bins/256] += 1
            hist[1, im[y, x, 1]*num_bins/256] += 1
            hist[2, im[y, x, 2]*num_bins/256] += 1

    x = np.arange(num_bins)
    width = 1.0/3
    plt.bar(x-1.0/4,hist[0,:],width,color='blue')
    plt.bar(x, hist[1,:], width, color='green')
    plt.bar(x+1.0/4, hist[2,:], width, color='red')
    plt.show()
    return hist



hist = my_colorHist(im,256)

