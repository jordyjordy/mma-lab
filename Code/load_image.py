import cv2 # this imports the OpenCV functions
import sys
import numpy as np

def convolve(image,filter):
    fil = np.rot90(filter)
    fil = np.rot90(fil)
    fil = filter
    res = np.zeros(image.shape,dtype='uint32')
    print len(image[0]),len(image)
    for y in range (1,len(image)-1):
        for x in range(1,len(image[0])-1):
            sum = 0
            for ystep in range(-1,2):
                for xstep in range(-1,2):
                    sum += image[y+ystep][x+xstep] * fil[1+ystep][1+xstep]
            if(sum < 0): sum = 0
            if(sum > 255): sum = 255
            res[y][x] = sum
    return res

def edges(image):
    xfilter = [[-1,0,1],[-2,0,2],[-1,0,1]]
    xfilter = np.asarray(xfilter)
    yfilter = [[-1,-2,-1],[0,0,0],[1,2,1]]
    yfilter = np.asarray(yfilter)
    Gx = convolve(image,xfilter)
    Gy = convolve(image,yfilter)
    print(np.max(Gx),np.max(Gx))
    #Gx = cv2.filter2D(image,0,xfilter)
    #Gy = cv2.filter2D(image,0,yfilter)
    return np.sqrt(Gx*Gx + Gy*Gy)



#np.set_printoptions(threshold=sys.maxsize)
m_grey = np.random.rand(200,150)
im_bgr = np.random.rand(200,200,3)
m_black = np.zeros((200,100))
im = cv2.imread('../Images/bookshelf.jpg')
im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
m_edge = edges(im)
print(np.max(m_edge))
m_edge = np.clip(m_edge,0,255)
m_edge= m_edge.astype('uint8')
print(np.max(m_edge))
print(m_edge)
cv2.imshow('Black', m_edge)
cv2.waitKey()

