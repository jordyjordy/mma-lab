import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

sift = cv2.xfeatures2d.SIFT_create()
im1 = cv2.imread('../../Images/nieuwekerk1.jpg',cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('../../Images/nieuwekerk2.jpg',cv2.IMREAD_GRAYSCALE)
#keypoints = sift.detect(im,None)
#k_im = cv2.drawKeypoints(im, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kp1, desc1 = sift.detectAndCompute(im1, None)
kp2, desc2 = sift.detectAndCompute(im2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors.
matches = bf.match(desc1,desc2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(im1,kp1,im2,kp2,matches[:12], None, flags=2)

plt.imshow(img3)
plt.show()
