from harris import *
import cv2

im1 = cv2.imread('../Images/son_ewi_flag1.jpg',cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('../Images/son_ewi_flag2.jpg',cv2.IMREAD_GRAYSCALE)
res1 = compute_harris_response(im1,3)
res2 = compute_harris_response(im2,3)

points1 = get_harris_points(res1,4,0.3)
d1 = get_descriptors(im1,points1,3)
points2 = get_harris_points(res2,4, 0.3)
d2 = get_descriptors(im2,points2,3)
# print "points", points1
# print "descriptors", d1
matchscores = match_twosided(d1,d2)

plt.figure()
plt.gray()
plot_matches(im1,im2,points1,points2,matchscores,True)
plt.show()