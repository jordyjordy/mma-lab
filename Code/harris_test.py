from harris import *
import cv2

im = cv2.imread('../Images/bookshelf.jpg',cv2.IMREAD_GRAYSCALE)

res = compute_harris_response(im)
print res

points = get_harris_points(res,15,0.05)
plot_harris_points(im,points)
descriptors = get_descriptors(im,points)
