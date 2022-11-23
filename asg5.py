import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("fro.png")
height, width, _ = img.shape
print("Original Size: ",height," x ",width)
width = int(480/height*width)
height = 480
print("New Size: ",height," x ",width)
img = cv2.resize(img,(width,height))

print("Original Image")
cv2.imshow("input",img)
# cv2.imshow("output",img)
# cv2.waitKey(0)
image = img.copy()
mask = np.zeros(image.shape[:2], np.uint8)

# The array is constructed of 1 row
# and 65 columns, and all array elements are 0
# Data type for the array is np.float64 (default)
backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)

# define the Region of Interest (ROI)
# as the coordinates of the rectangle
# where the values are entered as
# (startingPoint_x, startingPoint_y, width, height)
# these coordinates are according to the input image
# it may vary for different images
rectangle = (10, 150, 450, 300)


# number of iterations = 3
# cv2.GC_INIT_WITH_RECT is used because
# of the rectangle mode is used
cv2.grabCut(image, mask, rectangle,
			backgroundModel, foregroundModel,
			3, cv2.GC_INIT_WITH_RECT)

# In the new mask image, pixels will
# be marked with four flags
# four flags denote the background / foreground
# mask is changed, all the 0 and 2 pixels
# are converted to the background
# mask is changed, all the 1 and 3 pixels
# are now the part of the foreground
# the return type is also mentioned,
# this gives us the final mask
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
# The final mask is multiplied with
# the input image to give the segmented image.
image = image * mask2[:, :, np.newaxis]

# output segmented image
cv2.imshow("output",image)
cv2.waitKey(0)

