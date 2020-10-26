import cv2
import numpy as np
import os

# Photo
# DIR_INPUT = ''
# Squear
DIR_INPUT = 'data/blob'
PATH = os.path.join(os.getcwd(), DIR_INPUT)
images = os.listdir(PATH)

img_path = os.path.join(PATH, images[1])

# read image

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# get image size

W, H = img.shape

# resize image
img = cv2.resize(img, (H // 2, W // 2))
detector = cv2.SimpleBlobDetector_create()

# Detect the blobd in the image
keypoints = detector.detect(img)

# Draw Detected keypoints as red circles

imgKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                 cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Display found keypoints

cv2.imshow('Keypoints', imgKeyPoints)
cv2.waitKey(0)

cv2.destroyAllWindows()


# Iterate until x becomes 0
x = 6
while x:
    print(x)
    x -= 1
# Prints 6 5 4 3 2 1
