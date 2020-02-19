import cv2
import numpy as np
import os

# Photo
# DIR_INPUT = ''
# Squear
DIR_INPUT = 'data/blob'
PATH = os.path.join(os.getcwd(), DIR_INPUT)
images = os.listdir(PATH)

img_path = os.path.join(PATH, images[0])

# read image

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# get image size

W, H = img.shape

# resize image
img = cv2.resize(img, (W // 4, H // 4))
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

