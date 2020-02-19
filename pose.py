import cv2
import numpy as np
import os

dist = np.array([[-0.02357899, 0.32398194, -0.01049933, 0.00349942, -0.2899601]], dtype=np.float32)
mtx = np.array([[551.35925517, 0., 248.33988363],
                [0., 551.84937767, 309.15983624],
                [0., 0., 1.]], dtype=np.float32)


# Photo

def get_image_point(img):
    detector = cv2.SimpleBlobDetector_create()
    # Detect the blobd in the image
    keypoints = detector.detect(img)
    # Draw Detected keypoints as red circles
    imgKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                     cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    points = np.array([[idx.pt[0], idx.pt[1]] for idx in keypoints], dtype=np.float32)
    return points


model_points = np.array([
    [0.0, 0.0, 0.0],  # Nose tip
    [1.0, 0.0, 0.0],  # Nose tip
    [-1.0, 0.0, 0.0],  # Nose tip
    [-1.0, -1.0, 0.0],  # Nose tip

], np.float32)



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

image_points = get_image_point(img)
