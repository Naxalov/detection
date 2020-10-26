import cv2
import numpy as np
import os

dist = np.array([[-0.02357899, 0.32398194, -0.01049933, 0.00349942, -0.2899601]], dtype=np.float32)
# dist = np.zeros((4, 1))  # Assuming no lens distortion
mtx = np.array([[551.35925517, 0., 248.33988363],
                [0., 551.84937767, 309.15983624],
                [0., 0., 1.]], dtype=np.float32)


# dist = np.array([[-6.26163541e-02, 1.10526699e+00 - 1.62313064e-02, -3.25249256e-03, -6.24637994e+00]],
#                 dtype=np.float32)
# mtx = np.array([[865.42285781, 0., 380.62296218],
#                 [0., 867.40707348, 346.83874496],
#                 [0., 0., 1.]], dtype=np.float32)
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


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


# Photo
# DIR_INPUT = ''
# Squear
DIR_INPUT = 'data/blob'
PATH = os.path.join(os.getcwd(), DIR_INPUT)
images = os.listdir(PATH)

img_path = os.path.join(PATH, images[1])

# read image

img_rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# get image size

W, H = img.shape

# resize image
img = cv2.resize(img, (H // 2, W // 2))

img_rgb = cv2.resize(img_rgb, (H // 2, W // 2))

image_points = get_image_point(img)

model_points = np.array([
    [0., 0., 0.],
    [8., 0., 0.],
    [0., 8., 0.],
    [8., 8., 0.],

], np.float32)

# image_points = np.array(
#
#     [
#         [248., 296.],
#         [497., 314.],
#         [186., 517.],
#         [486., 545.],
#     ]
#
#     , dtype=np.float32)

for i, (x, y) in enumerate(image_points.astype(int)):
    # draw the circle in the output image, then draw a rectangle
    # corresponding to the center of the circle
    cv2.putText(img_rgb, "centroid: " + str(i), (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# cv2.imshow('img', img_rgb)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, mtx, dist)

# project 3D points to image plane
imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, mtx, dist)

img = draw(img_rgb, image_points, imgpts)
cv2.imshow('img', img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(success)

print(image_points)
