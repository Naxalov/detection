#!/usr/bin/env python

import cv2
import numpy as np

# Read Image
im = cv2.imread("/home/ubuntuml/development/python/detection/data/blob/00.jpg")
W, H, _ = im.shape

# resize image
im = cv2.resize(im, (H // 2, W // 2))
size = im.shape

# 2D image points. If you change the image, you need to change vector
image_points = np.array([
    [194.11012, 414.1554],
    [287.17834, 407.9679],
    [188.64015, 340.65848],
    [272.56433, 336.51465]

], dtype=np.float32)

# 3D model points.
model_points = np.array([
    [0.0, 0.0, 0.0],  # Nose tip
    [0.0, -330.0, -65.0],  # Chin
    [-225.0, 170.0, -135.0],  # Left eye left corner
    [225.0, 170.0, -135.0],  # Right eye right corne

], np.float32)

# Camera internals

focal_length = size[1]
center = (size[1] / 2, size[0] / 2)

camera_matrix = np.array([[551.35925517, 0., 248.33988363],
                          [0., 551.84937767, 309.15983624],
                          [0., 0., 1.]], dtype=np.float32)

print(
    "Camera Matrix :\n {0}".format(camera_matrix))

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,flags=)

print("Rotation Vector:\n {0}".format(rotation_vector))
print("Translation Vector:\n {0}".format(translation_vector))

# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose


(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector,
                                                 camera_matrix, dist_coeffs)

for p in image_points:
    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

p1 = (int(image_points[0][0]), int(image_points[0][1]))
p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

cv2.line(im, p1, p2, (255, 0, 0), 2)

# Display image
cv2.imshow("Output", im)
cv2.waitKey(0)
