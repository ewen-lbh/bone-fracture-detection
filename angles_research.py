# coding: utf-8
import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import potrace
import matplotlib.pyplot as plt

processed = np.mean(cv2.imread('datasets/various/coude_clair_2_edges.bmp'), axis=2)
hspaces, angles, distances = hough_line(processed)
plt.imshow(hspaces)
plt.show()
print(hspaces)
hspaces, angles, distances = hough_line_peaks(hspaces, angles, distances)
print(hspaces)
# angless = []
# for _, a, distance in zip(*hough_line_peaks(hspaces, angles, distances)):
    # angless.append(a)

# angless = [a*180/np.pi for a in angless]

# print(angless)
