# coding: utf-8
import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks

processed = cv2.imread('datasets/various/coude_clair_2_edges.bmp')
hspaces, angles, distances = hough_line(processed)
processed = np.mean(processed, axis=2)
angless = []
for _, a, distance in zip(*hough_line_peaks(hspaces, angles, distances)):
    angless.append(a)
    
angless = [a*180/np.pi for a in angless]
