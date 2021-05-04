#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys


_, σ, threshold_low, threshold_high, = sys.argv


def detect_edges(filename: str, σ: int, low: int, high: int):
    σ, low, high = map(int, (σ, low, high))
    print(f"Using σ={σ}, high={high}, low={low}")

    image = cv2.imread(filename, 0)
    edges = cv2.Canny(image, low, high, σ)
    return image, edges


i = 0
for filename in [ 'coude_clair.png', 'baseline_coude_clair.png' ]:
    i += 1
    print(f"Detecting {filename}...")
    img, edges = detect_edges(f"datasets/various/cropped/{filename}", σ, threshold_low, threshold_high)
    plt.subplot(140 + i)
    plt.title(filename)
    plt.imshow(img, cmap='gray')
    i += 1
    plt.subplot(140 + i)
    plt.title(f"Edges of {filename}")
    plt.imshow(edges, cmap='gray')

plt.show()
