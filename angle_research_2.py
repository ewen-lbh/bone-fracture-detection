import pathlib
from typing import Optional
import numpy as np

from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

import matplotlib.pyplot as plt


def get_lines(edges: np.ndarray) -> list[tuple[int, int, float]]:
    """
    Returns list of lines: (x, y, angle)
    """
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 90, endpoint=False)
    lines = []
    h, theta, d = hough_line(edges, theta=tested_angles)

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        lines.append((x0, y0, np.tan(angle + np.pi/2)))

    return lines

def display_lines(image: np.ndarray, lines: list[tuple[int, int, float]], save: Optional[pathlib.Path] = None):
    """
    Display lines on top of image with matplotlib
    """
    plt.imshow(image, cmap="gray")
    for x, y, slope in lines:
        try:
            print(f"at {x} {y}: {image[int(x), int(y)]}")
        except IndexError:
            continue
        plt.axline((x, y), slope=slope)
        plt.scatter(x, y, c="red")
    plt.xlim(right=len(image[:, 0]))
    plt.ylim(0, len(image[0]))

    if save:
        plt.savefig(str(save))
    else:
        plt.show()
