import pathlib
from typing import Iterable, Optional
import numpy as np
tau = 2*np.pi

from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

import matplotlib.pyplot as plt


def get_lines(edges: np.ndarray) -> Iterable[tuple[int, int, float]]:
    """
    Returns list of lines: (x, y, angle)
    """
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(edges, theta=tested_angles)

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        yield (x0, y0, np.tan(angle + np.pi / 2))


def get_lines_probabilistic(
    edges: np.ndarray, length: int = 5, gap: int = 3
) -> Iterable[tuple[tuple[int, int], tuple[int, int], float]]:
    """
    Return value:
    list of (start point, end point, angle with the vertical projection in radians)
    """
    for beginning, end in probabilistic_hough_line(
        edges, threshold=10, line_length=length, line_gap=gap
    ):
        x0, y0 = beginning
        x1, y1 = end
        # CAH
        angle = np.arccos(abs(y1 - y0) / np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

        yield beginning, end, angle


def display_lines(
    ax,
    image: np.ndarray,
    lines: list[tuple[int, int, float]],
    probabilistic: bool = True,
):
    """
    Display lines on top of image with matplotlib
    """
    plt.imshow(image, cmap="gray")
    midway = lambda p1, p2: ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)
    if probabilistic:
        counter = 0
        # for beginning, end, angle in lines[5:6]:
        for beginning, end, angle in lines:
            # print({
            #     "i": counter,
            #     "begin": beginning,
            #     "end": end,
            #     "angle": angle / tau * 180,
            # })
            ax.plot([beginning[0], end[0]], [beginning[1], end[1]], color="red")
            ax.plot([beginning[0], beginning[0]], [beginning[1], end[1]], color="blue")
            ax.text(*midway(beginning, end), f"{int(angle*180/tau)}°", color="white")
            counter+=1
    else:
        for *point, angle in lines:
            ax.axline(point, slope=angle)
            ax.scatter(*point)

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
